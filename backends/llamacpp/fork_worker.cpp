// Resident fork worker for the Jev engine's BonsaiLLM backend.
//
// Two-level fork, and it never truncates a sequence:
//
//     seq 0..K-1       = catalog slots        (K of them, Python owns the policy)
//     seq K            = seq_cp(slot) + state (one call)
//     seq K+1..K+N     = seq_cp(K) + node tail, all decoded in ONE batch
//
// K slots rather than one because callers alternate question sets: a caller that
// classifies with one catalogue and then decides with another touches both on
// every turn. With a single slot each call evicted the other one's catalogue and
// cost 930 ms against 274 ms on the constant-catalogue path.
//
// Why not "keep seq 0 and extend it": the longest common prefix WITHIN a call
// contains the state, so the next call (different state) never extends it --
// measured at LCP 1058 tokens but only 535 shared across calls. And the
// obvious repair, seq_rm(0, p, -1) to trim seq 0 back to the shared part, is not
// safe on this architecture: the model is hybrid GDN, its recurrent state is not
// addressable by position, so dropping a suffix cannot restore the state at p
// unless the fork's snapshot ring is on and exact (untested -- the fork ships
// examples/rs-rollback because this is delicate). Both hops above are seq_cp,
// which measured exactly 0.00000000 on this model (runs/bonsai_fork.log).
//
// This is deliberately dumb: the Python side owns prefix learning, batching and
// every policy decision. Kept separate from fork_probe.cpp, which produced the
// published gate numbers and is left untouched so the gate stays re-runnable.
//
// Protocol -- whitespace separated, one command per message, replies flushed:
//   CAT <slot> <n> <tok...>               -> CATOK <slot> <n> <ms>
//   RUN <slot> <n_work> <work...> <n_seq>
//       (<n_tail> <tail...> <n_cand> <cand...>) x n_seq
//                                         -> RES <n_seq>
//                                            LP <i> <n> <logprob...>   x n_seq
//                                            DONE <reused> <decoded> <ms>
//   PING -> PONG ;  QUIT -> exits 0 ;  anything wrong -> ERR <what> and exit 1
//
// Logprobs are log-softmax over the WHOLE vocabulary, then the requested ids are
// picked out. Never renormalize over the requested ids here: the engine derives
// in_set_mass from exactly these numbers, and a local renormalization would pin
// it at 1.0 and silently blind the format-bug check (AGENTS 4.7).
//
// Build: see backends/llamacpp/README.md.
#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

// the effective n_batch, so decode_seqs can refuse an oversized batch instead of
// letting llama.cpp's assert abort the process
static int g_n_batch = 0;

static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

static void die(const std::string & what) {
    printf("ERR %s\n", what.c_str());
    fflush(stdout);
    fprintf(stderr, "fork_worker: %s\n", what.c_str());
    exit(1);
}

// log-softmax over the full vocab, then pick the candidates
static std::vector<double> cand_logp(const float * logits, int n_vocab,
                                     const std::vector<llama_token> & cand) {
    double mx = -1e30;
    for (int i = 0; i < n_vocab; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; i++) sum += std::exp((double) logits[i] - mx);
    const double lse = mx + std::log(sum);
    std::vector<double> out;
    out.reserve(cand.size());
    for (auto c : cand) out.push_back((double) logits[c] - lse);
    return out;
}

// Decode a set of sequences in one batch. `outs` receives the logits index of
// each requested output (llama_get_logits_ith indexes by BATCH position).
static bool decode_seqs(llama_context * ctx,
                        const std::vector<std::vector<llama_token>> & seqs,
                        const std::vector<llama_seq_id> & seq_ids,
                        const std::vector<llama_pos> & start_pos,
                        bool want_last_logits,
                        std::vector<int> & outs) {
    int total = 0;
    for (auto & s : seqs) total += (int) s.size();
    if (total == 0) return true;
    if (total > g_n_batch) {
        fprintf(stderr, "batch of %d tokens over n_batch %d\n", total, g_n_batch);
        return false;
    }
    outs.clear();
    llama_batch b = llama_batch_init(total, 0, 1);
    int i = 0;
    for (size_t k = 0; k < seqs.size(); k++) {
        for (size_t j = 0; j < seqs[k].size(); j++) {
            b.token[i]       = seqs[k][j];
            b.pos[i]         = start_pos[k] + (llama_pos) j;
            b.n_seq_id[i]    = 1;
            b.seq_id[i][0]   = seq_ids[k];
            const bool last  = (j + 1 == seqs[k].size());
            b.logits[i]      = (want_last_logits && last) ? 1 : 0;
            if (want_last_logits && last) outs.push_back(i);
            i++;
        }
    }
    b.n_tokens = total;
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) {
        fprintf(stderr, "llama_decode failed: %d (n_tokens=%d)\n", rc, total);
        return false;
    }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [n_ctx] [n_seq_max] [ngl] [n_batch] "
                "[n_slots] [n_ubatch]\n", argv[0]);
        return 2;
    }
    const int n_ctx     = (argc > 2) ? atoi(argv[2]) : 16384;
    const int n_seq_max = (argc > 3) ? atoi(argv[3]) : 8;    // 0=catalog, 1=state, rest nodes
    const int ngl       = (argc > 4) ? atoi(argv[4]) : 99;
    int n_batch         = (argc > 5) ? atoi(argv[5]) : 2048;
    const int n_slots   = (argc > 6) ? atoi(argv[6]) : 4;
    int n_ubatch        = (argc > 7) ? atoi(argv[7]) : 0;   // 0 = follow n_batch
    // llama.cpp divides n_ctx across the sequences: each one gets n_ctx/n_seq_max
    // cells, NOT n_ctx. Asking for 16 sequences in 4096 gave 256 cells each and
    // a 1583-token prompt died with "find_slot: n_tokens = 1583 > size = 256".
    const int per_seq = n_ctx / (n_seq_max > 0 ? n_seq_max : 1);
    // A single decode can be as long as one sequence's whole prompt, so n_batch
    // below per_seq is not a tuning choice, it is a crash: llama.cpp asserts
    // n_tokens_all <= n_batch and aborts the process, taking the worker with it.
    if (n_batch < per_seq) n_batch = per_seq;
    if (n_batch > n_ctx)   n_batch = n_ctx;
    // n_batch is a correctness bound -- one decode may carry a whole sequence's
    // prompt -- while n_ubatch only sizes the compute buffer, and llama.cpp
    // splits a batch into micro-batches for us. Tying them together cost 4.2 GiB
    // of VRAM for nothing.
    if (n_ubatch <= 0 || n_ubatch > n_batch) n_ubatch = n_batch;
    if (n_slots < 1 || n_slots + 2 > n_seq_max) {
        fprintf(stderr, "n_slots %d needs n_seq_max >= %d (got %d)\n",
                n_slots, n_slots + 2, n_seq_max);
        return 2;
    }
    const llama_seq_id seq_work = (llama_seq_id) n_slots;   // slots are 0..n_slots-1

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) die("model load failed");
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    // n_batch follows the largest single decode, not n_ctx: the compute buffer
    // is sized from n_ubatch, and n_ubatch = n_ctx cost 4.2 GiB of VRAM for
    // nothing
    cp.n_batch         = n_batch;
    cp.n_ubatch        = n_ubatch;
    cp.n_seq_max       = n_seq_max;
    cp.type_k          = GGML_TYPE_Q8_0;
    cp.type_v          = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) die("ctx init failed");
    g_n_batch = n_batch;
    llama_memory_t mem = llama_get_memory(ctx);

    printf("READY %d %d %d %d %d %d %d\n",
           n_vocab, n_ctx, n_seq_max, per_seq, n_batch, n_slots, n_ubatch);
    fflush(stdout);

    std::vector<int> slot_len((size_t) n_slots, 0);
    std::vector<int> outs;
    std::string cmd;

    while (std::cin >> cmd) {
        if (cmd == "QUIT") break;

        if (cmd == "PING") { printf("PONG\n"); fflush(stdout); continue; }

        if (cmd == "CAT") {
            int slot = 0, n = 0;
            if (!(std::cin >> slot) || slot < 0 || slot >= n_slots) die("CAT: bad slot");
            if (!(std::cin >> n) || n < 0) die("CAT: bad length");
            std::vector<llama_token> cat((size_t) n);
            for (int i = 0; i < n; i++) if (!(std::cin >> cat[i])) die("CAT: short read");
            auto t0 = clk::now();
            // only this slot is rebuilt; the other catalogues stay resident, which
            // is the whole point of having more than one
            llama_memory_seq_rm(mem, (llama_seq_id) slot, -1, -1);
            slot_len[(size_t) slot] = 0;
            if (n > 0) {
                if (n > per_seq)
                    die("CAT: catalog " + std::to_string(n) + " over per-sequence "
                        "capacity " + std::to_string(per_seq));
                std::vector<std::vector<llama_token>> one{cat};
                std::vector<llama_seq_id> z{(llama_seq_id) slot};
                std::vector<llama_pos>    p0{0};
                if (!decode_seqs(ctx, one, z, p0, false, outs)) die("CAT: decode failed");
                slot_len[(size_t) slot] = n;
            }
            printf("CATOK %d %d %.3f\n", slot, slot_len[(size_t) slot], ms_since(t0));
            fflush(stdout);
            continue;
        }

        if (cmd == "RUN") {
            auto t0 = clk::now();
            int slot = 0, n_work = 0, n_seq = 0;
            if (!(std::cin >> slot) || slot < 0 || slot >= n_slots) die("RUN: bad slot");
            if (!(std::cin >> n_work) || n_work < 0) die("RUN: bad work length");
            std::vector<llama_token> work((size_t) n_work);
            for (int i = 0; i < n_work; i++) if (!(std::cin >> work[i])) die("RUN: short work");
            if (!(std::cin >> n_seq) || n_seq <= 0) die("RUN: bad n_seq");
            // slots 0..n_slots-1, the work sequence at n_slots, nodes after it
            if (n_seq + n_slots + 1 > n_seq_max)
                die("RUN: n_seq " + std::to_string(n_seq) + " with " +
                    std::to_string(n_slots) + " slots needs n_seq_max " +
                    std::to_string(n_seq + n_slots + 1) + ", worker started with " +
                    std::to_string(n_seq_max));

            std::vector<std::vector<llama_token>> tails((size_t) n_seq);
            std::vector<std::vector<llama_token>> cands((size_t) n_seq);
            int sum_tail = 0;
            for (int s = 0; s < n_seq; s++) {
                int nt = 0;
                if (!(std::cin >> nt) || nt <= 0)
                    die("RUN: tail " + std::to_string(s) + " must be non-empty "
                        "(the scored position lives in the tail)");
                tails[s].resize((size_t) nt);
                for (int i = 0; i < nt; i++)
                    if (!(std::cin >> tails[s][i])) die("RUN: short tail");
                sum_tail += nt;
                int nc = 0;
                if (!(std::cin >> nc) || nc < 0) die("RUN: bad cand count");
                cands[s].resize((size_t) nc);
                for (int i = 0; i < nc; i++) {
                    if (!(std::cin >> cands[s][i])) die("RUN: short cand");
                    if (cands[s][i] < 0 || cands[s][i] >= n_vocab) die("RUN: cand out of vocab");
                }
            }
            // every node sequence holds catalog+state+its own tail, and the
            // budget that binds is per-sequence, not the total
            int longest = 0;
            for (auto & t : tails) longest = (int) t.size() > longest ? (int) t.size() : longest;
            const int catalog_len = slot_len[(size_t) slot];
            if (catalog_len + n_work + longest > per_seq)
                die("RUN: a sequence needs " +
                    std::to_string(catalog_len + n_work + longest) +
                    " cells, per-sequence capacity is " + std::to_string(per_seq) +
                    " (n_ctx " + std::to_string(n_ctx) + " / n_seq_max " +
                    std::to_string(n_seq_max) + ")");

            // fresh working sequences; the catalogue slots are never touched
            for (int i = n_slots; i < n_seq_max; i++) llama_memory_seq_rm(mem, i, -1, -1);
            if (catalog_len > 0)
                llama_memory_seq_cp(mem, (llama_seq_id) slot, seq_work, -1, -1);

            if (n_work > 0) {
                std::vector<std::vector<llama_token>> one{work};
                std::vector<llama_seq_id> s1{seq_work};
                std::vector<llama_pos>    p{(llama_pos) catalog_len};
                if (!decode_seqs(ctx, one, s1, p, false, outs)) die("RUN: state decode failed");
            }

            std::vector<llama_seq_id> ids((size_t) n_seq);
            std::vector<llama_pos>    pos((size_t) n_seq, (llama_pos) (catalog_len + n_work));
            for (int s = 0; s < n_seq; s++) {
                ids[s] = (llama_seq_id) (n_slots + 1 + s);
                llama_memory_seq_cp(mem, seq_work, ids[s], -1, -1);
            }
            if (!decode_seqs(ctx, tails, ids, pos, true, outs)) die("RUN: tail decode failed");
            if ((int) outs.size() != n_seq) die("RUN: logits index count mismatch");

            printf("RES %d\n", n_seq);
            for (int s = 0; s < n_seq; s++) {
                const float * lg = llama_get_logits_ith(ctx, outs[s]);
                if (!lg) die("RUN: no logits for seq " + std::to_string(s));
                const std::vector<double> lp = cand_logp(lg, n_vocab, cands[s]);
                printf("LP %d %d", s, (int) lp.size());
                for (double v : lp) printf(" %.9f", v);
                printf("\n");
            }
            printf("DONE %d %d %.3f\n", catalog_len, n_work + sum_tail, ms_since(t0));
            fflush(stdout);
            continue;
        }

        die("unknown command '" + cmd + "'");
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
