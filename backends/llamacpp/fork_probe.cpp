// Step 1b of the llama.cpp gate: does FORK work where ROLLBACK failed?
//
// The server test (reports/llamacpp_recon.md §g) failed because llama-server has
// to roll a recurrent GDN state back from a long prompt to a prefix boundary.
// The engine never needs that. It needs FORK:
//
//     decode the catalog once into seq 0
//     llama_memory_seq_cp(0 -> 1..N)          <- fork at the boundary, no rollback
//     one llama_batch: N different suffixes, one per seq
//
// Pre-registered pass criteria (written before the run, Biscuit + Pooh):
//   1. tokens processed ~= len(catalog) + sum(len(suffix)), not N * len(prompt)
//   2. control: the same suffixes decoded WITHOUT fork must give the same logits
//      -- max |d log p| over the candidate set ~ 0. Fast with a wrong state is
//      worse than slow, and #22384 is open exactly here.
//   3. determinism: forking twice gives the same numbers.
// Fails any of them -> stop, report, do not build a backend.
//
// Dax, while re-checking his own batch-invariance gate, found the trap this
// probe would otherwise walk into: the FIRST run of any given batch shape
// differs from every later run. Comparing fork-run-1 against control-run-1
// cannot tell "fork carries a wrong state" apart from "first run of a new
// shape". So each branch runs TWICE and the verdict uses the SECOND runs;
// the first-vs-first number is printed too, only to show the effect.
//
// THIRD ARM (added after the CPU dry run): fork and the sequential control
// differ in TWO ways at once -- fork forks, and fork puts 8 sequences in one
// batch while the control decodes one prompt at a time. The observed gap
// (0.245 log p) is the same order as the batched-vs-single gap Dax measured on
// vLLM (0.2425), so "seq_cp lost the recurrent state" and "batching changes the
// arithmetic" both predict it. Arm C decodes the 8 FULL prompts in ONE batch,
// no fork. Then:
//     C vs B  = batching effect alone (both full decodes)
//     A vs C  = fork effect, but the batch sizes still differ (749 vs 2517)
//
// FOURTH/FIFTH ARM (Biscuit): the clean isolation I wrongly called impossible.
//     A'' : decode catalog in seq 0, seq_cp 0->1, decode ONE suffix in seq 1
//     B'' : decode catalog in seq 1, then decode the SAME suffix in seq 1
// Two consecutive decodes of identical sizes in both arms, single sequence,
// same seq id at the scored position. The ONLY difference is whether the state
// travelled through seq_cp. Delta = 0 means seq_cp carries the GDN state
// correctly; delta != 0 is the number for upstream #22384.
//
// Build: see backends/llamacpp/README.md

#include "llama.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;
static double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

struct Tokens {
    std::vector<llama_token> prefix;
    std::vector<std::vector<llama_token>> suffix;
    std::vector<llama_token> cand;
};

static bool load_tokens(const char * path, Tokens & t) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return false; }
    int np = 0, ns = 0, nc = 0;
    f >> np >> ns >> nc;
    t.prefix.resize(np);
    for (auto & x : t.prefix) f >> x;
    t.suffix.resize(ns);
    for (auto & s : t.suffix) {
        int len = 0; f >> len; s.resize(len);
        for (auto & x : s) f >> x;
    }
    t.cand.resize(nc);
    for (auto & x : t.cand) f >> x;
    return (bool) f;
}

// log-softmax, then pick the candidates
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

static double max_abs_diff(const std::vector<double> & a, const std::vector<double> & b) {
    double d = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++) d = std::max(d, std::fabs(a[i] - b[i]));
    return d;
}

// Decode one batch; `outs` receives the logits index of each requested output.
static bool decode_seqs(llama_context * ctx,
                        const std::vector<std::vector<llama_token>> & seqs,
                        const std::vector<llama_seq_id> & seq_ids,
                        const std::vector<llama_pos> & start_pos,
                        bool want_last_logits,
                        std::vector<int> & outs) {
    int total = 0;
    for (auto & s : seqs) total += (int) s.size();
    llama_batch b = llama_batch_init(total, 0, 1);
    outs.assign(seqs.size(), -1);
    for (size_t k = 0; k < seqs.size(); k++) {
        for (size_t j = 0; j < seqs[k].size(); j++) {
            const bool last = (j + 1 == seqs[k].size());
            const int i = b.n_tokens;
            b.token[i]    = seqs[k][j];
            b.pos[i]      = start_pos[k] + (llama_pos) j;
            b.n_seq_id[i] = 1;
            b.seq_id[i][0] = seq_ids[k];
            b.logits[i]   = (want_last_logits && last) ? 1 : 0;
            // llama_get_logits_ith() indexes by BATCH position, not by a running
            // count of requested outputs -- it validates batch.logits[i] == true.
            if (b.logits[i]) outs[k] = i;
            b.n_tokens++;
        }
    }
    const int rc = llama_decode(ctx, b);
    llama_batch_free(b);
    if (rc != 0) { fprintf(stderr, "llama_decode failed: %d\n", rc); return false; }
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.gguf> <fork_tokens.txt> [n_gpu_layers]\n", argv[0]);
        fprintf(stderr, "  n_gpu_layers 0 = CPU only; run that first, it needs no GPU lock\n");
        return 2;
    }
    const int ngl = (argc > 3) ? atoi(argv[3]) : 99;
    Tokens t;
    if (!load_tokens(argv[2], t)) return 2;
    const int N = (int) t.suffix.size();
    const int P = (int) t.prefix.size();
    int sum_suf = 0;
    for (auto & s : t.suffix) sum_suf += (int) s.size();
    int sum_full = 0;
    for (auto & s : t.suffix) sum_full += P + (int) s.size();

    printf("catalog prefix      : %d tokens\n", P);
    printf("states              : %d, suffix total %d\n", N, sum_suf);
    printf("fork should process : %d  (no sharing would be %d)\n\n", P + sum_suf, sum_full);

    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fprintf(stderr, "model load failed\n"); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    auto cp = llama_context_default_params();
    cp.n_ctx           = 8192;
    cp.n_batch         = 4096;
    cp.n_ubatch        = 4096;
    cp.n_seq_max       = N + 1;
    cp.type_k          = GGML_TYPE_Q8_0;
    cp.type_v          = GGML_TYPE_Q8_0;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "ctx init failed\n"); return 1; }
    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<llama_seq_id> ids(N);
    std::vector<llama_pos>    pos(N, (llama_pos) P);
    std::vector<int>          outs;
    for (int i = 0; i < N; i++) ids[i] = i + 1;

    // ---------------------------------------------------------------- FORK --
    std::vector<std::vector<double>> fork_lp(N), fork_lp2(N);
    double t_prefix = 0, t_batch = 0, t_batch2 = 0;
    for (int round = 0; round < 2; round++) {          // round 1 = determinism
        for (int i = 0; i <= N; i++) llama_memory_seq_rm(mem, i, -1, -1);

        auto t0 = clk::now();
        std::vector<std::vector<llama_token>> one{t.prefix};
        std::vector<llama_seq_id> z{0};
        std::vector<llama_pos>    p0{0};
        if (!decode_seqs(ctx, one, z, p0, false, outs)) return 1;
        const double tp = ms_since(t0);

        auto t1 = clk::now();
        for (int i = 0; i < N; i++) llama_memory_seq_cp(mem, 0, ids[i], -1, -1);
        if (!decode_seqs(ctx, t.suffix, ids, pos, true, outs)) return 1;
        const double tb = ms_since(t1);

        for (int i = 0; i < N; i++) {
            const float * lg = llama_get_logits_ith(ctx, outs[i]);
            if (!lg) { fprintf(stderr, "no logits for seq %d\n", i); return 1; }
            (round == 0 ? fork_lp : fork_lp2)[i] = cand_logp(lg, n_vocab, t.cand);
        }
        if (round == 0) { t_prefix = tp; t_batch = tb; } else { t_batch2 = tb; }
    }
    printf("FORK   catalog decode %7.1f ms | batch of %d suffixes %7.1f ms | total %7.1f ms\n",
           t_prefix, N, t_batch, t_prefix + t_batch);

    // ------------------------------------------------------------- CONTROL --
    // Same suffixes, no fork: every sequence decodes catalog+suffix from zero.
    std::vector<std::vector<double>> ctl_lp(N), ctl_lp2(N);
    double t_ctl = 0;
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i <= N; i++) llama_memory_seq_rm(mem, i, -1, -1);
        auto t2 = clk::now();
        for (int i = 0; i < N; i++) {
            std::vector<llama_token> full = t.prefix;
            full.insert(full.end(), t.suffix[i].begin(), t.suffix[i].end());
            std::vector<std::vector<llama_token>> one{full};
            std::vector<llama_seq_id> s{ids[i]};
            std::vector<llama_pos>    p0{0};
            if (!decode_seqs(ctx, one, s, p0, true, outs)) return 1;
            const float * lg = llama_get_logits_ith(ctx, outs[0]);
            if (!lg) { fprintf(stderr, "no control logits for seq %d\n", i); return 1; }
            (round == 0 ? ctl_lp : ctl_lp2)[i] = cand_logp(lg, n_vocab, t.cand);
            llama_memory_seq_rm(mem, ids[i], -1, -1);
        }
        if (round == 1) t_ctl = ms_since(t2);   // time the settled run, like fork
    }

    // ------------------------------------------------ ARM C: batched, no fork --
    std::vector<std::vector<double>> bat_lp(N), bat_lp2(N);
    double t_bat = 0;
    {
        std::vector<std::vector<llama_token>> fulls(N);
        for (int i = 0; i < N; i++) {
            fulls[i] = t.prefix;
            fulls[i].insert(fulls[i].end(), t.suffix[i].begin(), t.suffix[i].end());
        }
        std::vector<llama_pos> zeros(N, 0);
        for (int round = 0; round < 2; round++) {
            for (int i = 0; i <= N; i++) llama_memory_seq_rm(mem, i, -1, -1);
            auto t3 = clk::now();
            if (!decode_seqs(ctx, fulls, ids, zeros, true, outs)) return 1;
            if (round == 1) t_bat = ms_since(t3);
            for (int i = 0; i < N; i++) {
                const float * lg = llama_get_logits_ith(ctx, outs[i]);
                if (!lg) { fprintf(stderr, "no batched logits for seq %d\n", i); return 1; }
                (round == 0 ? bat_lp : bat_lp2)[i] = cand_logp(lg, n_vocab, t.cand);
            }
        }
    }
    printf("BATCH  %d full decodes in ONE batch (no fork)            %7.1f ms\n\n", N, t_bat);
    printf("CTRL   %d full decodes, no sharing                      %7.1f ms\n\n", N, t_ctl);

    // --------------------------------- ARMS A''/B'': fork isolated, one suffix --
    // Identical compute shapes; the only difference is the seq_cp hop.
    std::vector<std::vector<double>> a2(N), b2(N);
    for (int round = 0; round < 2; round++) {
        for (int i = 0; i < N; i++) {
            // A'': catalog in seq 0, fork to seq 1, suffix in seq 1
            for (int k = 0; k <= N; k++) llama_memory_seq_rm(mem, k, -1, -1);
            {
                std::vector<std::vector<llama_token>> pre{t.prefix};
                std::vector<llama_seq_id> s0{0};
                std::vector<llama_pos>    p0{0};
                if (!decode_seqs(ctx, pre, s0, p0, false, outs)) return 1;
                llama_memory_seq_cp(mem, 0, 1, -1, -1);
                std::vector<std::vector<llama_token>> suf{t.suffix[i]};
                std::vector<llama_seq_id> s1{1};
                std::vector<llama_pos>    p1{(llama_pos) P};
                if (!decode_seqs(ctx, suf, s1, p1, true, outs)) return 1;
                const float * lg = llama_get_logits_ith(ctx, outs[0]);
                if (!lg) { fprintf(stderr, "A'' no logits seq %d\n", i); return 1; }
                if (round == 1) a2[i] = cand_logp(lg, n_vocab, t.cand);
            }
            // B'': catalog in seq 1, then suffix in seq 1 -- no seq_cp
            for (int k = 0; k <= N; k++) llama_memory_seq_rm(mem, k, -1, -1);
            {
                std::vector<std::vector<llama_token>> pre{t.prefix};
                std::vector<llama_seq_id> s1{1};
                std::vector<llama_pos>    p0{0};
                if (!decode_seqs(ctx, pre, s1, p0, false, outs)) return 1;
                std::vector<std::vector<llama_token>> suf{t.suffix[i]};
                std::vector<llama_pos>    p1{(llama_pos) P};
                if (!decode_seqs(ctx, suf, s1, p1, true, outs)) return 1;
                const float * lg = llama_get_logits_ith(ctx, outs[0]);
                if (!lg) { fprintf(stderr, "B'' no logits seq %d\n", i); return 1; }
                if (round == 1) b2[i] = cand_logp(lg, n_vocab, t.cand);
            }
        }
    }
    double w_seqcp = 0.0;
    for (int i = 0; i < N; i++) w_seqcp = std::max(w_seqcp, max_abs_diff(a2[i], b2[i]));
    printf("SEQ_CP ISOLATED  A''(fork,1 seq) vs B''(chunked,same seq, no fork)\n");
    printf("  same shapes, same seq id, only difference = the seq_cp hop\n");
    printf("  max |d log p| = %.8f   -> %s\n\n", w_seqcp,
           w_seqcp < 1e-9 ? "seq_cp CARRIES the GDN state exactly"
                          : "seq_cp changes the state (this is the #22384 number)");

    // ------------------------------------------------------------- VERDICT --
    double w_AC = 0.0, w_CB = 0.0, w_AB = 0.0, w_det = 0.0;
    printf("seq   A-vs-C (fork)     C-vs-B (batching)   A-vs-B (both)     A run1-vs-run2\n");
    for (int i = 0; i < N; i++) {
        const double ac = max_abs_diff(fork_lp2[i], bat_lp2[i]);   // fork alone
        const double cb = max_abs_diff(bat_lp2[i],  ctl_lp2[i]);   // batching alone
        const double ab = max_abs_diff(fork_lp2[i], ctl_lp2[i]);   // confounded
        const double dd = max_abs_diff(fork_lp[i],  fork_lp2[i]);
        w_AC = std::max(w_AC, ac); w_CB = std::max(w_CB, cb);
        w_AB = std::max(w_AB, ab); w_det = std::max(w_det, dd);
        printf("%3d  %16.8f   %16.8f   %16.8f   %16.8f\n", i, ac, cb, ab, dd);
    }
    printf("\nA fork(batched) vs C full(batched) -- FORK ALONE, THE VERDICT : %.8f\n", w_AC);
    printf("C full(batched) vs B full(sequential) -- BATCHING ALONE        : %.8f\n", w_CB);
    printf("A vs B (what a 2-arm test would have reported, confounded)     : %.8f\n", w_AB);
    printf("determinism, fork run1 vs run2                                 : %.8f\n", w_det);
    // State correctness is decided by the ISOLATED arms (A'' vs B''), never by
    // A-vs-C: those two differ in batch size as well as in forking, so A-vs-C
    // reports the batching effect and would falsely convict seq_cp.
    const double worst = w_seqcp;
    const double worst_det = w_det;
    printf("\nspeedup fork vs sequential full : %.2fx\n", t_ctl / (t_prefix + t_batch));
    printf("speedup fork vs batched  full   : %.2fx\n", t_bat / (t_prefix + t_batch));

    // Cross-SESSION determinism: vLLM's batched path fails this (Dax: two
    // identical warm batched runs matched on 29/307 options, max 0.2425) and
    // VLLM_BATCH_INVARIANT=1 is refused for GDN_ATTN, so there is currently no
    // reproducible batched path on vLLM at all. If llama.cpp's fork is stable
    // across processes that matters more than the speedup. Diff these lines
    // between two separate runs of this binary.
    // Dax's lesson from his own permutation bug: when a run produces several
    // variants of the same thing, the strongest check lives INSIDE the result --
    // assert the variants differ. If two sequences came back identical, the batch
    // is reading one sequence's logits for several outputs, or seq_cp overwrote a
    // neighbour. Every comparison above would still look perfect in that case.
    int dup = 0;
    for (int i = 0; i < N && !dup; i++)
        for (int j = i + 1; j < N && !dup; j++)
            if (max_abs_diff(fork_lp2[i], fork_lp2[j]) == 0.0) dup = 1;
    printf("distinct outputs across the %d forked sequences : %s\n", N,
           dup ? "NO  <-- two sequences returned identical logits, the batch is wrong"
               : "yes");

    printf("\nRAW fork run2 candidate log p (diff these across two processes):\n");
    for (int i = 0; i < N; i++) {
        printf("RAW %d", i);
        for (double v : fork_lp2[i]) printf(" %.9f", v);
        printf("\n");
    }

    const bool ok_state = worst < 1e-9;         // A'' vs B'': must be exact
    const bool ok_det   = worst_det < 1e-9;
    printf("\nPASS state correctness (A'' vs B'') : %s\n",
           ok_state ? "yes -- seq_cp carries the GDN state exactly"
                    : "NO  -- seq_cp changed the state");
    printf("PASS determinism       : %s\n", ok_det ? "yes" : "no");

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return (ok_state && !dup) ? 0 : 1;
}
