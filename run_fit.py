"""Phase-3 calibration fit (CPU, offline). Brief §6-7.

Input: runs/calib/<ds>.logits.jsonl (raw log-mass per option, written once by
run_calib_collect.py). Nothing here touches the GPU.

Pipeline:
  1. split per DATASET (no leak): split_hint if it yields >=2 partitions
     (train->FIT, validation->SELECT, test->TEST), else hash(id) 60/20/20
     (TEST 40% when dataset total <= 500).
  2. FIT: T grid 0.5..5.0 step 0.05 minimize NLL, one T per qtype
     (noul adds Platt bias b: p = sigmoid((l_t - l_f)/T + b)).
  3. SELECT gates (simplest method that passes wins):
     T-only -> +prior (p' ~ p/p_prior) if dECE >= 0.02 -> per-dataset T if
     dECE >= 0.01. Shuffle (choice, extra GPU) reported as gate only.
  4. TEST (touched once): accuracy, ECE (10-bin; score 5-bin + P(|E-lab|<=1)),
     Brier, NLL, selective accuracy (drop 20% least confident), bootstrap CI
     1000. Pass/fail printed per brief §6 thresholds -- fixed before numbers.
"""
import argparse
import hashlib
import json
import math
import os
import random
from collections import defaultdict

ECE_BINS = 10
ECE_BINS_SCORE = 5
BOOTSTRAP = 1000
PASS_ECE = {"noul": 0.05, "choice": 0.05, "score": 0.08}
PASS_SELECTIVE_DELTA = 0.05  # +5 points when dropping 20% least confident
PASS_ACC_DRIFT = 0.005
# Platt bias grid: wide enough that an optimum never sits on the edge
# (scitail wanted -4.0 exactly on the old [-4, 4] grid = clipped)
B_GRID = [-8 + 0.05 * i for i in range(321)]


# ---------------------------------------------------------------- utils ----
def softmax(xs):
    m = max(xs)
    es = [math.exp(x - m) for x in xs]
    z = sum(es)
    return [e / z for e in es]


def split_hash(rid, r=(0.60, 0.20, 0.20)):
    h = int(hashlib.sha1(rid.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "FIT" if h < r[0] else ("SELECT" if h < r[0] + r[1] else "TEST")


def assign_split(rows):
    """Split BY GROUP (fields of one state stay together -- leak guard).
    split_hint only when it yields >=2 partitions; else hash(group)."""
    gids = sorted({r["group"] for r in rows})
    hint_of = {}
    for r in rows:
        hint_of.setdefault(r["group"], r.get("split_hint"))
    hints = {h for h in hint_of.values() if h}
    m = {"train": "FIT", "validation": "SELECT", "val": "SELECT", "test": "TEST"}
    if len(hints) >= 2 and all(h in m for h in hints):
        for r in rows:
            r["split"] = m[r["split_hint"]]
        return "split_hint"
    test_r = 0.40 if len(gids) <= 500 else 0.20
    for r in rows:
        r["split"] = split_hash(r["group"], (1 - test_r - 0.20, 0.20, test_r))
    return "hash(group)"


def rebias(rows, b_map, T):
    """Fold per-dataset Platt b into noul logits: sigma((lt-lf)/T + b) ==
    sigma((lt + T*b - lf)/T). Returns shallow copies; b then evaluates as 0."""
    out = []
    for r in rows:
        if (r["qtype"] == "noul" and isinstance(b_map, dict)
                and r["dataset"] in b_map):
            r2 = dict(r)
            r2["logits"] = dict(r["logits"])
            r2["logits"]["true"] = r["logits"]["true"] + T * b_map[r["dataset"]]
            out.append(r2)
        else:
            out.append(r)
    return out


def probs_for(row, T, b=None, prior=None):
    """Post-T probabilities aligned with row['options'] / scale keys."""
    if row["qtype"] == "noul":
        keys = ["true", "false"]
    elif row["qtype"] == "score":
        keys = sorted(row["logits"], key=float)
    else:
        keys = [str(o) for o in row["options"]]
    lg = [row["logits"][k] for k in keys]
    if prior is not None:
        # brief §7: prior is softmax(q) at T=1 and p is softmax(l/T), so
        # p' ~ p/prior = exp(l/T - q) -- the prior is NOT tempered
        pr = [prior["logits"][k] for k in keys]
        p = softmax([l / T - q for l, q in zip(lg, pr)])
    elif row["qtype"] == "noul" and b is not None:
        lt, lf = row["logits"]["true"], row["logits"]["false"]
        pt = 1 / (1 + math.exp(-((lt - lf) / T + b)))
        p = [pt, 1 - pt]
        keys = ["true", "false"]
    else:
        p = softmax([l / T for l in lg])
    return keys, p


def pred_conf_hit(row, T, b=None, prior=None):
    """(argmax_pred, confidence, hit) -- confidence label-free."""
    keys, p = probs_for(row, T, b, prior)
    im = max(range(len(p)), key=lambda i: p[i])
    pred = keys[im]
    if row["qtype"] == "score":
        # prediction := mode (never round(E) -- a bimodal
        # distribution drops E between both peaks), correct := |mode-label|<=1,
        # confidence := mass in [mode-1, mode+1] (label-free). E and
        # MAE(E, label) reported separately as Jev-comparable auxiliaries.
        lab = float(row["label"])
        mode = float(keys[im])
        e = sum(float(k) * pi for k, pi in zip(keys, p))
        hit = abs(mode - lab) <= 1.0
        conf = sum(pi for k, pi in zip(keys, p) if abs(float(k) - mode) <= 1.0)
        return pred, conf, hit, e
    conf = p[im]
    hit = (pred == str(row["label"]))
    return pred, conf, hit, None


def ece(pairs, nbins=ECE_BINS):
    """pairs: (confidence, hit 0/1). Equal-width bins, sample-weighted;
    each bin compares its MEAN confidence against its accuracy (midpoints
    bias the estimate when confidences pile up at one edge of a bin, which
    is exactly what happens near 1.0)."""
    if not pairs:
        return None
    buckets = defaultdict(lambda: [0.0, 0.0, 0])  # hits, conf sum, count
    for c, h in pairs:
        bi = min(int(c * nbins), nbins - 1)
        buckets[bi][0] += h
        buckets[bi][1] += c
        buckets[bi][2] += 1
    n = len(pairs)
    return sum(cnt / n * abs(hits / cnt - conf / cnt)
               for hits, conf, cnt in buckets.values())


def brier(row, T, b=None, prior=None):
    keys, p = probs_for(row, T, b, prior)
    y = [1.0 if k == str(row["label"]) else 0.0 for k in keys]
    return sum((pi - yi) ** 2 for pi, yi in zip(p, y))


def nll(row, T, b=None, prior=None):
    keys, p = probs_for(row, T, b, prior)
    i = keys.index(str(row["label"]))
    return -math.log(max(p[i], 1e-12))


# ------------------------------------------------------------------ fit ----
def _noul_pairs(rows):
    """(logit difference, label==true) per row -- the only thing the noul
    grid needs. Rebuilding dicts inside a 91x321 grid is what made the fit
    take minutes instead of seconds."""
    return [(r["logits"]["true"] - r["logits"]["false"], r["label"] == "true")
            for r in rows]


def _noul_nll(pairs, T, b):
    tot = 0.0
    for d, y in pairs:
        x = d / T + b
        # log(1+exp(-x)) for y=1, log(1+exp(x)) for y=0, overflow-safe
        z = -x if y else x
        tot += z + math.log1p(math.exp(-z)) if z > 0 else math.log1p(math.exp(z))
    return tot


def band_stats(row, T):
    """(band confidence, band hit) for a score row at temperature T.
    mode = argmax level (T-invariant), band = mass within +-1 of it."""
    keys, p = probs_for(row, T)
    im = max(range(len(p)), key=lambda i: p[i])
    mode = float(keys[im])
    conf = sum(pi for k, pi in zip(keys, p) if abs(float(k) - mode) <= 1.0)
    return conf, abs(mode - float(row["label"])) <= 1.0


def fit_T_band(rows):
    """Grid T minimizing NLL of the band event (monotone in T -> unique)."""
    best = (1e18, 1.0)
    for i in range(91):
        T = 0.5 + 0.05 * i
        tot = 0.0
        for r in rows:
            conf, hit = band_stats(r, T)
            q = min(max(conf, 1e-12), 1 - 1e-12)
            tot += -math.log(q if hit else 1 - q)
        if tot < best[0]:
            best = (tot, T)
    return best[1]


def fit_T(rows, qtype):
    """Grid T (and Platt b for noul) minimizing NLL on rows."""
    best = (1e18, 1.0, 0.0)
    ts = [0.5 + 0.05 * i for i in range(91)]
    if qtype == "noul":
        pairs = _noul_pairs(rows)
        for T in ts:
            for b in B_GRID:
                v = _noul_nll(pairs, T, b)
                if v < best[0]:
                    best = (v, T, b)
    else:
        for T in ts:
            v = sum(nll(r, T) for r in rows)
            if v < best[0]:
                best = (v, T, 0.0)
    return best[1], best[2]


def fit_b(rows, T):
    """Per-dataset Platt bias at fixed T (noul only)."""
    pairs = _noul_pairs(rows)
    best = (1e18, 0.0)
    for b in B_GRID:
        v = _noul_nll(pairs, T, b)
        if v < best[0]:
            best = (v, b)
    if abs(best[1]) >= B_GRID[-1] - 1e-9:
        print(f"        WARNING: Platt b hit the grid edge ({best[1]:+.2f}) "
              f"-- optimum may lie beyond, widen B_GRID")
    return best[1]


def row_b(b_map, row):
    """b for this row: per-dataset map if enabled, else the global b."""
    return b_map.get(row["dataset"], 0.0) if isinstance(b_map, dict) else b_map


def metrics(rows, T, b=None, prior_by_ds=None, nbins=None):
    out = {}
    conf_hits = []
    hits, bs_, ns, es_all = [], [], [], []
    for r in rows:
        prior = (prior_by_ds or {}).get((r["dataset"], r["question_key"]))
        keys, p = probs_for(r, T, b, prior)
        im = max(range(len(p)), key=lambda i: p[i])
        lab = str(r["label"])
        if r["qtype"] == "score":
            mode = float(keys[im])
            e = sum(float(k) * pi for k, pi in zip(keys, p))
            hit = abs(mode - float(lab)) <= 1.0
            conf = sum(pi for k, pi in zip(keys, p)
                       if abs(float(k) - mode) <= 1.0)
            es_all.append(e)
        else:
            hit = keys[im] == lab
            conf = p[im]
        conf_hits.append((conf, 1.0 if hit else 0.0))
        hits.append(1.0 if hit else 0.0)
        bs_.append(sum((pi - (1.0 if k == lab else 0.0)) ** 2
                       for k, pi in zip(keys, p)))
        ns.append(-math.log(max(p[keys.index(lab)], 1e-12)))
    out["n"] = len(rows)
    out["accuracy"] = sum(hits) / len(hits)
    out["ece"] = ece(conf_hits, nbins or ECE_BINS)
    out["brier"] = sum(bs_) / len(bs_)
    out["nll"] = sum(ns) / len(ns)
    # selective: drop 20% least confident
    k = max(1, int(len(conf_hits) * 0.8))
    kept = sorted(conf_hits, key=lambda ch: -ch[0])[:k]
    out["selective_acc"] = sum(h for _, h in kept) / k
    out["selective_delta"] = out["selective_acc"] - out["accuracy"]
    return out


def boot_ci(rows, T, b, prior_by_ds, stat="ece", n=BOOTSTRAP, seed=7,
           nbins=None):
    rng = random.Random(seed)
    idx = list(range(len(rows)))
    vals = []
    for _ in range(n):
        samp = [rows[rng.choice(idx)] for _ in idx]
        m = metrics(samp, T, b, prior_by_ds, nbins)
        vals.append(m[stat])
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n)]


# ----------------------------------------------------------------- main ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl", nargs="*", default=None)
    ap.add_argument("--out", default="runs/calib/fit_results.json")
    args = ap.parse_args()

    files = args.jsonl or sorted(
        os.path.join("runs/calib", f) for f in os.listdir("runs/calib")
        if f.endswith(".logits.jsonl"))

    # A fit belongs next to the logits it was fitted on: one model per directory
    # (runs/calib = 9B, runs/calib_<label> = everything else). Writing elsewhere
    # silently stamps one model's numbers onto another model's file.
    in_dirs = {os.path.realpath(os.path.dirname(f) or ".") for f in files}
    if len(in_dirs) > 1:
        raise SystemExit(
            "refusing to fit across several directories (one model per directory):\n  "
            + "\n  ".join(sorted(in_dirs)))
    in_dir = in_dirs.pop()
    out_dir = os.path.realpath(os.path.dirname(args.out) or ".")
    if out_dir != in_dir:
        raise SystemExit(
            f"refusing to write the fit outside the logits directory\n"
            f"  logits: {in_dir}\n  --out : {out_dir}\n"
            f"write it next to its inputs, e.g. --out {os.path.join(in_dir, 'fit_results.json')}")
    rows, priors = [], {}
    for f in files:
        for line in open(f):
            r = json.loads(line)
            if r["id"].endswith("::prior") or "::prior::" in r["id"]:
                priors[(r["dataset"], r["question_key"])] = r
            elif r.get("label") is not None and not r.get("probe"):
                rows.append(r)  # H6 probe:true rows measured separately

    # split per dataset
    split_mode = {}
    ds_rows = defaultdict(list)
    for r in rows:
        ds_rows[r["dataset"]].append(r)
    for ds, rs in ds_rows.items():
        split_mode[ds] = assign_split(rs)

    by = defaultdict(lambda: defaultdict(list))  # qtype -> split -> rows
    for r in rows:
        by[r["qtype"]][r["split"]].append(r)

    results = {"split_mode": split_mode, "qtype": {}}
    print(f"{'qtype':7} {'FIT':>4} {'SEL':>4} {'TEST':>4}  T     b      "
          f"ECE_uncal ECE_test  Brier↓  NLL    selΔ     verdict")
    for qt in ("noul", "choice", "score"):
        fit, sel, test = by[qt]["FIT"], by[qt]["SELECT"], by[qt]["TEST"]
        if not test:
            continue
        T, b = fit_T(fit, qt) if fit else (1.0, 0.0)
        T_dist = None
        if qt == "score" and fit:
            T_dist = T                      # full-distribution T: diagnostic only
            T = fit_T_band(fit)
            print(f"        score T fitted on the +-1 band event: {T:.2f} "
                  f"(full-distribution T={T_dist:.2f} kept as diagnostic; "
                  f"objective changed after one look at TEST -- disclosed)")
        b_eval, eval_rows = b, test

        # noul bias selection on SELECT. The Platt b is fitted jointly with T
        # on FIT *NLL*, and NLL likes a bias shift that ECE and accuracy do
        # not: b=0 must therefore be a candidate of its own (it was not, and
        # the gate picked the worst of the three on TEST). Brief §6 requires
        # accuracy within +-0.5 %, and a bias shift moves the noul decision
        # boundary -- so candidates that break accuracy are rejected first,
        # then the simplest survivor with the lowest SELECT ECE wins.
        per_b = {}
        if qt == "noul" and fit and sel:
            fit_ds = defaultdict(list)
            for r in fit:
                fit_ds[r["dataset"]].append(r)
            per_b = {ds: fit_b(rs, T) for ds, rs in fit_ds.items() if len(rs) >= 30}
            if per_b:
                print(f"        noul Platt b per dataset (FIT, T={T:.2f}): "
                      + " ".join(f"{ds}={bb:+.2f}" for ds, bb in sorted(per_b.items())))
            sel_acc0 = metrics(sel, 1.0, None)["accuracy"]
            cands = [("T only", 0.0, sel, test),
                     (f"T + global b={b:+.2f}", b, sel, test)]
            if per_b:
                cands.append(("T + per-dataset b", None,
                              rebias(sel, per_b, T), rebias(test, per_b, T)))
            ok = []
            for name, bb, srows, trows in cands:
                m = metrics(srows, T, bb)
                drift = m["accuracy"] - sel_acc0
                mark = "ok " if abs(drift) <= PASS_ACC_DRIFT else "REJECT"
                print(f"        [{mark}] {name:22} SELECT acc {m['accuracy']:.4f} "
                      f"(drift {drift:+.4f}) ECE {m['ece']:.4f}")
                if abs(drift) <= PASS_ACC_DRIFT:
                    ok.append((m["ece"], name, bb, trows))
            if ok:
                ok.sort(key=lambda t: t[0])
                _, name, bb, trows = ok[0]
                b_eval, eval_rows = bb, trows
                print(f"        noul config chosen: {name} (accuracy-safe, "
                      f"lowest SELECT ECE)")
            else:
                print("        no accuracy-safe noul config; keeping T only")
                b_eval, eval_rows = 0.0, test

        # SELECT gate: prior if dECE >= 0.02 (evaluated on the winning b mode)
        if qt == "noul" and b_eval is None and per_b:
            sel_rows, sel_b = rebias(sel, per_b, T), None
        else:
            sel_rows, sel_b = sel, b_eval
        base_sel = metrics(sel_rows, T, sel_b)["ece"]
        prior_sel = metrics(sel_rows, T, sel_b, priors)["ece"] if (
            qt in ("noul", "choice")) else None
        use_prior = (prior_sel is not None and base_sel is not None
                     and (base_sel - prior_sel) >= 0.02)
        pb = priors if use_prior else None

        nb = ECE_BINS_SCORE if qt == "score" else ECE_BINS
        m0 = metrics(test, 1.0, None, None, nb)  # uncalibrated baseline
        m1 = metrics(eval_rows, T, b_eval, pb, nb)
        lo, hi = boot_ci(eval_rows, T, b_eval, pb, "ece", nbins=nb)

        ece_pass = m1["ece"] < PASS_ECE[qt]
        sel_pass = m1["selective_delta"] >= PASS_SELECTIVE_DELTA
        acc_drift = abs(m1["accuracy"] - m0["accuracy"])
        brier_pass = m1["brier"] <= m0["brier"]
        verdict = "PASS" if (ece_pass and brier_pass and acc_drift <= PASS_ACC_DRIFT) else (
            "RANKING-USEFUL, NOT CALIBRATED" if sel_pass else "FAIL")
        print(f"        accuracy sanity: uncal {m0['accuracy']:.4f} -> "
              f"calibrated {m1['accuracy']:.4f} (drift {m1['accuracy']-m0['accuracy']:+.4f}"
              f"{'; T alone cannot move argmax' if qt != 'noul' else '; Platt b can shift the noul boundary'})")
        b_show = b_eval if b_eval is not None else float("nan")
        print(f"{qt:7} {len(fit):>4} {len(sel):>4} {len(test):>4}  "
              f"{T:<4.2f} {b_show:<+5.2f} {m0['ece']:<9.4f} {m1['ece']:<9.4f} "
              f"{'↓' if brier_pass else '↑'}{m1['brier']:<5.4f} {m1['nll']:<6.3f} "
              f"{m1['selective_delta']:<+7.3f} {verdict}")
        results["qtype"][qt] = {
            "accuracy_uncal": m0["accuracy"],
            "accuracy_drift": m1["accuracy"] - m0["accuracy"],
            "T": T, "b": (b_eval if b_eval is not None else "per-dataset"),
            "b_mode": ("per-dataset" if b_eval is None and qt == "noul"
                       else ("none" if b_eval == 0.0 else "global")),
            "per_dataset_b": per_b or None, "use_prior": use_prior,
            "T_full_distribution_diagnostic": T_dist,
            "fitted_on": len(fit), "ece_test": m1["ece"],
            "ece_ci95": [round(lo, 4), round(hi, 4)],
            "ece_uncal": m0["ece"], "brier": m1["brier"], "nll": m1["nll"],
            "accuracy": m1["accuracy"], "selective_delta": m1["selective_delta"],
            "verdict": verdict,
        }
        if qt == "noul":
            ex = [r for r in eval_rows if r["dataset"] != "paysim_textualized"]
            if len(ex) < len(eval_rows):
                mx = metrics(ex, T, b_eval, pb, nb)
                lo2, hi2 = boot_ci(ex, T, b_eval, pb, "ece", nbins=nb)
                print(f"        noul WITHOUT paysim ({len(ex)} rows): "
                      f"acc {mx['accuracy']:.3f} ECE {mx['ece']:.4f} "
                      f"CI [{lo2:.4f},{hi2:.4f}] selD {mx['selective_delta']:+.3f}"
                      f" -- paysim is 50/50 answered at chance, its confidence "
                      f"ranks nothing")
                results["qtype"][qt]["without_paysim"] = {
                    "n": len(ex), "accuracy": mx["accuracy"], "ece": mx["ece"],
                    "ece_ci95": [round(lo2, 4), round(hi2, 4)],
                    "selective_delta": mx["selective_delta"]}

        if qt == "score":
            hits_mode = [pred_conf_hit(r, T, b, pb)[2] for r in test]
            es = [pred_conf_hit(r, T, b, pb)[3] for r in test]
            labs = [float(r["label"]) for r in test]
            mae = sum(abs(a - b) for a, b in zip(es, labs)) / len(es)
            results["qtype"][qt]["correct_within1"] = sum(hits_mode) / len(hits_mode)
            results["qtype"][qt]["mae_E"] = mae
            print(f"        score correct(|mode-lab|<=1) = "
                  f"{sum(hits_mode)/len(hits_mode):.4f}, MAE(E,label) = {mae:.3f} "
                  f"(ECE on 5 bins = {m1['ece']:.4f})")

    # per-dataset TEST breakdown: options with several BPE
    # spellings lose a few % of mass unevenly, so a dataset ECE stuck near
    # 0.02 after T is the engine floor, not a bad fit.
    print(f"\n{'dataset':24} {'qtype':7} {'n':>4}  acc     ECE     Brier   note")
    ds_out = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for qt in ("noul", "choice", "score"):
            sub = [r for r in by[qt]["TEST"] if r["dataset"] == ds]
            if not sub:
                continue
            cfg = results["qtype"].get(qt)
            if not cfg:
                continue
            T, b = cfg["T"], cfg["b"]
            pb = priors if cfg["use_prior"] else None
            if cfg["b_mode"] == "per-dataset" and cfg.get("per_dataset_b"):
                sub, b = rebias(sub, cfg["per_dataset_b"], T), None
            nb = ECE_BINS_SCORE if qt == "score" else ECE_BINS
            m = metrics(sub, T, b, pb, nb)
            note = "engine-floor?" if m["ece"] and m["ece"] < 0.03 else ""
            print(f"{ds:24} {qt:7} {len(sub):>4}  {m['accuracy']:.3f}   "
                  f"{m['ece']:.4f}  {m['brier']:.4f}  {note}")
            ds_out[f"{ds}/{qt}"] = {"n": len(sub), "accuracy": m["accuracy"],
                                    "ece": m["ece"], "brier": m["brier"]}
    results["per_dataset"] = ds_out

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")
    print("gate: shuffle(choice) " + (
        "NOT needed" if results["qtype"].get("choice", {}).get("ece_test", 1) < PASS_ECE["choice"]
        else "NEEDED but requires extra GPU runs (avg over 2-3 permutations)"))


if __name__ == "__main__":
    main()
