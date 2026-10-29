"""gen_golden.py — build the Phase A golden set for the fpga/ sim.

Pipeline (all deterministic, offline):
  1. load deploy checkpoint + bundled quantizer (fraud/results/deploy/)
  2. FeatureQuantizer.transform(User0 CSV) -> ids (19963, 13)
  3. calibrate int8/Q16.16 constants on a 5000-row sample -> dump hex consts
  4. pick 50 golden cases (10 fraud / 25 normal / 15 edge, fixed seed)
  5. FixpointRef forward per case -> per-layer golden hex + cases.csv
  6. whole-User0 drift: AUC(p_fp32) vs AUC(p_int8ref), max abs diff

Golden layout: fpga/tb/golden/consts/*.hex (weights/LUTs/constants, shared by
ref and RTL $readmemh) + fpga/tb/golden/case_XXX/{stim,pts,pf,pp,pain,pmask,
pf8p,ppair,ln16,state,logit,p16}.hex + fpga/tb/golden/cases.csv
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "fraud", "src"))
sys.path.insert(0, HERE)

import config  # noqa: E402  (bootstraps repo root)
from edge_eval import DeployPath  # noqa: E402
from qcloud_fraud import QCloudFraudClassifier  # noqa: E402
from quantize import FeatureQuantizer  # noqa: E402
import datasets  # noqa: E402
import fixpoint_ref as fx  # noqa: E402

CHECKPOINT = os.path.join(config.RESULTS_DIR, "quann_full_hybrid.pt")
DEPLOY_DIR = os.path.join(config.RESULTS_DIR, "deploy")
GOLDEN = os.path.join(HERE, "..", "tb", "golden")
K = 13
SEED = 20261019
N_CAL = 5000
N_CASES = 50


def log(msg: str):
    print(f"[gen_golden] {msg}", flush=True)


def hexw(path: str, a: np.ndarray, width: int):
    """Write values to hex file for $readmemh.
    
    FIXED: Properly mask negative values for unsigned hex format.
    Negative int8 values use 2's complement (e.g., -1 = 0xFF for width=1).
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fmt = {1: "%02x", 2: "%04x", 4: "%08x"}[width]
    # Mask for unsigned representation
    mask = {1: 0xFF, 2: 0xFFFF, 4: 0xFFFFFFFF}[width]
    with open(path, "w") as f:
        for v in np.asarray(a).reshape(-1):
            # FIXED: Mask negative values for proper hex representation
            v_masked = int(v) & mask
            f.write((fmt % v_masked) + "\n")


def main():
    t0 = time.time()
    os.makedirs(GOLDEN, exist_ok=True)
    os.makedirs(os.path.join(GOLDEN, "consts"), exist_ok=True)

    log("1/5 load checkpoint + quantizer + User0 CSV")
    model = QCloudFraudClassifier.load_quann(CHECKPOINT, torch.device("cpu"))
    with open(os.path.join(DEPLOY_DIR, "quantizer.json")) as f:
        fq = FeatureQuantizer.from_dict(json.load(f))
    frame = datasets.load_small()
    labels = datasets.get_label(frame)
    ids = fq.transform(frame)
    ids = np.asarray(ids, dtype=np.int64)
    log(f"  ids {ids.shape} | vocab={fq.vocab_size} | fraud rows={int(labels.sum()):,}")
    assert ids.shape[1] == K

    log(f"2/5 calibrate constants on {N_CAL} rows")
    rng = np.random.default_rng(SEED)
    cal_idx = rng.choice(len(ids), size=min(N_CAL, len(ids)), replace=False)
    cal = fx.calibrate(model, ids[cal_idx])
    D = fx.derive_consts(model, cal, vocab=fq.vocab_size)
    fx.save_consts(D, os.path.join(GOLDEN, "consts"))
    with open(os.path.join(GOLDEN, "cal.json"), "w") as f:
        json.dump({k: (v if not isinstance(v, list) else v) for k, v in cal.items()}, f, indent=1)
    log(f"  scales: tab={np.round(cal['m_tab'], 4).tolist()} pf={cal['m_pf']:.4f} "
        f"pf2={cal['m_pf2']:.4f} pp={cal['m_pp']:.4f} state={cal['m_state']:.4f} "
        f"logit={cal['m_logit']:.3f} dist={cal['m_dist']:.3f} norm={cal['m_norm']:.3f}")

    log("3/5 pick 50 golden cases (fraud/normal/edge)")
    ref = fx.FixpointRef(consts=D)
    fraud_idx = np.flatnonzero(labels == 1)
    normal_idx = np.flatnonzero(labels == 0)
    picks = []
    picks += list(rng.choice(fraud_idx, size=10, replace=False))
    picks += list(rng.choice(normal_idx, size=25, replace=False))
    edge = []
    for i in rng.choice(len(ids), size=4, replace=False):        # all-same token
        r = np.full(K, int(ids[i, 0]), dtype=np.int64)
        edge.append(("allsame", i, r))
    for i in rng.choice(len(ids), size=4, replace=False):        # pad/eos holes
        r = ids[i].copy()
        for pos in rng.choice(K, size=rng.integers(1, 4), replace=False):
            r[pos] = rng.integers(0, 2)
        edge.append(("hole", i, r))
    t1 = time.time()
    p_ref_all = fx_batch(ref, ids, labels)                      # for extremes + step 5
    log(f"  ref over all {len(ids):,} rows: {time.time() - t1:.1f}s")
    order = np.argsort(p_ref_all)
    edge += [("extreme", int(order[i]), ids[order[i]]) for i in np.linspace(0, len(order) - 1, 4).astype(int)]
    freq = np.bincount(ids.reshape(-1), minlength=fq.vocab_size)
    rare = np.argsort(freq)[8:12]
    for tok in rare:                                            # rarest-token rows
        rows = np.flatnonzero((ids == tok).any(axis=1))
        if len(rows):
            edge.append(("rare", int(rows[0]), ids[rows[0]]))
    cases = []
    for i, tag in [(int(i), "real") for i in picks[:35]]:
        cases.append((tag, i, ids[i]))
    for tag, i, r in edge[:15]:
        cases.append((tag, i, r))
    cases = cases[:N_CASES]

    log("4/5 write per-case golden (bit-exact ref outputs)")
    rows = []
    for c, (tag, i, r) in enumerate(cases):
        msk = ((r != 0) & (r != 1)).astype(np.int64)
        out = ref.forward(r, msk)
        d = os.path.join(GOLDEN, f"case_{c:03d}")
        hexw(os.path.join(d, "stim.hex"), r, 2)
        hexw(os.path.join(d, "mask.hex"), msk, 1)
        hexw(os.path.join(d, "pts.hex"), out["pts8"], 1)
        hexw(os.path.join(d, "pf.hex"), out["pf8"], 1)
        hexw(os.path.join(d, "pp.hex"), out["point_pool8"], 1)
        hexw(os.path.join(d, "pain.hex"), out["pain8"], 1)
        hexw(os.path.join(d, "pmask.hex"), out["pmask"].astype(np.int64).reshape(-1), 1)
        hexw(os.path.join(d, "pf8p.hex"), out["pf8p"], 1)
        hexw(os.path.join(d, "ppair.hex"), out["pair_pool8"], 1)
        hexw(os.path.join(d, "ln16.hex"), out["ln16"], 4)
        hexw(os.path.join(d, "state.hex"), out["state8"], 1)
        hexw(os.path.join(d, "logit.hex"), np.array([out["logit16"]]), 4)
        hexw(os.path.join(d, "p16.hex"), np.array([out["p16"]]), 2)
        with open(os.path.join(d, "meta.json"), "w") as f:
            json.dump({"tag": tag, "src_row": int(i), "pair_count": int(out["pair_count"])}, f)
        p16 = int(out["p16"]) / 65535.0
        rows.append({"case": c, "tag": tag, "src_row": int(i), "k": K,
                     "ids": ",".join(str(int(v)) for v in r),
                     "mask": ",".join(str(int(v)) for v in msk),
                     "label": int(labels[i]), "p_ref": p16})
    with open(os.path.join(GOLDEN, "cases.csv"), "w") as f:
        f.write("case,tag,src_row,k,ids,mask,label,p_ref\n")
        for r_ in rows:
            f.write(f"{r_['case']},{r_['tag']},{r_['src_row']},{r_['k']},"
                    f"{r_['ids']},{r_['mask']},{r_['label']},{r_['p_ref']:.9f}\n")

    log("5/5 whole-User0 drift: fp32 vs int8 ref")
    dp = DeployPath(model)
    t = torch.from_numpy(ids)
    msk = (t != 0) & (t != 1)
    with torch.no_grad():
        p_fp32 = dp(t, msk).cpu().numpy()
    n = len(ids)
    p_ref = p_ref_all
    maxd = float(np.abs(p_fp32 - p_ref).max())
    a_fp32, a_ref = _rank_auc(p_fp32, labels), _rank_auc(p_ref, labels)
    summary = {
        "n_rows": n, "seed": SEED, "n_cal": int(len(cal_idx)),
        "max_abs_p_diff": maxd,
        "auc_fp32": a_fp32, "auc_int8ref": a_ref, "auc_delta": a_ref - a_fp32,
        "fraud_rows": int(labels.sum()),
        "cal_scales": cal,
        "wall_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(GOLDEN, "drift.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    log(f"  max|dp|={maxd:.4f} AUC fp32={a_fp32:.4f} int8ref={a_ref:.4f} "
        f"(Δ={a_ref - a_fp32:+.4f}) | wall {summary['wall_s']}s")
    log("done")


def _rank_auc(p: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), dtype=np.float64)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ties
    s = 0.0
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and p[order[j + 1]] == p[order[i]]:
            j += 1
        if j > i:
            avg = ranks[order[i:j + 1]].mean()
            ranks[order[i:j + 1]] = avg
        i = j + 1
    pos = int(y.sum())
    neg = len(y) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def fx_batch(ref: fx.FixpointRef, ids: np.ndarray, labels: np.ndarray) -> np.ndarray:
    out = np.zeros(len(ids))
    for i in range(len(ids)):
        out[i] = ref.forward(ids[i], (ids[i] != 0) & (ids[i] != 1))["p16"] / 65535.0
    return out


if __name__ == "__main__":
    main()
