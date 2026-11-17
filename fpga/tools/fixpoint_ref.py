"""fixpoint_ref.py — bit-exact integer reference of the fraud deploy graph.

Deploy graph (fraud/src/edge_eval.py DeployPath; LM path NOT deployed):
    ids (13) + mask (13)
      pts = table[ids]*mask (13,8)  -> CloudEncoderND:
        point_mlp (8->64->64->128, GELU between) + max-pool
        pair geometry [diff8,dist,cos,|ni|,|nj| -> 26] + pair_mlp (26->64->64->128)
          + masked mean-pool
        fusion (256->256->128 + GELU + LayerNorm(128))
      class_head (128->64->1, GELU) -> sigmoid -> p

Pure-integer pipeline (runtime has NO float ops; the RTL mirrors this bit-for-bit):
  * Linear: int8 x int8, per-output-channel weight scale, int32 acc,
            one 24-bit constant Cq per output channel, round-half-away, clip int8, + int8 bias
  * mixed-scale inputs (table channels / pair-26ch / fusion halves): 16-bit
    rescale constant per channel  S[ch] = rha(2^16 * m_src[ch] / m_dst)
  * geometry + LayerNorm: Q16.16 fixed point, integer sqrt, 24-bit constants
  * GELU: 256-entry int8 LUT per instance ; sigmoid: 256-entry Q16.16 LUT
  * masked mean-pool: precomputed 169-entry constant table C_tab[pc]

Rounding convention everywhere (numpy ref AND RTL): round-half-away-from-zero.
All constants are derived at CALIBRATION time from the fp32 model and dumped to
hex files (save_consts); FixpointRef loads them, so the two sides cannot diverge.
"""
from __future__ import annotations

import os
import struct

import numpy as np
import torch


# ---------------------------------------------------------------------------
# integer rounding primitives  (MUST match the RTL)
# ---------------------------------------------------------------------------
def rha(x: np.ndarray, k: int) -> np.ndarray:
    """round-half-away-from-zero: q = sign * (abs(x) + 2^(k-1)) // 2^k."""
    x = np.asarray(x, dtype=np.int64)
    a = np.abs(x)
    q = (a + (1 << (k - 1))) >> k
    return np.where(x < 0, -q, q)


def rha_scalar(x_int: int, k: int) -> int:
    a = abs(x_int)
    q = (a + (1 << (k - 1))) >> k
    return -q if x_int < 0 else q


def rha_float(xf: float) -> int:
    """round-half-away-from-zero for a python float -> int (calibration time)."""
    if xf < 0:
        return -int((-xf) + 0.5)
    return int(xf + 0.5)


def clip8(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -128, 127)


def isqrt(x: np.ndarray) -> np.ndarray:
    """integer sqrt, elementwise, int64."""
    x = np.maximum(np.asarray(x, dtype=np.int64).reshape(-1), 0)
    if x.size == 0:
        return np.zeros(0, dtype=np.int64)
    hi = int(max(1, int(np.ceil(np.log2(int(x.max()) + 1)))))
    y = np.full_like(x, 1 << (hi // 2), dtype=np.int64)
    for _ in range(80):
        y1 = (y + x // y) >> 1
        if (y1 == y).all():
            y = y1
            break
        y = y1
    while (y * y > x).any():
        y = np.where(y * y > x, y - 1, y)
    return y


# ---------------------------------------------------------------------------
# calibration: float max-abs of every tensor (single fp32 run over the sample)
# ---------------------------------------------------------------------------
def calibrate(model, ids: np.ndarray) -> dict:
    """ids: (N,13) token matrix. Returns max-abs scales of every graph tensor."""
    from edge_eval import DeployPath
    dp = DeployPath(model)
    t = torch.from_numpy(np.asarray(ids, dtype=np.int64))
    msk = (t != 0) & (t != 1)
    pts = dp.table[t.long()] * msk.float().unsqueeze(-1)
    enc = model.cloud_encoder
    h1a = enc.point_mlp[0](pts)
    g1a = enc.point_mlp[1](h1a)
    h1b = enc.point_mlp[2](g1a)
    g1b = enc.point_mlp[3](h1b)
    pf = g1b.masked_fill(~msk.unsqueeze(-1), 0)
    point_pool = pf.max(dim=1).values
    diff = pts.unsqueeze(2) - pts.unsqueeze(1)
    dist = diff.norm(dim=-1)
    norm_i = pts.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cos_ang = (pts.unsqueeze(2) * pts.unsqueeze(1)).sum(-1) / \
        (norm_i * norm_i.transpose(1, 2)).clamp(min=1e-8)
    pair_in = torch.cat(
        [diff, dist.unsqueeze(-1), cos_ang.unsqueeze(-1),
         norm_i.unsqueeze(2).expand(-1, -1, 13, -1),
         norm_i.unsqueeze(1).expand(-1, 13, -1, -1)], -1)
    h2a = enc.pair_mlp[0](pair_in)
    g2a = enc.pair_mlp[1](h2a)
    h2b = enc.pair_mlp[2](g2a)
    g2b = enc.pair_mlp[3](h2b)
    pair_feats = g2b
    pmask = msk.unsqueeze(2) & msk.unsqueeze(1)
    pair_feats_m = pair_feats.masked_fill(~pmask.unsqueeze(-1), 0)
    pair_count = pmask.float().sum(dim=(1, 2)).clamp(min=1).unsqueeze(-1)
    pair_pool = pair_feats_m.sum(dim=(1, 2)) / pair_count
    fus_in = torch.cat([point_pool, pair_pool], -1)
    hf1 = enc.fusion[0](fus_in)
    gf = enc.fusion[1](hf1)
    hf2 = enc.fusion[2](gf)
    state = enc.fusion[3](hf2)
    hh = model.class_head[0](state)
    gh = model.class_head[1](hh)
    logit = model.class_head[2](gh).squeeze(-1)
    f = lambda x: float(x.detach().abs().max().clamp(min=1e-8))
    return {
        "m_tab": [float(dp.table[:, c].abs().max().clamp(min=1e-8)) for c in range(8)],
        "m_h1a": f(h1a), "m_g1a": f(g1a), "m_h1b": f(h1b), "m_g1b": f(g1b),
        "m_pf": f(pf), "m_diff": f(diff), "m_dist": f(dist),
        "m_norm": f(pts.norm(dim=-1)), "m_h2a": f(h2a), "m_g2a": f(g2a),
        "m_h2b": f(h2b), "m_g2b": f(g2b), "m_pf2": f(pair_feats_m),
        "m_pp": f(pair_pool), "m_hf1": f(hf1), "m_gf": f(gf),
        "m_hf2": f(hf2), "m_state": f(state),
        "m_hh": f(hh), "m_gh": f(gh), "m_logit": f(logit),
    }


# ---------------------------------------------------------------------------
# constant derivation (float -> int) + weight quantization + LUTs
# ---------------------------------------------------------------------------
def derive_consts(model, cal: dict, vocab: int = 221) -> dict:
    from edge_eval import DeployPath
    c = cal
    D = {}
    # table (per-channel) + Q16 rescale of the same int8 values
    tab = DeployPath(model).table.numpy()
    q_tab = np.zeros((vocab, 8), dtype=np.int8)
    for i in range(vocab):
        for ch in range(8):
            q_tab[i, ch] = int(clip8(np.array([rha_float(tab[i, ch] * 127.0 / c["m_tab"][ch])]))[0])
    D["q_tab"] = q_tab
    D["R_tab"] = np.array([rha_float(2 ** 16 * s / 127.0) for s in c["m_tab"]], dtype=np.int64)

    def lin8(layer, m_in, m_out):
        w = layer.weight.detach().numpy()
        b = layer.bias.detach().numpy() if layer.bias is not None else np.zeros(w.shape[0])
        m_w = [float(w[k].abs().max().clamp(min=1e-8)) for k in range(w.shape[0])]
        w8 = np.empty(w.shape, dtype=np.int8)
        for k in range(w.shape[0]):
            w8[k, :] = clip8(np.array([rha_float(v * 127.0 / m_w[k]) for v in w[k]]))
        b8 = clip8(np.array([rha_float(v * 127.0 / m_out) for v in b])).astype(np.int8)
        cq = np.array([rha_float(2 ** 24 * 127.0 * m_in * m_w[k] / m_out)
                       for k in range(w.shape[0])], dtype=np.int64)
        return w8, b8, cq

    pm, wm, fm, hm = model.cloud_encoder.point_mlp, model.cloud_encoder.pair_mlp, \
        model.cloud_encoder.fusion, model.class_head
    D["m_in1"] = max(c["m_tab"])
    D["S_tab"] = np.array([rha_float(2 ** 16 * s / D["m_in1"]) for s in c["m_tab"]], dtype=np.int64)
    D["pA"] = lin8(pm[0], D["m_in1"], c["m_h1a"])
    D["pB"] = lin8(pm[1], c["m_g1a"], c["m_h1b"])
    D["pC"] = lin8(pm[2], c["m_g1b"], c["m_pf"])
    D["g1a_lut"] = _gelu_lut(c["m_h1a"], c["m_g1a"])
    D["g1b_lut"] = _gelu_lut(c["m_h1b"], c["m_g1b"])

    # geometry Q16 -> int8 constants:  q8 = rha(x16 * Cq / 2^24), Cq = 2^8*127/m
    D["Cq_dist"] = np.array([rha_float(2 ** 8 * 127.0 / c["m_dist"])], dtype=np.int64)
    D["Cq_norm"] = np.array([rha_float(2 ** 8 * 127.0 / c["m_norm"])], dtype=np.int64)
    D["Cq_diff"] = np.array([rha_float(2 ** 8 * 127.0 / c["m_diff"])], dtype=np.int64)
    D["Cq_cos"] = np.array([2 ** 8 * 127], dtype=np.int64)          # cos scale = 1.0

    ch_scales = [c["m_diff"]] * 8 + [c["m_dist"], 1.0] + [c["m_norm"]] * 16
    D["m_inP"] = max(ch_scales)
    D["S_pain"] = np.array([rha_float(2 ** 16 * s / D["m_inP"]) for s in ch_scales], dtype=np.int64)
    D["qA"] = lin8(wm[0], D["m_inP"], c["m_h2a"])
    D["qB"] = lin8(wm[1], c["m_g2a"], c["m_h2b"])
    D["qC"] = lin8(wm[2], c["m_g2b"], c["m_pf2"])
    D["g2a_lut"] = _gelu_lut(c["m_h2a"], c["m_g2a"])
    D["g2b_lut"] = _gelu_lut(c["m_h2b"], c["m_g2b"])

    # masked mean-pool: q = clip8(rha(psum * C_tab[pc] / 2^24))
    D["C_tab"] = np.array([rha_float(c["m_pf2"] * 2 ** 24 / (c["m_pp"] * pc)) for pc in range(1, 170)],
                          dtype=np.int64)

    D["m_inF"] = max(c["m_pf"], c["m_pp"])
    D["S_fus"] = np.array([rha_float(2 ** 16 * s / D["m_inF"])
                           for s in [c["m_pf"]] * 128 + [c["m_pp"]] * 128], dtype=np.int64)
    D["fA"] = lin8(fm[0], D["m_inF"], c["m_hf1"])
    D["gf_lut"] = _gelu_lut(c["m_hf1"], c["m_gf"] if "m_gf" in c else c["m_hf1"])
    D["fB"] = lin8(fm[1], c["m_gf"] if "m_gf" in c else c["m_hf1"], c["m_hf2"])
    ln = fm[3]
    lnw, lnb = ln.weight.detach().numpy(), ln.bias.detach().numpy()
    D["ln_w16"] = np.array([rha_float(v * 2 ** 16) for v in lnw], dtype=np.int64)
    D["ln_b16"] = np.array([rha_float(v * 2 ** 16) for v in lnb], dtype=np.int64)
    D["R_hf2"] = np.array([rha_float(2 ** 16 * c["m_hf2"] / 127.0)], dtype=np.int64)
    D["ln_eps32"] = np.array([rha_float(1e-5 * 2 ** 32)], dtype=np.int64)
    D["Cq_state"] = np.array([rha_float(2 ** 8 * 127.0 / c["m_state"])], dtype=np.int64)

    D["hA"] = lin8(hm[0], c["m_state"], c["m_hh"])
    D["gh_lut"] = _gelu_lut(c["m_hh"], c["m_gh"])
    w = hm[1].weight.detach().numpy()
    m_w = float(w[0].abs().max().clamp(min=1e-8))
    D["hB_w8"] = clip8(np.array([rha_float(v * 127.0 / m_w) for v in w[0]])).astype(np.int8)
    D["hB_b16"] = np.array([rha_float(hm[1].bias.detach().numpy()[0] * 2 ** 16)], dtype=np.int64)
    D["Cq_logit"] = np.array([rha_float(2 ** 24 * c["m_gh"] * m_w)], dtype=np.int64)
    D["Cq_sig"] = np.array([rha_float(2 ** 8 * 127.0 / c["m_logit"])], dtype=np.int64)
    D["sig_lut"] = _sigmoid_lut(c["m_logit"])
    return D


def _gelu_lut(m_in: float, m_out: float) -> np.ndarray:
    import torch.nn.functional as F
    q = np.arange(-128, 128)
    y = F.gelu(torch.tensor(q, dtype=torch.float64) * (m_in / 127.0)).numpy()
    return clip8(np.array([rha_float(v * 127.0 / m_out) for v in y])).astype(np.int8)


def _sigmoid_lut(m_logit: float) -> np.ndarray:
    q = np.arange(-128, 128)
    y = 1.0 / (1.0 + np.exp(-q * (m_logit / 127.0)))
    return np.clip(np.round(y * 65535.0), 0, 65535).astype(np.uint16)


# ---------------------------------------------------------------------------
# hex (de)serialization — shared by ref (read) and RTL $readmemh (write)
# ---------------------------------------------------------------------------
def save_consts(D: dict, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    def hex8(a: np.ndarray, name: str):
        a = a.reshape(-1)
        with open(os.path.join(outdir, name), "w") as f:
            for v in a:
                f.write(f"{int(v) & 0xFF:02x}\n")

    def hex16(a: np.ndarray, name: str):
        a = a.reshape(-1)
        with open(os.path.join(outdir, name), "w") as f:
            for v in a:
                f.write(f"{int(v) & 0xFFFF:04x}\n")

    def hex32(a: np.ndarray, name: str):
        a = a.reshape(-1)
        with open(os.path.join(outdir, name), "w") as f:
            for v in a:
                f.write(f"{int(v) & 0xFFFFFFFF:08x}\n")

    hex8(D["q_tab"], "q_tab.hex")            # 221*8
    hex8(D["pA"][0], "pA_w.hex"); hex8(D["pA"][1], "pA_b.hex")
    hex8(D["pB"][0], "pB_w.hex"); hex8(D["pB"][1], "pB_b.hex")
    hex8(D["pC"][0], "pC_w.hex"); hex8(D["pC"][1], "pC_b.hex")
    hex8(D["qA"][0], "qA_w.hex"); hex8(D["qA"][1], "qA_b.hex")
    hex8(D["qB"][0], "qB_w.hex"); hex8(D["qB"][1], "qB_b.hex")
    hex8(D["qC"][0], "qC_w.hex"); hex8(D["qC"][1], "qC_b.hex")
    hex8(D["fA"][0], "fA_w.hex"); hex8(D["fA"][1], "fA_b.hex")
    hex8(D["fB"][0], "fB_w.hex"); hex8(D["fB"][1], "fB_b.hex")
    hex8(D["hA"][0], "hA_w.hex"); hex8(D["hA"][1], "hA_b.hex")
    hex8(D["hB_w8"], "hB_w.hex")
    hex8(D["g1a_lut"], "g1a.hex"); hex8(D["g1b_lut"], "g1b.hex")
    hex8(D["g2a_lut"], "g2a.hex"); hex8(D["g2b_lut"], "g2b.hex")
    hex8(D["gf_lut"], "gf.hex"); hex8(D["gh_lut"], "gh.hex")
    hex16(D["sig_lut"], "sig.hex")
    hex32(D["S_tab"], "S_tab.hex"); hex32(D["S_pain"], "S_pain.hex")
    hex32(D["S_fus"], "S_fus.hex")
    hex32(D["R_tab"], "R_tab.hex"); hex32(D["R_hf2"], "R_hf2.hex")
    hex32(D["Cq_dist"], "Cq_dist.hex"); hex32(D["Cq_norm"], "Cq_norm.hex")
    hex32(D["Cq_diff"], "Cq_diff.hex"); hex32(D["Cq_cos"], "Cq_cos.hex")
    hex32(D["Cq_state"], "Cq_state.hex"); hex32(D["Cq_logit"], "Cq_logit.hex")
    hex32(D["Cq_sig"], "Cq_sig.hex"); hex32(D["ln_eps32"], "ln_eps32.hex")
    hex32(D["C_tab"], "C_tab.hex")           # 169 entries
    hex32(D["pA"][2], "pA_cq.hex"); hex32(D["pB"][2], "pB_cq.hex"); hex32(D["pC"][2], "pC_cq.hex")
    hex32(D["qA"][2], "qA_cq.hex"); hex32(D["qB"][2], "qB_cq.hex"); hex32(D["qC"][2], "qC_cq.hex")
    hex32(D["fA"][2], "fA_cq.hex"); hex32(D["fB"][2], "fB_cq.hex"); hex32(D["hA"][2], "hA_cq.hex")
    hex32(D["ln_w16"], "ln_w16.hex"); hex32(D["ln_b16"], "ln_b16.hex")
    hex32(D["hB_b16"], "hB_b16.hex")


def load_consts(dirpath: str) -> dict:
    def rd8(name):
        with open(os.path.join(dirpath, name)) as f:
            vals = [int(x, 16) for x in f if x.strip()]
        a = np.array(vals, dtype=np.int64)
        a = np.where(a > 127, a - 256, a)          # two's-complement 8-bit
        return a

    def rd16(name):
        with open(os.path.join(dirpath, name)) as f:
            return np.array([int(x, 16) for x in f if x.strip()], dtype=np.int64)

    def rd32(name):
        with open(os.path.join(dirpath, name)) as f:
            vals = [int(x, 16) for x in f if x.strip()]
        a = np.array(vals, dtype=np.int64)
        a = np.where(a >= 1 << 31, a - (1 << 32), a)   # two's-complement 32-bit
        return a

    D = {}
    D["q_tab"] = rd8("q_tab.hex").reshape(221, 8)
    for p in ["pA", "pB", "pC", "qA", "qB", "qC", "fA", "fB", "hA"]:
        D[p + "_w8"] = rd8(f"{p}_w.hex")
        D[p + "_b8"] = rd8(f"{p}_b.hex")
        D[p + "_cq"] = rd32(f"{p}_cq.hex")
    D["hB_w8"] = rd8("hB_w.hex")
    D["hB_b16"] = rd32("hB_b16.hex")
    for n in ["g1a", "g1b", "g2a", "g2b", "gf", "gh"]:
        D[n + "_lut"] = rd8(f"{n}.hex")
    D["sig_lut"] = rd16("sig.hex")
    D["S_tab"] = rd32("S_tab.hex"); D["S_pain"] = rd32("S_pain.hex"); D["S_fus"] = rd32("S_fus.hex")
    D["R_tab"] = rd32("R_tab.hex"); D["R_hf2"] = rd32("R_hf2.hex")
    for n in ["Cq_dist", "Cq_norm", "Cq_diff", "Cq_cos", "Cq_state", "Cq_logit", "Cq_sig", "ln_eps32"]:
        D[n] = rd32(f"{n}.hex")
    D["C_tab"] = rd32("C_tab.hex")
    D["ln_w16"] = rd32("ln_w16.hex"); D["ln_b16"] = rd32("ln_b16.hex")
    return D


# ---------------------------------------------------------------------------
# the pure-integer forward  (one transaction)
# ---------------------------------------------------------------------------
class FixpointRef:
    K = 13
    V = 221

    def __init__(self, consts: dict | None = None, dirpath: str | None = None):
        self.D = consts if consts is not None else load_consts(dirpath)

    # -- primitives (mirrored 1:1 in RTL) --------------------------------
    @staticmethod
    def _rescale(x8: np.ndarray, S: np.ndarray) -> np.ndarray:
        return clip8(rha(x8.astype(np.int64) * S[None, :], 16))

    def _lin(self, x8: np.ndarray, S, w8: np.ndarray, b8: np.ndarray, cq: np.ndarray) -> np.ndarray:
        """x8 (M,in) int8 -> rescale -> int8 -> acc int32 -> Cq dequant -> int8 + bias."""
        x = self._rescale(x8, S) if S is not None else x8.astype(np.int64)
        acc = x.astype(np.int64) @ w8.astype(np.int64).T
        y = rha(acc * cq[None, :], 24) + b8[None, :]
        return clip8(y).astype(np.int8)

    def _gelu(self, h8: np.ndarray, lut: np.ndarray) -> np.ndarray:
        return lut[h8.astype(np.int64) + 128]

    # -- forward ----------------------------------------------------------
    def forward(self, ids: np.ndarray, mask: np.ndarray) -> dict:
        D = self.D
        K = self.K
        ids = np.asarray(ids, dtype=np.int64).reshape(K)
        m8 = np.asarray(mask, dtype=np.int8).reshape(K)
        out: dict = {}

        # stage 0 — table gather (int8) + Q16
        pts8 = D["q_tab"][ids] * m8[:, None]
        out["pts8"] = pts8
        pts16 = pts8.astype(np.int64) * D["R_tab"][None, :]          # Q16.16

        # stage 1 — point_mlp
        h = self._lin(pts8, D["S_tab"], D["pA_w8"], D["pA_b8"], D["pA_cq"])
        h = self._gelu(h, D["g1a_lut"])
        h = self._lin(h, None, D["pB_w8"], D["pB_b8"], D["pB_cq"])
        h = self._gelu(h, D["g1b_lut"])
        pf8 = self._lin(h, None, D["pC_w8"], D["pC_b8"], D["pC_cq"]) * m8[:, None]
        out["pf8"] = pf8
        pp8 = pf8.max(axis=0)
        out["point_pool8"] = pp8

        # stage 2 — geometry (Q16.16 -> int8)
        diff16 = pts16[:, None, :] - pts16[None, :, :]               # Q16 (13,13,8)
        ss = (diff16 * diff16).sum(axis=2).astype(np.int64)         # Q32
        dist16 = isqrt(ss)                                          # Q16
        dist8 = clip8(rha(dist16 * D["Cq_dist"], 24)).astype(np.int8)
        n16 = isqrt((pts16 * pts16).sum(axis=1))                    # Q16 (13,)
        n16 = np.maximum(n16, 1)                                    # clamp(min=1e-8)
        norm8 = clip8(rha(n16 * D["Cq_norm"], 24)).astype(np.int8)
        num = (pts16[:, None, :] * pts16[None, :, :]).sum(axis=2)   # Q32 (13,13)
        den = np.outer(n16, n16)                                    # Q32
        cos16 = np.clip((num << 16) // np.maximum(den, 1), -(1 << 16), 1 << 16)
        cos8 = clip8(rha(cos16 * D["Cq_cos"], 24)).astype(np.int8)
        pmask = (m8[:, None] & m8[None, :]).astype(bool)
        pc = max(int(pmask.sum()), 1)
        out["pmask"] = pmask
        out["pair_count"] = pc
        diff8 = clip8(rha(diff16 * D["Cq_diff"], 24)).astype(np.int8)
        pain8 = np.zeros((K, K, 26), dtype=np.int8)
        pain8[:, :, 0:8] = diff8
        pain8[:, :, 8] = dist8
        pain8[:, :, 9] = cos8
        pain8[:, :, 10:18] = norm8[:, None, :]
        pain8[:, :, 18:26] = norm8[None, :, :]
        out["pain8"] = pain8

        # stage 3 — pair_mlp + masked mean-pool
        h = self._lin(pain8.reshape(-1, 26), D["S_pain"], D["qA_w8"], D["qA_b8"], D["qA_cq"])
        h = self._gelu(h, D["g2a_lut"])
        h = self._lin(h, None, D["qB_w8"], D["qB_b8"], D["qB_cq"])
        h = self._gelu(h, D["g2b_lut"])
        pf8p = self._lin(h, None, D["qC_w8"], D["qC_b8"], D["qC_cq"]).reshape(K, K, 128)
        pf8p = pf8p * pmask[:, :, None].astype(np.int8)
        out["pf8p"] = pf8p
        psum = pf8p.sum(axis=(0, 1)).astype(np.int64)
        ppair8 = clip8(rha(psum * D["C_tab"][pc - 1], 24)).astype(np.int8)
        out["pair_pool8"] = ppair8

        # stage 4 — fusion + LayerNorm
        fin8 = np.concatenate([pp8, ppair8])
        fin8 = self._rescale(fin8, D["S_fus"])
        h = self._lin(fin8, None, D["fA_w8"], D["fA_b8"], D["fA_cq"])
        h = self._gelu(h, D["gf_lut"])
        h = self._lin(h, None, D["fB_w8"], D["fB_b8"], D["fB_cq"])   # (128,) int8
        h16 = h.astype(np.int64) * D["R_hf2"]                        # Q16.16
        s16 = h16.sum()
        mean16 = rha_scalar(s16, 7)                                  # /128 = /2^7
        x16 = h16 - mean16
        var32 = max(int((x16 * x16).sum() - mean16 * mean16), 0) + int(D["ln_eps32"])
        rstd16 = (1 << 16) // int(isqrt(np.array([var32]))[0])
        y16 = rha(x16 * rstd16, 16)
        ln16 = rha(y16 * D["ln_w16"], 16) + D["ln_b16"]
        out["ln16"] = ln16
        st8 = clip8(rha(ln16 * D["Cq_state"], 24)).astype(np.int8)
        out["state8"] = st8

        # stage 5 — head + sigmoid
        h = self._lin(st8, None, D["hA_w8"], D["hA_b8"], D["hA_cq"])
        h = self._gelu(h, D["gh_lut"])
        acc = int((h.astype(np.int64) * D["hB_w8"]).sum())
        logit16 = rha_scalar(acc * int(D["Cq_logit"]), 24) + int(D["hB_b16"])
        out["logit16"] = logit16
        q = int(clip8(np.array([rha_scalar(logit16 * int(D["Cq_sig"]), 24)]))[0])
        p16 = int(D["sig_lut"][q + 128])
        out["p16"] = p16
        return out
