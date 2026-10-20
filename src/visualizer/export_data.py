"""
Quann-3D 3D Görselleştirme — veri dışa aktarıcı
================================================
Gerçek sistemden gerçek veri çekip `index.html` (tek dosya, Three.js) üretir.

Bölümler:
  1) CLOUD  — eğitilmiş wiki 200K CloudLM (21M, D=32, BPE 20K): seçili cümlelerin
              gerçek bulut noktaları (32D), ortak PCA ile 3D'ye projeksiyon,
              mesafe/açı matrisleri, next-word tahmini + next-point,
              ve "son kelime gizli" (fragment) varyantı.
  2) QUANN  — (a) Temiz görev: hedef bulut konfigürasyonuna 0th-order aramanın
                   yakınsaması (motor kuralları birebir: qcloud_quann.py).
               (b) Gerçek LM: küçük CloudLM (backprop'la eğitip dondurulur) +
                   nokta-bazı 0th-order arama (QCloudWeight tasarımı);
                   hata = gerçek cross-entropy forward'ı.
  3) BULGULAR — RESEARCH.md/FINDINGS.md'deki ana sayılar.

Notlar:
  - YÜKLEME CPU'da (MPS canlı eğitimle paylaşılmıyor).
  - Çalıştır:  python3 src/visualizer/export_data.py
  - Çıktı:     src/visualizer/index.html  (tarayıcıda doğrudan açılır)
"""

import json
import math
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F

torch.set_num_threads(4)

HERE = os.path.dirname(os.path.abspath(__file__))


def r3(x):
    return [round(float(v), 4) for v in np.asarray(x).ravel()]


def r3m(M):
    return [[round(float(v), 4) for v in row] for row in np.asarray(M)]


def geom3(A):
    """3D nokta setinden mesafe + cos- açı matrisleri."""
    A = np.asarray(A, dtype=float)
    n = len(A)
    D = np.zeros((n, n))
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = float(np.linalg.norm(A[i] - A[j]))
            ni, nj = max(1e-8, np.linalg.norm(A[i])), max(1e-8, np.linalg.norm(A[j]))
            cos = float(A[i] @ A[j]) / (ni * nj)
            D[i, j] = D[j, i] = dist
            C[i, j] = C[j, i] = cos
    return r3m(D), r3m(C)


# ═══════════════════════════════════════════════════════════════
# 1) CLOUD — wiki 200K modeli
# ═══════════════════════════════════════════════════════════════

CLOUD_SENTENCES = [
    "Fıkırdakça hareket eden kedi, avını yakalamak için gizlice yaklaşmaya çalışıyordu.",
    "Küçük köpek, sahibinin sesini duyduğunda heyecanla hoplamada bulunuyordu.",
    "Bahçedeki sahipsiz köpekler için her sabah kapının önüne bir kap su bırakıyorum.",
    "Kocaman bir pasta görünce çocukların yüzündeki mutluluk aman dedirtiyor.",
    "Hastalandığı için günlerdir yatağından kalkamayan zavallı çocuk pencereden dışarıdaki oynayan arkadaşlarını izliyordu.",
    "Köylü kadınlar odun ateşinde pişirdikleri bazlamaları ve ekmekleri misafirlere sundular.",
    "Evin yan penceresinden süzülen güneş ışığı odanın içini sıcacık ve aydınlık yapıyordu.",
    "Karneler dağıtılırken, bazı öğrencilerin yüzündeki sevinç ifadesi, tüm okul koridorunu aydınlatıyordu.",
    "Eski eşyalarını kullanabileceklerin ve çöpe atılacakların diye ikiye ayıran kadın evi ferahlattı.",
    "Yangın güvenliği uzmanları, özellikle çocuklu evlerde fitilsiz aydınlatma çözümlerini şiddetle tavsiye etmektedirler.",
]


def cloud_section():
    print("== CLOUD: wiki 200K modeli yükleniyor ...", flush=True)
    from cloudlm import CloudLM, BPETokenizer

    ck = torch.load(os.path.join(ROOT, "checkpoints/wiki_checkpoints_200k/wiki_cloud_best.pt"),
                    map_location="cpu")
    model = CloudLM(ck["vocab_size"], **ck["config"])
    model.load_state_dict(ck["model_state"])
    model.eval()
    tok = BPETokenizer.load(os.path.join(ROOT, "checkpoints/wiki_checkpoints_200k/wiki_bpe.json"))
    print(f"   yüklendi: {sum(p.numel() for p in model.parameters())/1e6:.1f}M param, "
          f"vocab {tok.num_tokens}, best PPL {ck['ppl']:.2f}", flush=True)

    # cümle → (kelime etiketleri, token id'leri, kelime→token grupları).
    # Tam cümle encode edilir (model bu biçimde eğitildi); kelime grupları
    # token offset'lerinden çıkarılır, etiketler BPE decoder ile temizlenir.
    # wordSpans: kelimelerin token dizisindeki [başlangıç, bitiş) aralıkları —
    # 1 kelime = 1+ BPE token olabildiğinden nokta/kelime sayıları eşit değildir.
    def encode_words(text):
        enc = tok._tokenizer.encode(text)
        ids, offsets = enc.ids, enc.offsets
        word_spans = []
        pos = 0
        for w in text.split():
            pos = text.find(w, pos)
            word_spans.append((pos, pos + len(w)))
        groups = [[] for _ in word_spans]
        for i, (a, b) in enumerate(offsets):
            s2 = a
            while s2 < b and text[s2] in " \t\n":  # token span öndeki boşluğu içerir
                s2 += 1
            for wi, (sa, sb) in enumerate(word_spans):
                if sa <= s2 < sb:
                    groups[wi].append(ids[i])
                    break
        words = [tok.decode(g).strip() if g else "" for g in groups]
        gsizes = [len(g) for g in groups]
        bounds = [0]
        for g in gsizes:
            bounds.append(bounds[-1] + g)
        spans = [[bounds[i], bounds[i + 1]] for i in range(len(bounds) - 1)]
        return words, ids, gsizes, spans

    sents = []
    all_pts = []
    for text in CLOUD_SENTENCES:
        words, ids, gsizes, spans = encode_words(text)
        full = ids
        frag = ids[: len(ids) - gsizes[-1]]  # son KELIMEYİ (tüm tokenları) gizle
        with torch.no_grad():
            m_full = torch.ones(1, len(full), dtype=torch.bool)
            lg_f, np_f, pts_f = model(torch.tensor([full]), m_full)
            m_fr = torch.ones(1, len(frag), dtype=torch.bool)
            lg_r, np_r, pts_r = model(torch.tensor([frag]), m_fr)
        probs_f = F.softmax(lg_f[0], dim=-1)
        probs_r = F.softmax(lg_r[0], dim=-1)

        def top5(probs):
            idx = torch.topk(probs, 5).indices.tolist()
            return [{"id": int(i), "w": tok.decode([i]).strip() or "·",
                     "p": round(float(probs[i]), 4)} for i in idx]

        sents.append({
            "text": text,
            "words": words,
            "wordSpans": spans,
            "pts": r3m(pts_f[0]),
            "nextPt": r3(np_f[0]),
            "top5": top5(probs_f),
            "fragWords": words[:-1],
            "fragSpans": spans[:-1],
            "fragPts": r3m(pts_r[0]),
            "fragNextPt": r3(np_r[0]),
            "fragTop5": top5(probs_r),
        })
        all_pts.append(pts_f[0].numpy())
        print(f"   cümle {len(sents):2d}: {len(words)} kelime / {len(ids)} token — "
              f"tahmin: {sents[-1]['fragTop5'][0]['w']!r}", flush=True)

    # ortak PCA 32 → 3 (tüm cümle noktaları birlikte)
    P = np.vstack(all_pts)
    P = P - P.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    comp = Vt[:3]
    proj = (P @ comp.T + P.mean(0, keepdims=True) @ comp.T)

    def to3(Pd):
        return (Pd - P.mean(0, keepdims=True)) @ comp.T

    for s, Pd in zip(sents, all_pts):
        s["pts3"] = r3m(to3(Pd))
        s["nextPt3"] = r3(to3(np.array([s["nextPt"]]))[0])
        s["fragPts3"] = r3m(to3(np.array(s["fragPts"])))
        s["fragNextPt3"] = r3(to3(np.array([s["fragNextPt"]]))[0])
        s["distM"], s["cosM"] = geom3(np.array(s["pts3"]))
        s["fragDistM"], s["fragCosM"] = geom3(np.array(s["fragPts3"]))

    info = {
        "model": "CloudLM wiki 200K (eğitilmiş, gerçek)",
        "cfg": ck["config"],
        "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "best_ppl": round(float(ck["ppl"]), 2),
        "vocab": tok.num_tokens,
        "pca": f"32D → 3D ortak PCA ({len(CLOUD_SENTENCES)} cümle, {len(P)} nokta)",
        "sents": sents,
    }
    print(f"   CLOUD tamam: {len(sents)} cümle, {len(P)} nokta", flush=True)
    return info


# ═══════════════════════════════════════════════════════════════
# 2) QUANN — 0th-order arama (qcloud_quann.py motor kuralları)
# ═══════════════════════════════════════════════════════════════

DEMO_TEXTS = [
    "kedi pencereden dışarı baktı", "kedi masanın üstünde uyudu",
    "köpek bahçede top kovaladı", "köpek bahçede koştu",
    "çocuk parkta bisiklet sürdü", "çocuk parkta kaydı",
    "kadın pazara domates aldı", "kadın pazara ekmek aldı",
    "adam otobüse bindi", "adam otobüsten indi",
    "kedi fareyi kovaladı", "kedi süt içti",
    "köpek kemiğini gömdü", "köpek topu ısırdı",
    "çocuk annesine sarıldı", "çocuk babasıyla oynadı",
    "kadın çay demledi", "kadın kitap okudu",
    "adam gazeteyi okudu", "adam kahve içti",
]


def quann_frames(n, D, oracle, words, steps, seed, eps=1e-4, min_width=1e-3,
                 sigma0=0.25, shrink=0.82, grow=1.0 / 0.82,
                 plateau_window=40, plateau_patience=1e-3, plateau_grow=1.5,
                 label_fn=None):
    """
    qcloud_quann.py QUANN.step dinamiklerini nokta-bazı D boyuta uygular:
      aday = value + yön·σ·genişlik (aralığa kırpılır)
      hata düştü → ÖDÜL : değer kabul, σ·=shrink (istismar); aralık değişmez
      hata arttı → CEZA  : geri al, yön çevir, aralığı aday tarafı yarıla, σ·=grow (keşif)
      nötr      : geri al, yön rastgele
      aralık < min_width → yakınsadı (atla)
      plato (pencerede düşüş yok) → σ·=plateau_grow
    `oracle(P) -> (err, top1_id)`: forward tabanlı hata fonksiyonu.
    `label_fn(top1_id) -> str`: top1 id'ini etikete çevirir (opsiyonel).
    """
    last_top1 = {"id": None}
    rng = np.random.default_rng(seed)
    pts = [rng.uniform(-1, 1, D) for _ in range(n)]
    lows = [np.full(D, -1.0) for _ in range(n)]
    highs = [np.full(D, 1.0) for _ in range(n)]
    dirs = [np.ones(D) for _ in range(n)]
    sigma = [sigma0] * n

    base_err, t1 = oracle(np.stack(pts))
    last_top1["id"] = t1
    frames = [{"s": 0, "pts": r3m(pts), "low": r3m(lows), "high": r3m(highs),
               "sigma": [sigma0] * n, "err": round(base_err, 5),
               "cand": None, "pi": None, "out": "start",
               "top1": label_fn(t1) if label_fn else None}]
    recent = []
    for step in range(1, steps + 1):
        pi = (step - 1) % n
        d = (step - 1) % D
        w = highs[pi][d] - lows[pi][d]
        v = pts[pi][d]
        out = None
        if w < min_width:
            out = "converged"
        else:
            c = min(highs[pi][d], max(lows[pi][d], v + dirs[pi][d] * sigma[pi] * w))
            if abs(c - v) < 1e-9:
                dirs[pi][d] = -dirs[pi][d]
                out = "noop"
            else:
                P = np.stack(pts)
                v2 = pts[pi].copy()
                v2[d] = c
                P[pi] = v2
                err, t1 = oracle(P)
                last_top1["id"] = t1
                if err < base_err - eps:
                    pts[pi] = v2
                    base_err = err
                    sigma[pi] = max(0.02, sigma[pi] * shrink)
                    out = "reward"
                elif err > base_err + eps:
                    mid = (v + c) / 2
                    if c > v:
                        highs[pi][d] = mid
                    else:
                        lows[pi][d] = mid
                    dirs[pi][d] = -dirs[pi][d]
                    sigma[pi] = min(0.6, sigma[pi] * grow)
                    out = "censure"
                else:
                    if rng.random() < 0.5:
                        dirs[pi][d] = -dirs[pi][d]
                    out = "neutral"
                recent.append(base_err)
                if len(recent) > plateau_window:
                    recent.pop(0)
                if len(recent) == plateau_window and (max(recent) - min(recent)) < plateau_patience:
                    sigma[pi] = min(0.6, sigma[pi] * plateau_grow)
        cand = None
        if out not in ("converged", "noop", "start"):
            cand = r3(np.stack(pts)[pi])
        f = {"s": step, "pts": r3m(pts), "low": r3m(lows), "high": r3m(highs),
             "sigma": [round(x, 4) for x in sigma], "err": round(base_err, 5),
             "cand": cand, "pi": pi, "out": out,
             "top1": label_fn(last_top1["id"]) if label_fn else None}
        frames.append(f)
    return frames


def quann_section():
    print("== QUANN: küçük demo modeli eğitiliyor (CPU) ...", flush=True)
    from cloudlm import CloudLM, Tok

    texts = DEMO_TEXTS * 10
    tok = Tok()
    vs = tok.build(texts)
    torch.manual_seed(0)
    np.random.seed(0)
    t0 = time.time()
    m = CloudLM(vs, d_model=32, d_cloud=64, d_cloud_dim=3, n_layers=1,
                hidden=32, point_hidden=16)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    all_t = [tok.encode(t)[:-1] for t in texts]
    m.train()
    for ep in range(30):
        ex = [(t[i:i + 3], t[i + 3]) for t in all_t for i in range(len(t) - 3)]
        np.random.shuffle(ex)
        for b in range(0, len(ex), 8):
            inp = torch.tensor([e[0] for e in ex[b:b + 8]])
            tgt = torch.tensor([e[1] for e in ex[b:b + 8]])
            lg, _, _ = m(inp, inp != 0)
            loss = F.cross_entropy(lg, tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    m.eval()
    print(f"   eğitim bitirildi: {time.time() - t0:.1f}s, {len(texts)} örnek", flush=True)

    itos = {k: v for k, v in tok.itos.items()}

    def make_oracle(m, words, tgt_id):
        def oracle(P):
            with torch.no_grad():
                cs = m.cloud(torch.tensor(P, dtype=torch.float32).unsqueeze(0),
                             torch.ones(1, P.shape[0], dtype=torch.bool))
                lg = m.word_head(cs)
                err = F.cross_entropy(lg, torch.tensor([tgt_id])).item()
            return err, int(lg.argmax(1).item())
        return oracle

    def true_pts(m, words):
        ids = [tok.stoi.get(w, 2) for w in words]
        with torch.no_grad():
            ctx = m.norm(m.encoder(m.embedding(torch.tensor([ids])) * math.sqrt(32)))
            return m.to_nd(ctx)[0].numpy()

    # --- (a) temiz görev: hedef konfigürasyon ---
    clean_words = ["kedi", "pencereden", "dışarı", "baktı"]
    target = np.array([
        [0.60, 0.50, 0.30], [-0.70, -0.60, -0.50],
        [-0.10, 0.55, 0.45], [0.75, 0.40, 0.45],
    ])
    print("   QUANN (a) temiz görev araması ...", flush=True)
    clean_frames = quann_frames(len(clean_words), 3,
                                oracle=lambda P: (float(((P - target) ** 2).sum()), -1),
                                words=clean_words, steps=300, seed=1,
                                plateau_patience=5e-4)

    # --- (b) gerçek LM: dondurulmuş küçük CloudLM, hata = CE ---
    lm_demos = []
    for sent, tgt, seed in [
        ("kedi masının üstünde", "uyudu", 2),
        ("kadın pazara domates", "aldı", 3),
    ]:
        words = sent.split()
        tgt_id = tok.stoi.get(tgt, 2)
        oracle = make_oracle(m, words, tgt_id)
        tp = true_pts(m, words)
        label_fn = lambda tid: itos.get(tid, "?") if tid is not None and tid >= 0 else None
        print(f"   QUANN (b) {sent!r} → {tgt!r} ...", flush=True)
        frames = quann_frames(len(words), 3, oracle, words, steps=500, seed=seed,
                              label_fn=label_fn)

        lm_demos.append({
            "sentence": sent,
            "words": words,
            "targetWord": tgt,
            "truePts": r3m(tp),
            "frames": frames,
            "model": f"CloudLM küçük (d_model=32, D=3, 1L, vocab {vs}), backprop'la eğitilip donduruldu",
        })

    vocab_small = {str(k): v for k, v in itos.items() if v not in ("<pad>", "<eos>", "<unk>")}
    return {
        "engine": {
            "ref": "quann_cloud/src/qcloud_quann.py (QUANN.step)",
            "rules": [
                "aday = değer + yön·σ·aralık_genişliği  (aralığa kırpılır)",
                "hata < base − ε  → ÖDÜL: değer kabul edilir, σ·0.82 (istismar)",
                "hata > base + ε  → CEZA: değer geri alınır, yön tersine döner, aralık aday tarafı yarılanır, σ·1.22 (keşif)",
                "değişim yok → NÖTR: yön rastgele çevrilir",
                "aralık < 1e-3 → yakınsadı;  plato (40 adımda < 1e-3 düşüş) → σ·1.5",
                "gradyan YOK — aday başına tam 1 forward pass",
            ],
        },
        "clean": {
            "title": "Temiz görev: hedef bulut konfigürasyonu",
            "desc": "4 kelimenin nokta-bazı aralıkları; hata = hedef konfigürasyona uzaklık. "
                    "0th-order arama, backprop kullanmadan hedefe yakınsıyor.",
            "words": clean_words,
            "target": r3m(target),
            "frames": clean_frames,
        },
        "lm": lm_demos,
        "vocab": vocab_small,
    }


# ═══════════════════════════════════════════════════════════════
# 3) BULGULAR
# ═══════════════════════════════════════════════════════════════

def findings_section():
    return {
        "cards": [
            {"title": "Held-out PPL (2000 cümle, 200K corpus)",
             "bars": [
                 {"label": "Causal baseline (d=384, 4L, 22.7M)", "v": 2.92},
                 {"label": "CloudLM (D=32, 21.0M)", "v": 6.45},
             ],
             "note": "Aynı parametre bütçesi; farkın kökü kapasite (D boyutu) — "
                     "sensitivity_analysis ölçümü."},
            {"title": "R1 kapasite taraması (20K held-out)",
             "bars": [
                 {"label": "Baseline", "v": 10.80},
                 {"label": "CloudLM D=32", "v": 201.18},
                 {"label": "CloudLM D=64", "v": 181.47},
                 {"label": "CloudLM D=128", "v": 189.47},
             ],
             "note": "Boyut artışı monoton fayda vermedi (D=128 geri gitti) — "
                     "200K'a varyant taşınmadı."},
            {"title": "QUANN (backprop'suz) — dürüst sınır",
             "bars": [
                 {"label": "XOR (backprop'suz)", "v": 100.0, "pct": True},
                 {"label": "Gerçek LM (backprop'suz)", "v": 0.0, "pct": True,
                  "label2": "≈ rastgele"},
                 {"label": "Hybrid (QUANN nokta + BP okuyucu) fraud AUC", "v": 97.45, "pct": True},
             ],
             "note": "Savunulabilir iddia: küçük/yorumlanabilir modeller için backprop'suz "
                     "eğitim. Detay: quann_cloud/FINDINGS.md."},
            {"title": "Edge / boyut",
             "bars": [
                 {"label": "FP32 (MB)", "v": 84.0},
                 {"label": "INT8 (MB) — kayıpsız", "v": 21.0},
             ],
             "note": "INT8 ile 84→21MB; ONNX/CoreML export çalışıyor."},
        ],
        "note": "Sayılar RESEARCH.md / quann_cloud/FINDINGS.md'den; "
                "bu görünüm bir özet kartıdır.",
    }


# ═══════════════════════════════════════════════════════════════
# ANA
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    data = {"meta": {"generated": time.strftime("%Y-%m-%d %H:%M"), "root": os.path.basename(ROOT)}}

    print("── Quann-3D 3D görselleştirme: veri dışa aktarma ──\n", flush=True)

    cloud = None
    try:
        cloud = cloud_section()
    except Exception as e:
        print(f"   CLOUD bölümü atlandı: {e}", flush=True)

    quann = None
    try:
        quann = quann_section()
    except Exception as e:
        print(f"   QUANN bölümü atlandı: {e}", flush=True)

    findings = findings_section()

    if cloud:
        data["cloud"] = cloud
    if quann:
        data["quann"] = quann
    data["findings"] = findings

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    tpl_path = os.path.join(HERE, "template.html")
    with open(tpl_path, "r", encoding="utf-8") as f:
        tpl = f.read()
    assert "/*__DATA__*/null" in tpl, "template.html'de /*__DATA__*/null yer tutucusu yok"
    out = tpl.replace("/*__DATA__*/null", payload)
    out_path = os.path.join(HERE, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(out)
    print(f"\nTamam: {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB) — "
          f"{time.time() - t0:.0f}s", flush=True)
    if cloud is None and quann is None:
        sys.exit("Hem CLOUD hem QUANN üretilemedi — index.html boş veriyle yazıldı!")


if __name__ == "__main__":
    main()
