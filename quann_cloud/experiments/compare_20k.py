"""
20K Corpus Karşılaştırması: CloudLM (backprop) vs QUANN-Cloud Hibrit
======================================================================

Aynı corpus (ilk 20K), aynı BPE (20K vocab), aynı held-out (20K-22K),
aynı eval protokolü (ilk 10 token → next token).

- Control : standart CloudLM — to_nd projeksiyonu, hepsi backprop (~21M aktif)
- Hibrit  : QUANN-Cloud — token kimliği bazlı QUANN noktaları (arama) +
            backprop ile eğitilen cloud_encoder+word_head (~12.6M aktif)
"""

import sys
import os
import math
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'quann_cloud', 'src'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

from qcloud_network import QCloudNetwork
from train_wiki import BPETokenizer, CloudLM

SEQ_LEN = 10
BATCH = 64


def load_eval_texts(path, start, n):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    texts = [t for t in lines if 20 <= len(t) <= 300]
    return texts[start:start + n]


def eval_ppl(model, tok, texts, device, use_quann_points=False):
    """İlk pencere → next token; batch'li CE; PPL."""
    model.eval()
    eos_id = 1
    examples = []
    for t in texts:
        ids = tok.encode(t)
        if len(ids) < SEQ_LEN + 1:
            continue
        examples.append((ids[:SEQ_LEN], ids[SEQ_LEN]))
    
    total_loss, total_tokens = 0.0, 0
    with torch.no_grad():
        for b in range(0, len(examples), BATCH):
            chunk = examples[b:b + BATCH]
            tokens = torch.tensor([e[0] for e in chunk], device=device)
            targets = torch.tensor([e[1] for e in chunk], device=device)
            mask = (tokens != 0) & (tokens != eos_id)
            if use_quann_points:
                pts = model.points_for(tokens, mask)
                logits, _, _ = model.forward(tokens, mask, points_override=pts)
            else:
                logits, _, _ = model.forward(tokens, mask)
            total_loss += F.cross_entropy(logits, targets, reduction='sum').item()
            total_tokens += len(chunk)
    return math.exp(total_loss / max(total_tokens, 1))


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = BPETokenizer.load(os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json"))
    print(f"Vocab: {tok.num_tokens}")
    
    eval_texts = load_eval_texts(os.path.join(ROOT, "data", "turkish_sentences.txt"), 20000, 2000)
    print(f"Held-out: {len(eval_texts)} cümle (20K-22K)")
    
    results = {}
    
    # ── Control: CloudLM (adil: tek pencere, hibritle aynı güncelleme) ──
    cpath = os.path.join(ROOT, "quann_cloud", "experiments", "control_20k_single", "wiki_cloud_best.pt")
    if os.path.exists(cpath):
        ck = torch.load(cpath, map_location="cpu", weights_only=False)
        m = CloudLM(ck["vocab_size"], **ck["config"])
        m.load_state_dict(ck["model_state"])
        m.to(device)
        ppl = eval_ppl(m, tok, eval_texts, device)
        results["CloudLM (adil control)"] = ppl
        print(f"\n  CloudLM adil control (epoch {ck['epoch']}): held-out PPL = {ppl:.2f}")
        del m
    else:
        print(f"\n  [!] Adil control checkpoint yok: {cpath}")

    # ── Control: CloudLM (tüm pencereler — referans) ──
    cpath2 = os.path.join(ROOT, "quann_cloud", "experiments", "control_20k", "wiki_cloud_best.pt")
    if os.path.exists(cpath2):
        ck = torch.load(cpath2, map_location="cpu", weights_only=False)
        m = CloudLM(ck["vocab_size"], **ck["config"])
        m.load_state_dict(ck["model_state"])
        m.to(device)
        ppl = eval_ppl(m, tok, eval_texts, device)
        results["CloudLM (tüm pencereler)"] = ppl
        print(f"  CloudLM tüm pencereler (epoch {ck['epoch']}): held-out PPL = {ppl:.2f}")
        del m
    
    # ── Hibrit: QUANN-Cloud ──
    hpath = os.path.join(ROOT, "quann_cloud", "logs", "qcloud_tiny_hybrid_20000x10ep.pt")
    if os.path.exists(hpath):
        ck = torch.load(hpath, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        m = QCloudNetwork(
            vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
            d_cloud=cfg["d_cloud"], d_cloud_dim=cfg["D"],
            n_layers=cfg["n_layers"], bidirectional=cfg.get("bidirectional", True),
        )
        m.load_state_dict(ck["model_state"])
        # QUANN ağırlıklarını geri yükle
        import numpy as np
        for entry in ck.get("token_weights", []):
            tid = entry["token_id"]
            w = m.get_weight(tid)
            w.low = np.array(entry["low"])
            w.high = np.array(entry["high"])
            w.value = np.array(entry["value"])
            w.direction = np.array(entry["direction"])
            w.sigma = entry["sigma"]
        m.to(device)
        ppl = eval_ppl(m, tok, eval_texts, device, use_quann_points=True)
        results["QUANN-Cloud (hibrit)"] = ppl
        print(f"  QUANN-Cloud hibrit (epoch {ck['epoch']}): held-out PPL = {ppl:.2f} "
              f"({len(ck.get('token_weights', []))} token ağırlığı)")
        
        # Üretim
        print("\n  ÜRETİM (hibrit, temp=0.8):")
        for p in ["Türkiye'nin başkenti", "bilgisayar bilimlerinde",
                  "sanat tarihi açısından", "ekonomik büyüme",
                  "demokrasi ve insan"]:
            ids = torch.tensor([tok.encode(p)], device=device)
            out = m.generate(ids, max_new=12, temperature=0.8)
            print(f"    {p[:28]:30s} → {tok.decode(out[0].tolist())[:70]}")
        del m
    else:
        print(f"  [!] Hibrit checkpoint yok: {hpath}")
    
    # ── Tablo ──
    print(f"\n{'='*60}")
    print("  20K CORPUS — HELD-OUT PPL KARŞILAŞTIRMASI")
    print(f"{'='*60}")
    print(f"  {'Model':28s} {'Held-out PPL':>14s}")
    print(f"  {'Random (untrained)':28s} {tok.num_tokens:>14,}")
    for name, ppl in results.items():
        print(f"  {name:28s} {ppl:>14.2f}")
    print()


if __name__ == "__main__":
    main()
