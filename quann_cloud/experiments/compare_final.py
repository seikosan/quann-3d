"""
Nihai Karşılaştırma — ortak held-out (5.5M wiki dosyasından 2K cümle)
======================================================================

Tüm modeller aynı held-out setiyle ölçülür (eğitim setleriyle kesişim yok —
doğrulandı):

- CloudLM 200K   : turkish_sentences.txt (200K) üzerinde eğitimli
- Baseline 200K  : aynı corpus, causal Transformer
- Hibrit 200K    : 1.5M wiki'nin ilk 200K'sı (QUANN noktalar + backprop)
- CloudLM 1.5M   : turkish_wiki_1500k.txt (ölçek testi)
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

    # Her modelin KENDİ BPE'si (1.5M modeli farklı BPE ile eğitimli)
    bpe_200k = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json")
    bpe_1500k = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_wiki1500k", "wiki_bpe.json")
    tok = BPETokenizer.load(bpe_200k)
    tok_1500k = BPETokenizer.load(bpe_1500k) if os.path.exists(bpe_1500k) else tok
    print(f"Vocab: 200K-BPE={tok.num_tokens} | 1.5M-BPE={tok_1500k.num_tokens}")

    eval_texts = load_eval_texts(os.path.join(ROOT, "data", "turkish_wiki_sentences.txt"), 0, 2000)
    print(f"Ortak held-out: {len(eval_texts)} cümle (5.5M wiki dosyası başı)")

    results = {}

    def run(name, path, model_cls, use_quann=False, tok_override=None):
        if not os.path.exists(path):
            print(f"  [!] {name}: checkpoint yok ({path})")
            return
        t = tok_override or tok
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if model_cls is CloudLM:
            m = CloudLM(ck["vocab_size"], **cfg)
        else:
            m = QCloudNetwork(
                vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                d_cloud=cfg["d_cloud"], d_cloud_dim=cfg["D"],
                n_layers=cfg["n_layers"], bidirectional=cfg.get("bidirectional", True),
            )
            import numpy as np
            for entry in ck.get("token_weights", []):
                w = m.get_weight(entry["token_id"])
                w.low = np.array(entry["low"]); w.high = np.array(entry["high"])
                w.value = np.array(entry["value"]); w.direction = np.array(entry["direction"])
                w.sigma = entry["sigma"]
        m.load_state_dict(ck["model_state"])
        m.to(device)
        ppl = eval_ppl(m, t, eval_texts, device, use_quann_points=use_quann)
        results[name] = ppl
        print(f"  {name} (epoch {ck['epoch']}): held-out PPL = {ppl:.2f}")
        del m

    run("CloudLM 200K", os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_cloud_best.pt"), CloudLM)
    run("Hibrit QUANN 200K", os.path.join(ROOT, "quann_cloud", "logs",
                                          "qcloud_tiny_hybrid_200000x10ep.pt"), QCloudNetwork,
        use_quann=True)
    run("CloudLM 1.5M (20ep)", os.path.join(ROOT, "checkpoints", "wiki_checkpoints_wiki1500k",
                                             "wiki_cloud_best.pt"), CloudLM,
        tok_override=tok_1500k)

    # Baseline (CausalLM) — ayrı sınıf
    from train_baseline import build_model
    p = os.path.join(ROOT, "checkpoints", "baseline_200k", "baseline_50.pt")
    if os.path.exists(p):
        ck = torch.load(p, map_location="cpu", weights_only=False)
        bcfg = ck["config"]
        m = build_model(bcfg["arch"], ck["vocab_size"], bcfg["d_model"], bcfg["n_layers"])
        m.load_state_dict(ck["model_state"])
        m.to(device)
        # CausalLM: son pozisyondan tek token
        m.eval()
        eos_id = 1
        total_loss, total_tokens = 0.0, 0
        with torch.no_grad():
            ex = []
            for t in eval_texts:
                ids = tok.encode(t)
                if len(ids) < SEQ_LEN + 1:
                    continue
                ex.append((ids[:SEQ_LEN], ids[SEQ_LEN]))
            for b in range(0, len(ex), BATCH):
                chunk = ex[b:b + BATCH]
                tokens = torch.tensor([e[0] for e in chunk], device=device)
                targets = torch.tensor([e[1] for e in chunk], device=device)
                mask = (tokens != 0) & (tokens != eos_id)
                logits = m(tokens, mask)
                total_loss += F.cross_entropy(logits[:, -1, :], targets, reduction='sum').item()
                total_tokens += len(chunk)
        ppl = math.exp(total_loss / max(total_tokens, 1))
        results["Baseline 200K (causal)"] = ppl
        print(f"  Baseline 200K (causal, epoch {ck['epoch']}): held-out PPL = {ppl:.2f}")
        del m

    # Üretim (CloudLM 1.5M + hibrit)
    print("\n  ÜRETİM ÖRNEKLERİ (temp=0.8):")
    prompts = ["Türkiye'nin başkenti", "bilgisayar bilimlerinde", "ekonomik büyüme"]
    for name, path, cls, quann, t in [
        ("CloudLM 1.5M", os.path.join(ROOT, "checkpoints", "wiki_checkpoints_wiki1500k",
                                       "wiki_cloud_best.pt"), CloudLM, False, tok_1500k),
        ("Hibrit 200K", os.path.join(ROOT, "quann_cloud", "logs",
                                     "qcloud_tiny_hybrid_200000x10ep.pt"), QCloudNetwork, True, tok),
    ]:
        if not os.path.exists(path):
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        if cls is CloudLM:
            m = CloudLM(ck["vocab_size"], **cfg)
        else:
            m = QCloudNetwork(vocab_size=cfg["vocab_size"], d_model=cfg["d_model"],
                              d_cloud=cfg["d_cloud"], d_cloud_dim=cfg["D"],
                              n_layers=cfg["n_layers"])
            import numpy as np
            for entry in ck.get("token_weights", []):
                w = m.get_weight(entry["token_id"])
                w.low = np.array(entry["low"]); w.high = np.array(entry["high"])
                w.value = np.array(entry["value"]); w.direction = np.array(entry["direction"])
                w.sigma = entry["sigma"]
        m.load_state_dict(ck["model_state"])
        m.to(device)
        print(f"  {name}:")
        for p2 in prompts:
            ids = torch.tensor([t.encode(p2)], device=device)
            out = m.generate(ids, max_new=12, temperature=0.8)
            print(f"    {p2[:26]:28s} → {t.decode(out[0].tolist())[:70]}")
        del m

    print(f"\n{'='*60}")
    print("  NİHAI TABLO — ortak held-out (5.5M wiki, 2K cümle)")
    print(f"{'='*60}")
    print(f"  {'Model':28s} {'Held-out PPL':>14s}")
    print(f"  {'Random (20K vocab)':28s} {20000:>14,}")
    for name, ppl in results.items():
        print(f"  {name:28s} {ppl:>14.2f}")
    print()


if __name__ == "__main__":
    main()
