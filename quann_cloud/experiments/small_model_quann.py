"""
QUANN-Cloud — Çok Küçük Model: QUANN vs Backprop
=================================================

Amaç: QUANN'ın GERÇEKTEN öğrendiğini göstermek.
- ~10K parametre → QUANN anlamlı ilerleme yapabilir
- Aynı model backprop ile → referans
- 15 epoch, aynı veri → adil karşılaştırma
"""

import sys, os, time, math, copy, random
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tokenizers import Tokenizer
from qcloud_network import QCloudNetwork
from qcloud_quann import QUANN, QUANNConfig


def load_corpus(path, max_lines=3000):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            t = line.strip()
            if len(t) > 50: lines.append(t)
    return lines


def make_examples(tokenizer, texts, seq_len=8, max_ex=100, vocab=100):
    examples = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        for i in range(0, len(ids)-1, seq_len):
            chunk = ids[i:i+seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [0]*(seq_len-len(chunk))
            tgt = ids[i+seq_len] if i+seq_len < len(ids) else 1
            examples.append((
                torch.tensor([x % vocab for x in chunk]),
                torch.tensor(tgt % vocab)
            ))
            if len(examples) >= max_ex: return examples
    return examples


def evaluate(model, examples, device, n=50):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inp, tgt in examples[:n]:
            inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
            mask = (inp != 0).bool()
            logits, _, _ = model(inp, mask)
            total += F.cross_entropy(logits, tgt).item()
    return math.exp(min(total/n, 20))


def main():
    print("="*60)
    print("  QUANN-Cloud — Çok Küçük Model: QUANN vs Backprop")
    print("="*60)

    tokenizer = Tokenizer.from_file(os.path.join(ROOT, "data/bpe_5k.json"))
    VOCAB = 100
    print(f"\n  Vocab: {VOCAB} (küçük)")

    corpus = load_corpus(os.path.join(ROOT, "data/akademik_100k.txt"), 3000)
    print(f"  Corpus: {len(corpus)} makale")

    examples = make_examples(tokenizer, corpus, seq_len=8, max_ex=100, vocab=VOCAB)
    print(f"  Örnekler: {len(examples)}")

    # Çok küçük model: ~10K param
    model = QCloudNetwork(vocab_size=VOCAB, d_model=16, d_cloud=16, d_cloud_dim=2, n_layers=1)
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model}")
    print(f"  Parameters: {params:,}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    base_ppl = evaluate(model, examples, device)
    print(f"\n  Baseline PPL: {base_ppl:.2f}")

    # BACKPROP
    print(f"\n{'='*60}\n  BACKPROP (referans)\n{'='*60}")
    model_bp = copy.deepcopy(model).to(device)
    t0 = time.time()
    opt = torch.optim.AdamW(model_bp.parameters(), lr=1e-2)
    model_bp.train()
    for ep in range(1, 16):
        total = 0.0
        order = list(range(len(examples)))
        random.shuffle(order)
        for b in range(0, len(order), 8):
            idxs = order[b:b+8]
            inps = torch.stack([examples[i][0] for i in idxs]).to(device)
            tgts = torch.stack([examples[i][1] for i in idxs]).to(device)
            mask = (inps != 0).bool()
            logits, _, _ = model_bp(inps, mask)
            loss = F.cross_entropy(logits, tgts)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*len(idxs)
        ppl = math.exp(min(total/len(examples), 20))
        if ep % 5 == 0 or ep == 1:
            print(f"    BP E{ep:2d}: PPL={ppl:.2f}")
    bp_time = time.time()-t0
    bp_final = evaluate(model_bp, examples, device)
    print(f"  BP: {bp_time:.1f}s → PPL={bp_final:.2f}")

    # QUANN
    print(f"\n{'='*60}\n  QUANN (backprop'suz)\n{'='*60}")
    model_q = copy.deepcopy(model).to(device)
    
    cfg = QUANNConfig(
        sigma0=0.5,
        k_weights=20,   # 200 param/adım (10K model için %20)
        minibatch=4,
        plateau_window=10,
        seed=42,
    )

    def loss_fn(m, inputs, targets):
        inputs, targets = inputs.to(device), targets.to(device)
        mask = (inputs != 0).bool()
        logits, _, _ = m(inputs, mask)
        return F.cross_entropy(logits, targets).item()

    quann = QUANN(model_q, device, cfg=cfg, loss_fn=loss_fn)
    t0 = time.time()
    quann.fit(examples, epochs=10, verbose=True)
    quann_time = time.time()-t0
    q_final = evaluate(model_q, examples, device)
    print(f"  QUANN: {quann_time:.1f}s → PPL={q_final:.2f}")

    # SONUÇ
    print(f"\n{'='*60}\n  📊 SONUÇ\n{'='*60}")
    print(f"  {'':20s} {'Backprop':>12s} {'QUANN':>12s}")
    print(f"  {'-'*44}")
    print(f"  {'Baseline PPL':20s} {base_ppl:>12.2f} {base_ppl:>12.2f}")
    print(f"  {'Final PPL':20s} {bp_final:>12.2f} {q_final:>12.2f}")
    print(f"  {'İyileşme':20s} {base_ppl/bp_final:>11.2f}x {base_ppl/q_final:>11.2f}x")
    print(f"  {'Süre':20s} {bp_time:>11.1f}s {quann_time:>11.1f}s")
    if quann_time > 0:
        print(f"  {'Hız (BP/QUANN)':20s} {'1.0x':>12s} {bp_time/quann_time:>11.2f}x")
    
    info = quann.info()
    print(f"\n  QUANN: width={info['mean_width']:.4f}, conv={info['pct_converged']*100:.1f}%, σ={info['sigma_mean']:.3f}")

    # Metin
    print(f"\n  --- Metin Üretimi ---")
    prompt = "Bu bir test cümlesidir."
    enc = [x % VOCAB for x in tokenizer.encode(prompt).ids]
    tokens = torch.tensor([enc], device=device)
    gen_q = model_q.generate(tokens, max_new=10, temperature=0.8)
    gen_bp = model_bp.generate(tokens, max_new=10, temperature=0.8)
    full_vocab = len(tokenizer.get_vocab())
    print(f"  QUANN:    {tokenizer.decode([x%full_vocab for x in gen_q[0].tolist()])}")
    print(f"  Backprop: {tokenizer.decode([x%full_vocab for x in gen_bp[0].tolist()])}")


if __name__ == "__main__":
    main()
