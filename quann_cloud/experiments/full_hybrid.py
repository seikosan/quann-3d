"""
QUANN-Cloud — Full Corpus Hibrit Eğitim
=========================================

100K akademik makale + BPE (5000 vocab) + QCloudNetwork
- Transformer: Backprop (AdamW)
- Nokta Bulut: QUANN (token weights)

Çalıştır:
    python3 experiments/full_hybrid.py
"""

import sys, os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tokenizers import Tokenizer
from qcloud_network import QCloudNetwork
from qcloud_weight import QCloudWeight


def load_corpus(path, max_lines=100_000):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            t = line.strip()
            if len(t) > 50: lines.append(t)
    return lines


def make_examples(tokenizer, texts, seq_len=12, max_ex=2000):
    examples = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        for i in range(0, len(ids)-1, seq_len):
            chunk = ids[i:i+seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [0]*(seq_len-len(chunk))
            tgt = ids[i+seq_len] if i+seq_len < len(ids) else 1
            examples.append((torch.tensor(chunk), torch.tensor(tgt)))
            if len(examples) >= max_ex: return examples
    return examples


def evaluate(model, examples, device, n=100):
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
    print("  QUANN-Cloud — Full Corpus Hibrit Eğitim")
    print("  100K Akademik Makale + BPE(5000) + QCloudNetwork")
    print("="*60)

    # 1. Tokenizer
    print("\n[1/6] BPE Tokenizer (5000 vocab)...")
    tokenizer = Tokenizer.from_file(os.path.join(ROOT, "data/bpe_5k.json"))
    VOCAB = len(tokenizer.get_vocab())
    print(f"  Vocab: {VOCAB}")

    # 2. Corpus
    print("\n[2/6] Corpus (100K makale)...")
    t0 = time.time()
    corpus = load_corpus(os.path.join(ROOT, "data/akademik_100k.txt"), 100_000)
    print(f"  {len(corpus)} makale ({time.time()-t0:.1f}s)")

    # 3. Örnekler
    print("\n[3/6] Örnekler (2000, seq_len=12)...")
    t0 = time.time()
    examples = make_examples(tokenizer, corpus, seq_len=12, max_ex=2000)
    print(f"  {len(examples)} örnek ({time.time()-t0:.1f}s)")

    # 4. Model (orta boyut)
    print("\n[4/6] Model...")
    model = QCloudNetwork(
        vocab_size=VOCAB,
        d_model=64,
        d_cloud=128,
        d_cloud_dim=8,
        n_layers=2,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  {model}")
    print(f"  Parameters: {params:,}")

    # Token weights (nokta bulut) — ilk 100 unique token
    unique_tokens = set()
    for inp, _ in examples:
        for t in inp.tolist():
            if t != 0: unique_tokens.add(t)
    tokens_to_track = list(unique_tokens)[:100]
    for tid in tokens_to_track:
        model.get_weight(tid)
    print(f"  Token weights: {len(model.token_weights)} (D={model.D})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")

    # 5. Baseline
    base_ppl = evaluate(model, examples, device, n=100)
    print(f"\n  Baseline PPL: {base_ppl:.2f}")

    # 6. Hibrit Eğitim
    print(f"\n{'='*60}")
    print(f"  EĞİTİM: Backprop (transformer) + QUANN (nokta bulut)")
    print(f"  10 epoch, lr=1e-3, batch=16")
    print(f"{'='*60}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    
    best_ppl = float('inf')
    t0 = time.time()
    
    for epoch in range(1, 11):
        total_loss = 0.0
        order = list(range(len(examples)))
        random.shuffle(order)
        
        for b in range(0, len(order), 16):
            idxs = order[b:b+16]
            inps = torch.stack([examples[i][0] for i in idxs]).to(device)
            tgts = torch.stack([examples[i][1] for i in idxs]).to(device)
            mask = (inps != 0).bool()
            
            logits, _, _ = model(inps, mask)
            loss = F.cross_entropy(logits, tgts)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(idxs)
        
        # QUANN: token weights pertürbasyonu (basit)
        for tid, w in list(model.token_weights.items())[:20]:
            old = w.value.copy()
            pert = np.random.uniform(-0.05, 0.05, w.D)
            w.value = np.clip(old + pert, w.low, w.high)
        
        avg_loss = total_loss / len(examples)
        ppl = math.exp(min(avg_loss, 20))
        elapsed = time.time() - t0
        
        if ppl < best_ppl:
            best_ppl = ppl
        
        print(f"  E{epoch:2d}: PPL={ppl:.2f}, best={best_ppl:.2f}, "
              f"tokens={len(model.token_weights)}, {elapsed:.0f}s")
    
    total_time = time.time() - t0
    final_ppl = evaluate(model, examples, device, n=100)
    
    # Sonuç
    print(f"\n{'='*60}")
    print(f"  📊 SONUÇ")
    print(f"{'='*60}")
    print(f"  Baseline PPL:  {base_ppl:.2f}")
    print(f"  Best PPL:      {best_ppl:.2f}")
    print(f"  Final PPL:     {final_ppl:.2f}")
    print(f"  İyileşme:      {base_ppl/final_ppl:.2f}x")
    print(f"  Toplam süre:   {total_time:.1f}s")
    print(f"  Token weights: {len(model.token_weights)}")
    
    converged = sum(1 for w in model.token_weights.values() if w.is_converged())
    print(f"  Converged:     {converged}/{len(model.token_weights)}")
    
    # Metin üretimi
    print(f"\n  --- Metin Üretimi ---")
    prompts = [
        "Türkiye bir coğrafyadır.",
        "Bilim ve teknoloji gelişiyor.",
        "Eğitim sistemi reform gerektiriyor.",
    ]
    for prompt in prompts:
        enc = tokenizer.encode(prompt)
        tokens = torch.tensor([enc.ids], device=device)
        generated = model.generate(tokens, max_new=15, temperature=0.8)
        gen_text = tokenizer.decode(generated[0].tolist())
        print(f"  {prompt[:40]:40s} → {gen_text[:60]}")
    
    # Kaydet
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    save_path = os.path.join(ROOT, "logs/full_hybrid_model.pt")
    torch.save({
        'model_state': model.state_dict(),
        'config': {'vocab': VOCAB, 'd_model': 64, 'd_cloud': 128, 'D': 8},
        'ppl': final_ppl,
        'best_ppl': best_ppl,
        'token_weights': {tid: w.info() for tid, w in model.token_weights.items()},
        'tokenizer': tokenizer.to_str(),
    }, save_path)
    print(f"\n  💾 Model: {save_path}")


if __name__ == "__main__":
    main()
