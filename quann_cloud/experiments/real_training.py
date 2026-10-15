"""
QUANN-Cloud — Gerçek Veri ile Eğitim Deneyi
=============================================

BellaTurca AkademikDerlem (100K makale) + BPE tokenizer ile
backprop'suz QUANN eğitimi.

Çalıştır:
    python3 experiments/real_training.py
"""

import sys
import os
import time
import math
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tokenizers import Tokenizer
from qcloud_network import QCloudNetwork
from qcloud_trainer import QCloudTrainer


def load_corpus(path: str, max_lines: int = 50_000) -> list[str]:
    """Corpus yükle."""
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            text = line.strip()
            if len(text) > 50:  # Çok kısa satırları atla
                lines.append(text)
    return lines


def main():
    print("=" * 60)
    print("  QUANN-Cloud — Gerçek Veri ile Eğitim")
    print("=" * 60)
    
    # 1. Tokenizer yükle
    print("\n[1/5] BPE Tokenizer yükleniyor...")
    tokenizer = Tokenizer.from_file(os.path.join(ROOT, "data/bpe_5k.json"))
    vocab_size = len(tokenizer.get_vocab())
    print(f"  Vocab size: {vocab_size}")
    
    # 2. Corpus yükle
    print("\n[2/5] Corpus yükleniyor (50K makale)...")
    corpus = load_corpus(os.path.join(ROOT, "data/akademik_100k.txt"), max_lines=50_000)
    print(f"  {len(corpus)} makale")
    
    # 3. Tokenize
    print("\n[3/5] Tokenize ediliyor...")
    t0 = time.time()
    all_token_ids = []
    for text in corpus[:5000]:  # 5000 makale ile başla
        enc = tokenizer.encode(text)
        all_token_ids.append(enc.ids)
    print(f"  {len(all_token_ids)} makale tokenize edildi ({time.time()-t0:.1f}s)")
    
    # 4. Model oluştur
    print("\n[4/5] Model oluşturuluyor...")
    model = QCloudNetwork(
        vocab_size=vocab_size,
        d_model=64,
        d_cloud=128,
        d_cloud_dim=8,
        n_layers=2,
        bidirectional=True,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {model}")
    print(f"  Parameters: {params:,}")
    
    # 5. Device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    print(f"  Device: {device}")
    
    # 6. Eğitim
    print("\n[5/5] QUANN eğitimi başlıyor...")
    
    # Basit: 5000 makaleyi 10'ar token'lık chunk'lara böl
    seq_len = 10
    batch_size = 8
    max_epochs = 3
    
    # Token ID'lerini hazırla
    samples = []
    for ids in all_token_ids:
        for i in range(0, len(ids) - 1, seq_len):
            chunk = ids[i:i+seq_len]
            if len(chunk) == seq_len:
                target = ids[i+seq_len] if i+seq_len < len(ids) else 1
                samples.append((chunk, target))
    
    print(f"  {len(samples)} örnek hazır")
    
    best_ppl = float('inf')
    
    for epoch in range(1, max_epochs + 1):
        t0 = time.time()
        
        # Her epoch'ta token weights sıfırla
        model.reset_token_weights()
        
        # Batch'ler
        total_rewarded = 0
        total_censured = 0
        total_unchanged = 0
        total_loss = 0.0
        total_tokens = 0
        
        for batch_start in range(0, min(len(samples), 200), batch_size):
            batch = samples[batch_start:batch_start + batch_size]
            
            tokens = torch.tensor(
                [s[0] + [0] * (seq_len - len(s[0])) for s in batch],
                device=device
            )
            mask = (tokens != 0).bool()
            targets = torch.tensor([s[1] for s in batch], device=device)
            
            # Forward
            word_logits, _, _ = model(tokens, mask)
            loss = F.cross_entropy(word_logits, targets)
            total_loss += loss.item() * len(batch)
            total_tokens += len(batch)
            
            # QUANN update (ilk batch ile test)
            if batch_start == 0:
                base_error = loss.item()
                stats = model.update_token_weights(
                    tokens[:1], mask[:1], targets[0].item()
                )
                total_rewarded += stats['rewarded']
                total_censured += stats['censured']
                total_unchanged += stats['unchanged']
        
        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 100))
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch}/{max_epochs}: PPL={ppl:.2f}, "
              f"R={total_rewarded}, C={total_censured}, "
              f"Time={elapsed:.1f}s")
        
        if ppl < best_ppl:
            best_ppl = ppl
    
    print(f"\n✅ Eğitim tamam! Best PPL: {best_ppl:.2f}")
    print(f"   (Referans: backprop ile aynı model ~4-5 PPL beklenir)")
    
    # Metin üretimi testi
    print("\n--- Metin Üretimi Testi ---")
    prompt = "Bu bir test cümlesidir."
    enc = tokenizer.encode(prompt)
    tokens = torch.tensor([enc.ids], device=device)
    generated = model.generate(tokens, max_new=10, temperature=0.8)
    gen_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt: {prompt}")
    print(f"  Üretilen: {gen_text[:100]}...")


if __name__ == "__main__":
    main()
