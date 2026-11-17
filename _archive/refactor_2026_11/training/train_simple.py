"""
CloudLM Eğitim - Basitleştirilmiş Versiyon
==========================================
Hafıza dostu, epoch bazlı eğitim
"""

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
import numpy as np
from cloudlm import BPETokenizer, CloudLM

def load_txt(path, max_lines=None, min_len=20, max_len=300):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            t = line.strip()
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts

def main():
    # Parametreler
    corpus = "data/turkish_wiki_1500k.txt"
    max_texts = 200000
    epochs = 10
    batch_size = 64
    lr = 2e-4
    seq_len = 20
    d_model = 256
    d_cloud = 512
    d_dim = 64
    layers = 8
    save_dir = "checkpoints/wiki_checkpoints_r2_cloud_attn_200k_test"
    log_file = "logs/run_r2_cloud_attn_200k_test.log"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print(f"Loading tokenizer...")
    tok = BPETokenizer.load(os.path.join(save_dir, "wiki_bpe.json"))
    
    print(f"Loading corpus...")
    texts = load_txt(corpus, max_lines=max_texts)
    print(f"  {len(texts)} cümle")
    
    print(f"Tokenizing...")
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= seq_len + 1]
    print(f"  {len(all_tokens)} cümle (≥{seq_len+1} token)")
    
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    
    model = CloudLM(
        tok.num_tokens,
        d_model=d_model,
        d_cloud=d_cloud,
        d_cloud_dim=d_dim,
        n_layers=layers,
        pool="max",
        cloud_attn=True,
        cloud_attn_heads=8,
        attn_scale_init=1.0,
    )
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model: {n_params:,} params")
    print(f"  Device: {device}")
    print(f"  Starting training...")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    t0 = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        
        # Shuffle
        random.shuffle(all_tokens)
        
        # Batch processing
        for i in range(0, len(all_tokens) - seq_len, batch_size):
            batch_tokens = []
            for j in range(i, min(i + batch_size, len(all_tokens) - seq_len)):
                sent = all_tokens[j]
                batch_tokens.append((sent[:seq_len], sent[seq_len]))
            
            if len(batch_tokens) < 2:
                continue
            
            inputs = torch.tensor([b[0] for b in batch_tokens], device=device)
            targets = torch.tensor([b[1] for b in batch_tokens], device=device)
            mask = (inputs != 0) & (inputs != eos_id)
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            
            if not torch.isfinite(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        
        print(f"  Epoch {epoch+1:3d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s", flush=True)
        
        # Save checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {"d_model": d_model, "d_cloud": d_cloud, "d_cloud_dim": d_dim, "n_layers": layers},
            "ppl": ppl,
        }
        torch.save(checkpoint, os.path.join(save_dir, f"wiki_cloud_epoch{epoch+1}.pt"))
        
        if epoch == 0 or ppl < checkpoint.get("best_ppl", float('inf')):
            checkpoint["best_ppl"] = ppl
            torch.save(checkpoint, os.path.join(save_dir, "wiki_cloud_best.pt"))
    
    total = time.time() - t0
    print(f"\n  ✅ Training complete: {total:.0f}s ({total/60:.1f}dk)")

if __name__ == "__main__":
    main()
