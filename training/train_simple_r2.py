"""Simple CloudLM training - minimal loop for debugging"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import math
import time
from cloudlm import BPETokenizer, CloudLM

print("=" * 60)
print("  SIMPLE CLOUDLM TRAINING (R2 Config)")
print("=" * 60)

# Load data
with open('data/turkish_wiki_1500k.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]  # Full 1.5M
print(f"Corpus: {len(texts)} cümle")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20/wiki_bpe.json')
print(f"BPE: {tok.num_tokens} tokens")

# Create model
cfg = {
    "d_model": 256, "d_cloud": 512, "d_cloud_dim": 64,
    "n_layers": 8, "pool": "max", "cloud_attn": True, "cloud_attn_heads": 8
}
model = CloudLM(tok.num_tokens, **cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

# Encode texts
all_tokens = [tok.encode(t) for t in texts]
all_tokens = [t for t in all_tokens if len(t) >= 11]
print(f"Valid: {len(all_tokens)} cümle")

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 10
batch_size = 64

best_ppl = float('inf')
t0 = time.time()

for epoch in range(1, 101):
    epoch_loss, n_batches = 0, 0
    
    for i in range(0, min(len(all_tokens), 1000), batch_size):
        batch_data = all_tokens[i:i+batch_size]
        inputs = torch.tensor([t[:seq_len] for t in batch_data], device=device)
        targets = torch.tensor([t[seq_len] for t in batch_data], device=device)
        mask = (inputs != 0) & (inputs != eos_id)
        
        optimizer.zero_grad()
        logits, next_point, _ = model(inputs, mask)
        word_loss = F.cross_entropy(logits, targets)
        
        target_embedding = model.embedding(targets) * math.sqrt(model.embedding.embedding_dim)
        target_points = model.to_nd(model.norm(target_embedding.unsqueeze(1)))[:, 0]
        point_loss = F.mse_loss(next_point, target_points)
        
        loss = word_loss + 0.1 * point_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
    
    avg = epoch_loss / max(n_batches, 1)
    ppl = math.exp(min(avg, 20))
    elapsed = time.time() - t0
    
    status = "✅ New best" if ppl < best_ppl else ""
    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_ppl": best_ppl,
            "config": cfg
        }, f'checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20/wiki_cloud_epoch{epoch}.pt')
    
    print(f"Epoch {epoch:2d}/10 | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s {status}")

print(f"\n✅ Simple training complete")
print(f"Best PPL: {best_ppl:.2f}")
