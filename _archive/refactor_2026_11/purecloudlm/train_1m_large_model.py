"""Train PureCloudLM - LARGER MODEL on 1M CORPUS"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - LARGER MODEL (1M CORPUS)")
print("=" * 70)

device = "cpu"
print(f"Device: {device}")

# Load 800K sentences
print("\nLoading corpus...")
texts = []
files = [
    ('data/turkish_sentences.txt', 200000),
    ('data/turkish_wiki_1500k.txt', 600000),
]

for filepath, limit in files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            take = min(len(lines), limit - len(texts))
            texts.extend([line.strip() for line in lines[:take]])
            print(f"  {os.path.basename(filepath)}: {take} sentences")
            if len(texts) >= 800000:
                break

texts = texts[:800000]
print(f"\nTotal: {len(texts):,} sentences")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_200k/wiki_bpe.json')
print(f"Vocab: {tok.num_tokens}")

# Encode
print("\nEncoding...")
start_encode = time.time()
all_tokens = []
for i, t in enumerate(texts):
    enc = tok.encode(t)
    if 11 <= len(enc) <= 256:
        all_tokens.append(enc)
    if (i + 1) % 200000 == 0:
        print(f"  Encoded {i+1:,} / {len(texts):,} ({len(all_tokens):,} valid)")
print(f"Valid: {len(all_tokens):,} ({time.time()-start_encode:.1f}s)")

# LARGER MODEL config
cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 256,      # 128 → 256 (2x)
    "d_cloud": 512,      # 256 → 512 (2x)
    "d_cloud_dim": 16,   # 8 → 16 (2x)
    "n_layers": 1,
    "hidden": 256,       # 128 → 256 (2x)
    "proj_hidden": 256,  # 128 → 256 (2x)
    "point_hidden": 128, # 64 → 128 (2x)
    "use_pair_approx": False,
    "pool_mode": "all",
}
model = PureCloudLM(**cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"Size: {n_params/7838064:.1f}x larger than 7.8M model")

# Low LR for larger model
LR = 8e-6
print(f"\nTraining config:")
print(f"  LR: {LR}")
print(f"  Batch: 256")
print(f"  Epochs: 25")
print(f"  Clip: 0.3")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
eos_id = tok._tokenizer.token_to_id("<eos>") or 1

batch_size = 256
seq_len = 20
best_ppl = float('inf')
t0 = time.time()

# Shuffle
import random
random.seed(42)
random.shuffle(all_tokens)

checkpoint_dir = Path('purecloudlm/checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

print("\n" + "=" * 70)
print("  TRAINING STARTED (LARGER MODEL)")
print("=" * 70)

for epoch in range(1, 26):
    epoch_loss, n_batches = 0, 0
    
    for i in range(0, len(all_tokens), batch_size):
        batch_data = all_tokens[i:i+batch_size]
        
        batch_inputs = []
        batch_targets = []
        for t in batch_data:
            batch_inputs.append(t[:seq_len] + [0] * max(0, seq_len - len(t)))
            batch_targets.append(t[seq_len] if len(t) > seq_len else 0)
        
        inputs = torch.tensor(batch_inputs, dtype=torch.long)
        targets = torch.tensor(batch_targets, dtype=torch.long)
        mask = (inputs != 0) & (inputs != eos_id)
        
        optimizer.zero_grad()
        logits, next_point, _ = model(inputs, mask)
        word_loss = F.cross_entropy(logits, targets)
        
        target_embedding = model.embedding(targets) * math.sqrt(model.embedding.embedding_dim)
        target_points = model.to_nd(target_embedding.unsqueeze(1))[:, 0]
        point_loss = F.mse_loss(next_point, target_points)
        
        loss = word_loss + 0.1 * point_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)
        optimizer.step()
        
        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
    
    scheduler.step()
    
    if n_batches == 0:
        print(f"\n⚠️ All batches NaN at epoch {epoch}, STOPPING!")
        break
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    
    status = "✅ NEW BEST" if ppl < best_ppl else ""
    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_ppl": best_ppl,
            "config": cfg,
            "corpus_size": len(all_tokens),
            "total_time": time.time() - t0
        }, checkpoint_dir / 'pure_cloud_best_1m_large.pt')
    
    elapsed = time.time() - t0
    print(f"Epoch {epoch:2d}/25 | loss={avg:.4f} | ppl={ppl:7.2f} | "
          f"{elapsed/60:.1f}m {status}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best_1m_large.pt")
