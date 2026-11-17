"""Train PureCloudLM - 1M CORPUS + 200 EPOCHS (ULTRA STABLE)"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - 1M CORPUS + 200 EPOCHS (ULTRA STABLE)")
print("=" * 70)

device = "cpu"
print(f"Device: {device}")

# Load 1M sentences
print("\nLoading corpus...")
texts = []
files = [
    ('data/turkish_sentences.txt', 200000),
    ('data/turkish_wiki_1500k.txt', 800000),
]

for filepath, limit in files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            take = min(len(lines), limit - len(texts))
            texts.extend([line.strip() for line in lines[:take]])
            print(f"  {os.path.basename(filepath)}: {take} sentences")
            if len(texts) >= 1000000:
                break

texts = texts[:1000000]
print(f"\nTotal: {len(texts):,} sentences")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_200k/wiki_bpe.json')
print(f"Vocab: {tok.num_tokens}")

# OPTIMIZED: Fast encoding
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

# Model - 7.8M (successful config)
cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 128,
    "d_cloud": 256,
    "d_cloud_dim": 8,
    "n_layers": 1,
    "hidden": 128,
    "proj_hidden": 128,
    "point_hidden": 64,
    "use_pair_approx": False,
    "pool_mode": "all",
}
model = PureCloudLM(**cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} params ({n_params/1e6:.1f}M)")

# ULTRA STABLE settings for 200 epochs
LR = 2e-6  # Daha düşük
print(f"\nTraining config (ULTRA STABLE - 200 EPOCHS):")
print(f"  LR: {LR}")
print(f"  Batch: 256")
print(f"  Epochs: 200")
print(f"  Clip: 0.15")
print(f"  NaN Recovery: ON")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
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
print("  TRAINING STARTED (200 EPOCHS)")
print("=" * 70)

nan_epochs = 0
for epoch in range(1, 201):
    epoch_loss, n_batches = 0, 0
    epoch_nan_count = 0
    
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
        
        # NaN recovery: skip batch if loss is bad
        if not torch.isfinite(loss) or loss.item() > 50:
            epoch_nan_count += 1
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.15)
        optimizer.step()
        
        if torch.isfinite(word_loss) and word_loss.item() < 50:
            epoch_loss += word_loss.item()
            n_batches += 1
    
    scheduler.step()
    
    # NaN recovery: if too many NaN batches, reduce LR
    if n_batches < len(all_tokens) // batch_size * 0.5:
        print(f"  ⚠️ High NaN rate ({epoch_nan_count}), reducing LR")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
    
    if n_batches == 0:
        print(f"\n⚠️ All batches NaN at epoch {epoch}")
        nan_epochs += 1
        if nan_epochs >= 2:
            print("⚠️ 2 consecutive NaN epochs, STOPPING!")
            break
        continue
    
    nan_epochs = 0  # Reset on successful epoch
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    
    status = "✅ NEW BEST" if ppl < best_ppl else ""
    if ppl < best_ppl:
        best_ppl = ppl
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_ppl": best_ppl,
            "config": cfg,
            "corpus_size": len(all_tokens),
            "total_time": time.time() - t0
        }, checkpoint_dir / 'pure_cloud_best_200epochs.pt')
    
    elapsed = time.time() - t0
    
    # Progress update every 20 epochs
    if epoch % 20 == 0 or epoch <= 10:
        print(f"Epoch {epoch:3d}/200 | loss={avg:.4f} | ppl={ppl:7.2f} | "
              f"{elapsed/60:.1f}m {status} (NaN: {epoch_nan_count})")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best_200epochs.pt")
