"""Train PureCloudLM on 1M sentences - FLOAT32 STABLE VERSION"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - 1M SENTENCES FLOAT32 STABLE TRAINING")
print("=" * 70)

# Force float32
torch.set_default_dtype(torch.float32)

# Load 1M sentences
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
bpe_path = 'checkpoints/wiki_checkpoints_200k/wiki_bpe.json'
tok = BPETokenizer.load(bpe_path)
print(f"BPE: {tok.num_tokens:,} tokens")

# Model configuration
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

# Try MPS first, fallback to CPU
if torch.backends.mps.is_available():
    device = "mps"
    print(f"Device: {device} (will fallback to CPU if NaN)")
else:
    device = "cpu"
    print(f"Device: {device}")

model.to(device)
model.train()

# Very stable settings
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

# Encode corpus
print("\nEncoding corpus...")
start_encode = time.time()
all_tokens = []
for i, t in enumerate(texts):
    encoded = tok.encode(t)
    if 11 <= len(encoded) <= 256:
        all_tokens.append(encoded)
    if (i + 1) % 200000 == 0:
        print(f"  Encoded {i+1:,} / {len(texts):,} ({len(all_tokens):,} valid)")

encode_time = time.time() - start_encode
print(f"Valid sentences: {len(all_tokens):,} ({encode_time/60:.1f}m)")

# Shuffle
import random
random.seed(42)
random.shuffle(all_tokens)

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 20
batch_size = 256  # Even smaller batch
n_epochs = 10

checkpoint_dir = Path('purecloudlm/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_ppl = float('inf')
t0 = time.time()

print("\n" + "=" * 70)
print("  TRAINING STARTED (FLOAT32 + STABLE)")
print("=" * 70)

current_device = device
nan_epochs = 0

for epoch in range(1, n_epochs + 1):
    epoch_loss, n_batches = 0, 0
    epoch_start = time.time()
    epoch_nan_count = 0
    
    for i in range(0, len(all_tokens), batch_size):
        batch_data = all_tokens[i:i+batch_size]
        
        # Pad sequences - EXPLICIT FLOAT32
        batch_inputs = []
        batch_targets = []
        for t in batch_data:
            batch_inputs.append(t[:seq_len] + [0] * max(0, seq_len - len(t)))
            batch_targets.append(t[seq_len] if len(t) > seq_len else 0)
        
        # EXPLICIT: Ensure float32 and long
        inputs = torch.tensor(batch_inputs, dtype=torch.long)
        targets = torch.tensor(batch_targets, dtype=torch.long)
        mask = (inputs != 0) & (inputs != eos_id)
        
        # Move to device
        inputs = inputs.to(current_device)
        targets = targets.to(current_device)
        mask = mask.to(current_device)
        
        optimizer.zero_grad()
        logits, next_point, _ = model(inputs, mask)
        
        # EXPLICIT: Ensure float32 for loss
        word_loss = F.cross_entropy(logits.float(), targets)
        
        # Auxiliary point loss - EXPLICIT float32
        target_embedding = model.embedding(targets) * math.sqrt(model.embedding.embedding_dim)
        target_points = model.to_nd(target_embedding.unsqueeze(1))[:, 0]
        point_loss = F.mse_loss(next_point.float(), target_points.float())
        
        loss = word_loss + 0.1 * point_loss
        
        # Check for NaN
        if not torch.isfinite(loss):
            print(f"  ⚠️ NaN loss at batch {i//batch_size}, skipping")
            epoch_nan_count += 1
            continue
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.3)  # Stronger clipping
        optimizer.step()
        
        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
        
        # Progress update
        if (i // batch_size) % 15 == 0:
            progress = (i / len(all_tokens)) * 100
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:2d}/{n_epochs} | {progress:5.1f}% | "
                  f"loss={word_loss.item():.4f} | {elapsed/60:.1f}m")
    
    scheduler.step()
    
    # Check if all batches were NaN
    if n_batches == 0:
        print(f"  ⚠️ All batches NaN in epoch {epoch}")
        nan_epochs += 1
        
        # If 2 consecutive NaN epochs, switch to CPU
        if nan_epochs >= 2 and current_device == "mps":
            print("  🔄 Switching to CPU (MPS unstable)")
            current_device = "cpu"
            model.to(current_device)
            nan_epochs = 0
            continue
        elif nan_epochs >= 3:
            print("  ⚠️ Too many NaN epochs, stopping!")
            break
        else:
            continue
    
    nan_epochs = 0  # Reset on successful epoch
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - t0
    
    # Save checkpoint
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
            "total_time": elapsed,
            "device": current_device
        }, checkpoint_dir / 'pure_cloud_best.pt')
    
    print(f"\nEpoch {epoch:2d}/{n_epochs} | loss={avg:.4f} | ppl={ppl:7.2f} | "
          f"{epoch_time/60:.1f}m | {elapsed/60:.1f}m total {status} (NaN: {epoch_nan_count})\n")
    
    # Early stopping
    if ppl < 100 and epoch >= 8:
        print("✅ PPL < 100, training converged!")
        break

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"Final device: {current_device}")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best.pt")
