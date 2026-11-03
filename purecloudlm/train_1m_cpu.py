"""Train PureCloudLM on 1M sentences - CPU VERSION (Most Stable)"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - 1M SENTENCES CPU TRAINING")
print("=" * 70)

# Force CPU and float32
device = "cpu"
torch.set_default_dtype(torch.float32)
print(f"Device: {device} (most stable)")

# Load data
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

# Model - smaller for CPU speed
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

# Stable settings for CPU
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Encode
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
batch_size = 512
n_epochs = 10

checkpoint_dir = Path('purecloudlm/checkpoints')
checkpoint_dir.mkdir(parents=True, exist_ok=True)

best_ppl = float('inf')
t0 = time.time()

print("\n" + "=" * 70)
print("  TRAINING STARTED (CPU)")
print("=" * 70)

for epoch in range(1, n_epochs + 1):
    epoch_loss, n_batches = 0, 0
    epoch_start = time.time()
    
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
        
        if (i // batch_size) % 20 == 0:
            progress = (i / len(all_tokens)) * 100
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:2d}/{n_epochs} | {progress:5.1f}% | "
                  f"loss={word_loss.item():.4f} | {elapsed/60:.1f}m")
    
    scheduler.step()
    
    if n_batches == 0:
        print(f"  ⚠️ All batches NaN, stopping!")
        break
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    epoch_time = time.time() - epoch_start
    elapsed = time.time() - t0
    
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
            "total_time": elapsed
        }, checkpoint_dir / 'pure_cloud_best_cpu.pt')
    
    print(f"\nEpoch {epoch:2d}/{n_epochs} | loss={avg:.4f} | ppl={ppl:7.2f} | "
          f"{epoch_time/60:.1f}m | {elapsed/60:.1f}m total {status}\n")
    
    if ppl < 100 and epoch >= 8:
        print("✅ PPL < 100, training converged!")
        break

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best_cpu.pt")
