"""Train PureCloudLM - 7.8M MODEL with PROPER DATA SPLIT (80/10/10)"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
import random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - 7.8M MODEL + PROPER DATA SPLIT (80/10/10)")
print("=" * 70)

device = "cpu"
print(f"Device: {device}")

# Load 2M sentences
print("\nLoading corpus...")
with open('data/turkish_wiki_sentences.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()][:2000000]
print(f"Total: {len(texts):,} sentences")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_200k/wiki_bpe.json')
print(f"Vocab: {tok.num_tokens}")

# Encode all texts
print("\nEncoding...")
start_encode = time.time()
all_tokens = []
for i, t in enumerate(texts):
    enc = tok.encode(t)
    if 11 <= len(enc) <= 256:
        all_tokens.append(enc)
    if (i + 1) % 500000 == 0:
        print(f"  Encoded {i+1:,} / {len(texts):,} ({len(all_tokens):,} valid)")
encode_time = time.time() - start_encode
print(f"Valid: {len(all_tokens):,} ({encode_time/60:.1f}m)")

# PROPER DATA SPLIT (80/10/10)
random.seed(42)
random.shuffle(all_tokens)

n = len(all_tokens)
train_size = int(n * 0.8)
val_size = int(n * 0.1)

train_data = all_tokens[:train_size]
val_data = all_tokens[train_size:train_size+val_size]
test_data = all_tokens[train_size+val_size:]

print(f"\n{'=' * 70}")
print("  DATA SPLIT (80/10/10)")
print(f"{'=' * 70}")
print(f"Training:  {len(train_data):,} sentences ({len(train_data)/len(all_tokens)*100:.1f}%)")
print(f"Validation:{len(val_data):,} sentences ({len(val_data)/len(all_tokens)*100:.1f}%)")
print(f"Test:      {len(test_data):,} sentences ({len(test_data)/len(all_tokens)*100:.1f}%)")

# 7.8M MODEL config (successful config)
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

# Stable settings
LR = 3e-6
print(f"\nTraining config:")
print(f"  LR: {LR}")
print(f"  Batch: 256")
print(f"  Epochs: 100")
print(f"  Clip: 0.3")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
eos_id = tok._tokenizer.token_to_id("<eos>") or 1

batch_size = 256
seq_len = 20
best_ppl = float('inf')
best_val_ppl = float('inf')
t0 = time.time()

checkpoint_dir = Path('purecloudlm/checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

print("\n" + "=" * 70)
print("  TRAINING STARTED (with Validation)")
print("=" * 70)

def evaluate_ppl(data, model, batch_size, seq_len, eos_id):
    """Calculate perplexity on a dataset"""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    with torch.no_grad():
        for i in range(0, len(data), batch_size):
            batch = data[i:i+batch_size]
            inputs = torch.tensor([t[:seq_len] + [0]*max(0,seq_len-len(t)) for t in batch], dtype=torch.long)
            targets = torch.tensor([t[seq_len] if len(t)>seq_len else 0 for t in batch], dtype=torch.long)
            mask = (inputs != 0) & (inputs != eos_id)
            
            logits, next_point, _ = model(inputs, mask)
            word_loss = F.cross_entropy(logits, targets)
            
            if torch.isfinite(word_loss):
                total_loss += word_loss.item()
                n_batches += 1
    
    model.train()
    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(min(avg_loss, 20))
    return ppl

for epoch in range(1, 101):
    epoch_loss, n_batches = 0, 0
    
    # Shuffle training data each epoch
    random.shuffle(train_data)
    
    for i in range(0, len(train_data), batch_size):
        batch = train_data[i:i+batch_size]
        inputs = torch.tensor([t[:seq_len] + [0]*max(0,seq_len-len(t)) for t in batch], dtype=torch.long)
        targets = torch.tensor([t[seq_len] if len(t)>seq_len else 0 for t in batch], dtype=torch.long)
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
    
    # Evaluate on validation set
    val_ppl = evaluate_ppl(val_data, model, batch_size, seq_len, eos_id)
    
    avg = epoch_loss / n_batches
    train_ppl = math.exp(min(avg, 20))
    
    elapsed = time.time() - t0
    
    # Save best model based on validation PPL
    status = ""
    if val_ppl < best_val_ppl:
        best_val_ppl = val_ppl
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_ppl": best_val_ppl,
            "config": cfg,
            "corpus_size": len(all_tokens),
            "train_size": len(train_data),
            "val_size": len(val_data),
            "test_size": len(test_data),
            "total_time": elapsed
        }, checkpoint_dir / 'pure_cloud_best_80_10_10.pt')
        status = "✅ NEW BEST (val)"
    
    print(f"Epoch {epoch:3d}/100 | train_ppl={train_ppl:7.2f} | val_ppl={val_ppl:7.2f} | "
          f"{elapsed/60:.1f}m {status}")

# Final evaluation on test set
print("\n" + "=" * 70)
print("  FINAL EVALUATION ON TEST SET")
print("=" * 70)

test_ppl = evaluate_ppl(test_data, model, batch_size, seq_len, eos_id)
print(f"Test PPL: {test_ppl:.2f}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best Validation PPL: {best_val_ppl:.2f}")
print(f"Test PPL: {test_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best_80_10_10.pt")
print("\nNote: Test set was NEVER used during training - only for final evaluation")
