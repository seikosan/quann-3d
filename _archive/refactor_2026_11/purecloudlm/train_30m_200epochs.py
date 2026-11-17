"""Train PureCloudLM - VERY LARGE MODEL (30M) + 200 EPOCHS (STABLE)"""
import sys, os, copy, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - VERY LARGE MODEL (30M) + 200 EPOCHS")
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

# VERY LARGE MODEL config (3x bigger than 7.8M)
cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 384,      # 128 → 384 (3x)
    "d_cloud": 768,      # 256 → 768 (3x)
    "d_cloud_dim": 24,   # 8 → 24 (3x)
    "n_layers": 2,       # 1 → 2 (2 layer)
    "hidden": 384,       # 128 → 384 (3x)
    "proj_hidden": 384,  # 128 → 384 (3x)
    "point_hidden": 192, # 64 → 192 (3x)
    "use_pair_approx": False,
    "pool_mode": "all",
}

model = PureCloudLM(**cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} params ({n_params/1e6:.1f}M)")

# Encode with progress
print("\nEncoding...")
start_encode = time.time()
all_tokens = []
for i, t in enumerate(texts):
    enc = tok.encode(t)
    if 11 <= len(enc) <= 256:
        all_tokens.append(enc)
    if (i + 1) % 500000 == 0:
        print(f"  Encoded {i+1:,} / {len(texts):,} ({len(all_tokens):,} valid)")
print(f"Valid: {len(all_tokens):,} ({time.time()-start_encode:.1f}s)")

# VERY STABLE settings for large model
LR = 2e-6  # Ultra düşük
BATCH = 64  # Ultra küçük
GRAD_ACCUM = 4  # Gradient accumulation
WEIGHT_DECAY = 0.02  # Daha yüksek
MAX_GRAD_NORM = 0.1  # Çok sıkı

print(f"\nTraining config (STABLE):")
print(f"  LR: {LR}")
print(f"  Batch: {BATCH}")
print(f"  Grad Accum: {GRAD_ACCUM}")
print(f"  Weight Decay: {WEIGHT_DECAY}")
print(f"  Max Grad Norm: {MAX_GRAD_NORM}")
print(f"  Epochs: 200")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, eps=1e-7)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
eos_id = tok._tokenizer.token_to_id("<eos>") or 1

seq_len = 20
best_ppl = float('inf')
t0 = time.time()

# Shuffle
import random
random.seed(42)
random.shuffle(all_tokens)

checkpoint_dir = Path('purecloudlm/checkpoints')
checkpoint_dir.mkdir(exist_ok=True)

# EMA for stability
ema_model = None
ema_decay = 0.99

print("\n" + "=" * 70)
print("  TRAINING STARTED (30M MODEL + 200 EPOCHS)")
print("=" * 70)

nan_epochs = 0
for epoch in range(1, 201):
    epoch_loss, n_batches = 0, 0
    grad_accumulator = []
    
    # Warmup for first 10 epochs
    if epoch <= 10:
        warmup_lr = LR * (epoch / 10)
        for param_group in optimizer.param_groups:
            param_group['lr'] = warmup_lr
    
    for i in range(0, len(all_tokens), BATCH):
        batch_data = all_tokens[i:i+BATCH]
        
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
        
        loss = (word_loss + 0.1 * point_loss) / GRAD_ACCUM
        
        if not torch.isfinite(loss) or loss.item() > 50:
            continue
        
        loss.backward()
        grad_accumulator.append(word_loss.item())
        
        # Gradient accumulation
        if (i // BATCH + 1) % GRAD_ACCUM == 0 or i + BATCH >= len(all_tokens):
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            grad_accumulator = []
        
        if torch.isfinite(word_loss) and word_loss.item() < 50:
            epoch_loss += word_loss.item()
            n_batches += 1
    
    # Update EMA model
    if ema_model is None:
        ema_model = copy.deepcopy(model)
    else:
        for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
            ema_p.data = ema_decay * ema_p.data + (1 - ema_decay) * model_p.data
    
    scheduler.step()
    
    if n_batches == 0:
        print(f"\n⚠️ All batches NaN at epoch {epoch}")
        nan_epochs += 1
        if nan_epochs >= 2:
            print("⚠️ 2 consecutive NaN epochs, STOPPING!")
            break
        # Try to continue with lower LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
        continue
    
    nan_epochs = 0  # Reset on successful epoch
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    
    status = "✅ NEW BEST" if ppl < best_ppl else ""
    if ppl < best_ppl:
        best_ppl = ppl
        # Save EMA model if available
        save_model = ema_model if ema_model else model
        torch.save({
            "epoch": epoch,
            "model_state": save_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_ppl": best_ppl,
            "config": cfg,
            "corpus_size": len(all_tokens),
            "total_time": time.time() - t0
        }, checkpoint_dir / 'pure_cloud_best_30m_200epochs.pt')
    
    elapsed = time.time() - t0
    
    # Progress update every 20 epochs
    if epoch % 20 == 0 or epoch <= 10:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}/200 | loss={avg:.4f} | ppl={ppl:7.2f} | "
              f"lr={lr:.0e} | {elapsed/60:.1f}m {status}")

print("\n" + "=" * 70)
print("  TRAINING COMPLETE")
print("=" * 70)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
print(f"Corpus: {len(all_tokens):,} sentences")
print(f"Model: {n_params/1e6:.1f}M params")
print(f"\nCheckpoint: purecloudlm/checkpoints/pure_cloud_best_30m_200epochs.pt")
