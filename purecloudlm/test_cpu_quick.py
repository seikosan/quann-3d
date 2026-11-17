"""Quick PureCloudLM CPU Test - 50K sentences"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 60)
print("  PURECLOUDLM - QUICK CPU TEST")
print("=" * 60)

device = "cpu"
torch.set_default_dtype(torch.float32)
print(f"Device: {device}")

# Load 50K sentences
with open('data/turkish_sentences.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f.readlines()][:50000]
print(f"Corpus: {len(texts)} sentences")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_200k/wiki_bpe.json')
print(f"Vocab: {tok.num_tokens}")

# Encode
print("Encoding...")
all_tokens = [tok.encode(t) for t in texts]
all_tokens = [t for t in all_tokens if 11 <= len(t) <= 256]
print(f"Valid: {len(all_tokens)} ({time.time():.1f}s)")

# Model
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
print(f"Model: {n_params:,} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
eos_id = tok._tokenizer.token_to_id("<eos>") or 1

# Train
batch_size = 256
seq_len = 20
best_ppl = float('inf')
t0 = time.time()

print("\n" + "=" * 60)
print("  TRAINING")
print("=" * 60)

for epoch in range(1, 6):
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        if torch.isfinite(word_loss):
            epoch_loss += word_loss.item()
            n_batches += 1
    
    scheduler.step()
    
    if n_batches == 0:
        print(f"Epoch {epoch} - ALL NaN, STOPPING!")
        break
    
    avg = epoch_loss / n_batches
    ppl = math.exp(min(avg, 20))
    
    status = "✅ NEW BEST" if ppl < best_ppl else ""
    if ppl < best_ppl:
        best_ppl = ppl
        Path('purecloudlm/checkpoints').mkdir(exist_ok=True)
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "best_ppl": best_ppl,
            "config": cfg
        }, 'purecloudlm/checkpoints/pure_cloud_cpu_test.pt')
    
    print(f"Epoch {epoch:2d}/5 | loss={avg:.4f} | ppl={ppl:6.2f} | {(time.time()-t0)/60:.1f}m {status}")

print("\n" + "=" * 60)
print("  COMPLETE")
print("=" * 60)
print(f"Best PPL: {best_ppl:.2f}")
print(f"Time: {(time.time()-t0)/60:.1f} minutes")
print(f"CPU: {'✅ STABLE' if best_ppl < 500 else '⚠️ UNSTABLE'}")
