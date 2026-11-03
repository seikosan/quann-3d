"""Train PureCloudLM on FULL corpus (7M+ sentences) - Performance Benchmark"""
import sys, os, torch, torch.nn.functional as F, math, time
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 70)
print("  PURECLOUDLM - FULL CORPUS TRAINING BENCHMARK")
print("=" * 70)

# Load data
corpus_files = [
    ('data/turkish_sentences.txt', '200K'),
    ('data/turkish_wiki_1500k.txt', '1.5M'),
    ('data/turkish_wiki_sentences.txt', '5.5M'),
]

texts = []
for filepath, label in corpus_files:
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            count = len(lines)
            texts.extend([line.strip() for line in lines])
            print(f"✓ {label}: {count} cümle (Toplam: {len(texts)})")

print(f"\nTotal corpus: {len(texts):,} cümle")

# Load tokenizer
bpe_path = 'checkpoints/wiki_checkpoints_200k/wiki_bpe.json'
if not os.path.exists(bpe_path):
    bpe_path = 'checkpoints/wiki_checkpoints/wiki_bpe.json'
    
tok = BPETokenizer.load(bpe_path)
print(f"BPE: {tok.num_tokens:,} tokens")

# Model configuration
cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 256,
    "d_cloud": 512,
    "d_cloud_dim": 16,
    "n_layers": 1,
    "hidden": 256,
    "proj_hidden": 256,
    "point_hidden": 128,
    "use_pair_approx": False,
    "num_sampled_pairs": 64,
    "pool_mode": "all",
}

model = PureCloudLM(**cfg)
n_params = sum(p.numel() for p in model.parameters())
print(f"\nModel: {n_params:,} params (Pure O(N) - NO Transformer)")

# Compare with CloudLM size
from cloudlm import CloudLM
cloudlm_cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 256,
    "d_cloud": 512,
    "d_cloud_dim": 16,
    "n_layers": 8,
    "nhead": 16,
    "hidden": 256,
    "proj_hidden": 256,
    "point_hidden": 128,
}
cloudlm = CloudLM(**cloudlm_cfg)
cloudlm_params = sum(p.numel() for p in cloudlm.parameters())
print(f"CloudLM (baseline): {cloudlm_params:,} params")
print(f"Size ratio: {cloudlm_params / n_params:.1f}x smaller")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {device}")
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)

# Encode and filter
print("\nEncoding corpus...")
start_time = time.time()
all_tokens = [tok.encode(t) for t in texts]
all_tokens = [t for t in all_tokens if len(t) >= 11 and len(t) <= 256]
encode_time = time.time() - start_time
print(f"Valid sentences: {len(all_tokens):,} ({encode_time:.0f}s)")

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 20
batch_size = 512

# Train
best_ppl = float('inf')
t0 = time.time()
epoch = 1

try:
    while epoch <= 10:
        epoch_loss, n_batches = 0, 0
        epoch_start = time.time()
        
        for i in range(0, len(all_tokens), batch_size):
            batch_data = all_tokens[i:i+batch_size]
            inputs = torch.tensor([t[:seq_len] for t in batch_data], device=device)
            targets = torch.tensor([t[seq_len] for t in batch_data], device=device)
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
        
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        epoch_time = time.time() - epoch_start
        
        status = "✅ New best" if ppl < best_ppl else ""
        if ppl < best_ppl:
            best_ppl = ppl
            checkpoint_path = Path('purecloudlm/checkpoints')
            checkpoint_path.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_ppl": best_ppl,
                "config": cfg,
                "corpus_size": len(all_tokens)
            }, checkpoint_path / f'pure_cloud_best.pt')
        
        print(f"Epoch {epoch:2d}/10 | loss={avg:.4f} | ppl={ppl:6.2f} | "
              f"{epoch_time/60:.1f}m | {elapsed/60:.1f}m total {status}")
        
        epoch += 1

except KeyboardInterrupt:
    print("\n\nTraining interrupted")

print(f"\n{'=' * 70}")
print(f"  PURECLOUDLM TRAINING COMPLETE")
print(f"{'=' * 70}")
print(f"Best PPL: {best_ppl:.2f}")
print(f"Total time: {elapsed/60:.1f} minutes")
print(f"Model size: {n_params:,} params ({n_params/1e6:.1f}M)")
print(f"Corpus size: {len(all_tokens):,} sentences")
print(f"\nNote: PureCloudLM is O(N) - no O(N²) attention")
print("Compare with CloudLM (Transformer) for performance trade-off")
