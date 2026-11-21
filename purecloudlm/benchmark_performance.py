"""Quick PureCloudLM Performance Benchmark - 100K samples"""
import sys, os, torch, torch.nn.functional as F, math, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import CloudLM, BPETokenizer

print("=" * 70)
print("  PURECLOUDLM PERFORMANCE BENCHMARK")
print("=" * 70)

# Load subset of data
with open('data/turkish_sentences.txt', 'r', encoding='utf-8') as f:
    all_texts = [line.strip() for line in f.readlines()]

# Take 100K samples
texts = all_texts[:100000]
print(f"Training on: {len(texts):,} sentences")

# Load tokenizer
bpe_path = 'checkpoints/wiki_checkpoints_200k/wiki_bpe.json'
tok = BPETokenizer.load(bpe_path)
print(f"BPE vocabulary: {tok.num_tokens:,} tokens")

# PureCloudLM config
pure_cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 256,
    "d_cloud": 512,
    "d_cloud_dim": 16,
    "n_layers": 1,
    "hidden": 256,
    "proj_hidden": 256,
    "point_hidden": 128,
    "use_pair_approx": False,
    "pool_mode": "all",
}

# CloudLM config (for comparison)
cloud_cfg = {
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

# Create models
pure_model = PureCloudLM(**pure_cfg)
pure_params = sum(p.numel() for p in pure_model.parameters())

cloud_model = CloudLM(**cloud_cfg)
cloud_params = sum(p.numel() for p in cloud_model.parameters())

print("\nModel sizes:")
print(f"  PureCloudLM:  {pure_params:,} params ({pure_params/1e6:.1f}M)")
print(f"  CloudLM:      {cloud_params:,} params ({cloud_params/1e6:.1f}M)")
print(f"  Ratio:        {cloud_params/pure_params:.1f}x")

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"\nDevice: {device}")

# Encode data
print("\nEncoding data...")
start = time.time()
all_tokens = [tok.encode(t) for t in texts]
all_tokens = [t for t in all_tokens if 11 <= len(t) <= 256]
print(f"Valid: {len(all_tokens):,} sentences ({time.time()-start:.1f}s)")

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 20
batch_size = 512

def train_model(model, name, epochs=5):
    """Train model and return metrics"""
    model.to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    best_ppl = float('inf')
    total_start = time.time()
    
    for epoch in range(1, epochs+1):
        epoch_loss, n_batches = 0, 0
        epoch_start = time.time()
        
        for i in range(0, len(all_tokens), batch_size):
            batch_data = all_tokens[i:i+batch_size]
            # Pad sequences to seq_len
            batch_inputs = []
            batch_targets = []
            for t in batch_data:
                batch_inputs.append(t[:seq_len] + [0] * max(0, seq_len - len(t)))
                batch_targets.append(t[seq_len] if len(t) > seq_len else 0)
            
            inputs = torch.tensor(batch_inputs, device=device)
            targets = torch.tensor(batch_targets, device=device)
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
        elapsed = time.time() - total_start
        epoch_time = time.time() - epoch_start
        
        if ppl < best_ppl:
            best_ppl = ppl
        
        print(f"  {name} Epoch {epoch}/{epochs} | loss={avg:.4f} | ppl={ppl:6.2f} | {epoch_time/60:.1f}m")
    
    return {
        "best_ppl": best_ppl,
        "total_time": time.time() - total_start,
        "params": sum(p.numel() for p in model.parameters())
    }

print("\n" + "=" * 70)
print("  TRAINING")
print("=" * 70)

print("\nTraining PureCloudLM...")
pure_metrics = train_model(pure_model, "PureCloudLM", epochs=5)

print("\nTraining CloudLM...")
cloud_metrics = train_model(cloud_model, "CloudLM", epochs=5)

# Results
print("\n" + "=" * 70)
print("  RESULTS")
print("=" * 70)
print(f"\n{'Metric':<25} {'PureCloudLM':<15} {'CloudLM':<15}")
print("-" * 55)
print(f"{'Best PPL':<25} {pure_metrics['best_ppl']:>10.2f}  {cloud_metrics['best_ppl']:>10.2f}")
print(f"{'Training Time':<25} {pure_metrics['total_time']/60:>10.1f}m  {cloud_metrics['total_time']/60:>10.1fm}")
print(f"{'Model Size':<25} {pure_metrics['params']/1e6:>10.1f}M  {cloud_metrics['params']/1e6:>10.1f}M")

print(f"\nSpeed ratio: {cloud_metrics['total_time']/pure_metrics['total_time']:.1f}x")
print(f"Size ratio: {cloud_metrics['params']/pure_metrics['params']:.1f}x")

print("\nConclusion:")
print("  PureCloudLM: O(N) complexity, no Transformer attention")
print("  CloudLM: O(N²) attention, but potentially better accuracy")
print("  Trade-off: Speed vs. Performance")
