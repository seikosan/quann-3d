"""Evaluate CloudLM on held-out test set"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudlm import BPETokenizer, CloudLM

print("=" * 60)
print("  CLOUDLM EVALUATION (Epoch 298)")
print("=" * 60)

# Load data
with open('data/turkish_wiki_1500k.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]
print(f"Total corpus: {len(texts)} cümle")

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/wiki_bpe.json')
print(f"BPE: {tok.num_tokens} tokens")

# Create model
cfg = {
    "d_model": 256, "d_cloud": 512, "d_cloud_dim": 64,
    "n_layers": 8, "pool": "max", "cloud_attn": True, "cloud_attn_heads": 8
}
model = CloudLM(tok.num_tokens, **cfg)

# Load checkpoint
checkpoint = torch.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/wiki_cloud_epoch298.pt', map_location='cpu')
model.load_state_dict(checkpoint["model_state"])
print(f"Loaded: epoch {checkpoint['epoch']}, best_ppl {checkpoint['best_ppl']:.2f}")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# Encode - use held-out portion (last 100K sentences)
all_tokens = [tok.encode(t) for t in texts]
valid_tokens = [t for t in all_tokens if len(t) >= 11]
print(f"Valid sentences: {len(valid_tokens)}")

# Split: train (first 1.2M), test (last 100K)
split_idx = int(len(valid_tokens) * 0.93)  # ~1.2M train, ~100K test
test_tokens = valid_tokens[split_idx:]
print(f"Test set: {len(test_tokens)} sentences")

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 10

# Evaluate on test set
total_loss = 0.0
n_batches = 0
batch_size = 256

print("\nEvaluating on test set...")
with torch.no_grad():
    for i in range(0, len(test_tokens), batch_size):
        batch_data = test_tokens[i:i+batch_size]
        inputs = torch.tensor([t[:seq_len] for t in batch_data], device=device)
        targets = torch.tensor([t[seq_len] for t in batch_data], device=device)
        mask = (inputs != 0) & (inputs != eos_id)
        
        logits, next_point, _ = model(inputs, mask)
        word_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        
        if torch.isfinite(word_loss):
            total_loss += word_loss.item()
            n_batches += 1
        
        if (i + 1) % 500 == 0:
            print(f"  Test: {min(i+len(batch_data), len(test_tokens))}/{len(test_tokens)} samples")

avg_loss = total_loss / max(n_batches, 1)
ppl = math.exp(min(avg_loss, 20))

print(f"\n{'=' * 60}")
print(f"  TEST SET RESULTS")
print(f"{'=' * 60}")
print(f"Test samples: {len(test_tokens)}")
print(f"Avg loss: {avg_loss:.4f}")
print(f"Perplexity: {ppl:.2f}")

# Also evaluate on full corpus
print(f"\nEvaluating on full corpus...")
total_loss_full = 0.0
n_batches_full = 0

with torch.no_grad():
    for i in range(0, len(valid_tokens), batch_size):
        batch_data = valid_tokens[i:i+batch_size]
        inputs = torch.tensor([t[:seq_len] for t in batch_data], device=device)
        targets = torch.tensor([t[seq_len] for t in batch_data], device=device)
        mask = (inputs != 0) & (inputs != eos_id)
        
        logits, next_point, _ = model(inputs, mask)
        word_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='mean')
        
        if torch.isfinite(word_loss):
            total_loss_full += word_loss.item()
            n_batches_full += 1

avg_loss_full = total_loss_full / max(n_batches_full, 1)
ppl_full = math.exp(min(avg_loss_full, 20))

print(f"\n{'=' * 60}")
print(f"  FULL CORPUS RESULTS")
print(f"{'=' * 60}")
print(f"Full corpus samples: {len(valid_tokens)}")
print(f"Avg loss: {avg_loss_full:.4f}")
print(f"Perplexity: {ppl_full:.2f}")

# Save results
import json
results = {
    "test": {
        "samples": len(test_tokens),
        "loss": avg_loss,
        "ppl": ppl
    },
    "full": {
        "samples": len(valid_tokens),
        "loss": avg_loss_full,
        "ppl": ppl_full
    },
    "checkpoint": "wiki_cloud_epoch298.pt",
    "config": cfg
}

output_path = 'checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/evaluation_results.json'
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to {output_path}")
