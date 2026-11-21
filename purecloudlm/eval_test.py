"""Evaluate PureCloudLM on held-out test set"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 60)
print("  PURECLOUDLM TEST EVALUATION")
print("=" * 60)

# Load data
with open('data/turkish_wiki_1500k.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20/wiki_bpe.json')

# Create model (same config as training)
cfg = {
    "vocab_size": tok.num_tokens,
    "d_model": 256,
    "d_cloud": 512,
    "d_cloud_dim": 16,
    "n_layers": 1,
    "pos_len": 256,
    "hidden": 256,
    "proj_hidden": 256,
    "point_hidden": 128,
    "use_pair_approx": False,
    "num_sampled_pairs": 64
}
model = PureCloudLM(**cfg)

# Load best checkpoint
checkpoint_path = 'purecloudlm/checkpoints/pure_cloud_epoch50.pt'
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint["model_state"])

device = "mps" if torch.backends.mps.is_available() else "cpu"
model.to(device)
model.eval()

# Encode - use held-out portion (last 100K sentences)
all_tokens = [tok.encode(t) for t in texts]
valid_tokens = [t for t in all_tokens if len(t) >= 11]

# Use last 100K as test set
test_size = 100000
test_tokens = valid_tokens[-test_size:]

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 10
batch_size = 256

print(f"\nTest set: {len(test_tokens)} sentences")
print("Computing test perplexity...\n")

total_loss = 0
total_batches = 0

with torch.no_grad():
    for i in range(0, len(test_tokens), batch_size):
        batch_data = test_tokens[i:i+batch_size]
        inputs = torch.tensor([t[:seq_len] for t in batch_data], device=device)
        targets = torch.tensor([t[seq_len] for t in batch_data], device=device)
        mask = (inputs != 0) & (inputs != eos_id)
        
        logits, _, _ = model(inputs, mask)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        
        if torch.isfinite(loss):
            total_loss += loss.item()
            total_batches += 1

avg_loss = total_loss / max(total_batches, 1)
test_ppl = math.exp(min(avg_loss, 20))

print("=" * 60)
print("  TEST RESULTS")
print("=" * 60)
print(f"  Average Loss: {avg_loss:.4f}")
print(f"  Test PPL: {test_ppl:.2f}")
print("=" * 60)

# Compare with CloudLM
print("\n  COMPARISON:")
print("  CloudLM Test PPL:     21.91")
print(f"  PureCloudLM Test PPL: {test_ppl:.2f}")
print(f"  Difference:           {test_ppl - 21.91:+.2f}")
print("=" * 60)
