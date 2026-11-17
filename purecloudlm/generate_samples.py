"""Generate sample predictions from PureCloudLM on held-out test set"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from purecloudlm import PureCloudLM
from cloudlm import BPETokenizer

print("=" * 60)
print("  PURECLOUDLM SAMPLE PREDICTIONS")
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
split_idx = int(len(valid_tokens) * 0.93)
test_tokens = valid_tokens[split_idx:split_idx + 10]  # First 10 test samples

eos_id = tok._tokenizer.token_to_id("<eos>") or 1
seq_len = 10

print(f"\nGenerating predictions for {len(test_tokens)} held-out samples:\n")

correct_top1 = 0
correct_top3 = 0
total = len(test_tokens)

for idx, token_ids in enumerate(test_tokens):
    original_text = texts[split_idx + idx]
    
    # Input: first 10 tokens
    input_ids = torch.tensor([token_ids[:seq_len]], device=device)
    mask = (input_ids != 0) & (input_ids != eos_id)
    
    with torch.no_grad():
        logits, next_point, _ = model(input_ids, mask)
    
    # Get top-5 predictions for next token
    probs = torch.softmax(logits[0], dim=-1)
    top_values, top_indices = torch.topk(probs, 5)
    
    print(f"Sample {idx + 1}:")
    print(f"  Original: {original_text}")
    
    # Decode input
    input_text = tok.decode(token_ids[:seq_len])
    print(f"  Input (first 10 tokens): {input_text}")
    
    # Decode predictions
    print(f"  Next token predictions:")
    for i, (val, idx) in enumerate(zip(top_values, top_indices)):
        predicted_text = tok.decode([idx.item()])
        print(f"    {i+1}. {predicted_text} (prob: {val:.4f})")
    
    # Actual next token
    actual_next = token_ids[seq_len]
    actual_text = tok.decode([actual_next])
    print(f"  Actual next token: {actual_text}")
    
    # Check accuracy
    if top_indices[0].item() == actual_next:
        correct_top1 += 1
        print(f"  ✅ Top-1 CORRECT")
    elif actual_next in top_indices[:3].tolist():
        print(f"  ✅ Top-3 CORRECT")
        correct_top3 += 1
    else:
        print(f"  ❌ Incorrect")
    
    print()

print("=" * 60)
print(f"  RESULTS: Top-1 Accuracy: {correct_top1}/{total} ({100*correct_top1/total:.1f}%)")
print(f"           Top-3 Accuracy: {correct_top3}/{total} ({100*correct_top3/total:.1f}%)")
print("=" * 60)
