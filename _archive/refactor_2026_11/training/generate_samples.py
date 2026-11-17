"""Generate sample predictions from CloudLM on held-out test set"""
import sys, os, torch, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cloudlm import BPETokenizer, CloudLM

print("=" * 60)
print("  CLOUDLM SAMPLE PREDICTIONS")
print("=" * 60)

# Load data
with open('data/turkish_wiki_1500k.txt', 'r', encoding='utf-8') as f:
    texts = [line.strip() for line in f]

# Load tokenizer
tok = BPETokenizer.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/wiki_bpe.json')

# Create and load model
cfg = {
    "d_model": 256, "d_cloud": 512, "d_cloud_dim": 64,
    "n_layers": 8, "pool": "max", "cloud_attn": True, "cloud_attn_heads": 8
}
model = CloudLM(tok.num_tokens, **cfg)
checkpoint = torch.load('checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20_v2/wiki_cloud_epoch298.pt', map_location='cpu')
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
    print()

print("=" * 60)
print("  SAMPLE GENERATION COMPLETE")
print("=" * 60)
