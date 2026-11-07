#!/usr/bin/env python3
"""Generate text from trained ContextCloudLM checkpoint."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from contextcloud import ContextCloudLM

# Load checkpoint
checkpoint = torch.load("contextcloud/checkpoints/context_best.pt", map_location="cpu")
vocab_size = checkpoint["vocab_size"]

# Create model with same config
model = ContextCloudLM(
    vocab_size=vocab_size,
    d_model=128,
    d_sentence=256,
    d_window=256,
    d_cloud_dim=16,
    d_window_dim=16,
    window_size=5,
    window_stride=2,
    hidden=128,
    max_tokens=128,
)

# Load weights
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("Loaded checkpoint from epoch", checkpoint.get("epoch", "N/A"))
print("Validation loss:", checkpoint.get("val_loss", "N/A"))

# Simple generation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)

# Create simple vocab for demo
vocab = {
    "<pad>": 0, "<unk>": 1,
    "merhaba": 2, "dunya": 3, "nasilsin": 4, "iyi": 5,
    "teşekkür": 6, "ederim": 7, "sen": 8, "nasıl": 9,
}
reverse_vocab = {v: k for k, v in vocab.items()}

# Generate from seed tokens
seed_text = "merhaba dunya"
seed_tokens = torch.tensor([[vocab.get(w, 1) for w in seed_text.split()]])
seed_tokens = seed_tokens.to(device)

print(f"\nSeed: '{seed_text}'")
print("Generating...")

# Generate
with torch.no_grad():
    generated = model.generate(seed_tokens, max_new=10, temperature=0.8)

# Decode
output_ids = generated[0].cpu().tolist()
output_words = [reverse_vocab.get(id, f"<unk:{id}>") for id in output_ids]
print(f"Generated: {' '.join(output_words)}")
