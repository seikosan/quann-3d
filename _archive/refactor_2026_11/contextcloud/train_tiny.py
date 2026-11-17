#!/usr/bin/env python3
"""Train tiny ContextCloudLM for quick debugging."""

import logging
import os
import torch
from torch.utils.data import DataLoader, TensorDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

from contextcloud import ContextCloudLM

# Tiny model
model = ContextCloudLM(
    vocab_size=1000,
    d_model=32,
    d_sentence=64,
    d_window=64,
    d_cloud_dim=4,
    d_window_dim=4,
    window_size=3,
    window_stride=1,
    hidden=32,
    max_tokens=16,
)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

# Create synthetic data
num_samples = 1000
tokens = torch.randint(1, 100, (num_samples, 1, 16))
targets = torch.randint(1, 1000, (num_samples,))
mask = torch.ones(num_samples, 1, 16, dtype=torch.bool)

dataset = TensorDataset(tokens, targets, mask)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Training
for epoch in range(20):
    model.train()
    total_loss = 0.0
    
    for batch_tokens, batch_targets, batch_mask in dataloader:
        batch_tokens = batch_tokens.to(device)
        batch_targets = batch_targets.to(device)
        batch_mask = batch_mask.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_tokens, sentence_mask=batch_mask)
        loss = criterion(logits, batch_targets)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    logger.info("Epoch %d | Loss: %.4f", epoch + 1, avg_loss)

logger.info("Training complete!")

# Test
model.eval()
test_tokens = torch.randint(1, 100, (1, 1, 16)).to(device)
with torch.no_grad():
    output = model(test_tokens)
    logger.info("Output shape: %s", output.shape)
    logger.info("Sample output: %s", output[0, :10])
