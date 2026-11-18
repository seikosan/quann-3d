#!/usr/bin/env python3
"""
Train ContextCloudLM with Memory Network for long texts.

Key Idea:
- Explicit memory bank that stores information across chunks
- Read operation: retrieve relevant info from memory
- Write operation: update memory with new chunk info
- Allows long-range dependency learning through memory

Architecture:
  Chunk 1 → Read Memory → Encode → Write Memory → Chunk 2 → ...
"""

import logging
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from contextcloud.dataset import ChunkedVocabDataset, collate_chunks
from contextcloud.task_utils import shift_target_3d, split_sentences

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class MemoryBank(nn.Module):
    """Explicit memory bank for long-range dependency."""
    
    def __init__(self, num_memory=64, dim=256):
        super().__init__()
        # Learnable memory slots
        self.memory = nn.Parameter(torch.randn(num_memory, dim) * 0.02)
        self.num_memory = num_memory
        self.dim = dim
        
        # Read: query → memory attention
        self.read_q = nn.Linear(dim, dim)
        self.read_k = nn.Linear(dim, dim)
        self.read_v = nn.Linear(dim, dim)
        
        # Write: chunk state → memory update
        self.write_gate = nn.Linear(dim, num_memory)
        self.write_update = nn.Linear(dim, dim)
    
    def read(self, query):
        """Read from memory using query vector."""
        # Compute attention over memory slots
        q = self.read_q(query)  # (B, Dim)
        k = self.read_k(self.memory)  # (NumMemory, Dim)
        v = self.read_v(self.memory)  # (NumMemory, Dim)
        
        # Attention scores
        scores = q @ k.T  # (B, NumMemory)
        attn = torch.softmax(scores, dim=-1)
        
        # Read weighted memory
        memory_read = attn @ v  # (B, Dim)
        return memory_read
    
    def write(self, chunk_state, memory):
        """Write chunk state to memory."""
        # Compute write gate (which memory slots to update)
        gate = torch.sigmoid(self.write_gate(chunk_state))  # (B, NumMemory)
        
        # Compute new memory values
        new_values = self.write_update(chunk_state)  # (B, Dim)
        
        # Update memory: memory = (1 - gate) * memory + gate * new_values
        # Expand for batch
        gate_exp = gate.unsqueeze(-1)  # (B, NumMemory, 1)
        new_values_exp = new_values.unsqueeze(1)  # (B, 1, Dim)
        
        updated_memory = (1 - gate_exp) * memory + gate_exp * new_values_exp
        
        return updated_memory


class SimpleCloudEncoder(nn.Module):
    """Simple chunk encoder (standalone) with moderate regularization."""
    
    def __init__(self, vocab_size, d_model=128, d_cloud=256, max_tokens=256,
                 nhead=4, n_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.zeros(1, max_tokens, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=d_model*2, dropout=dropout, batch_first=True
        )
        self.encoder_stack = nn.Sequential(*[encoder_layer for _ in range(n_layers)])
        
        self.cloud_proj = nn.Sequential(
            nn.Linear(d_model, d_cloud),
            nn.Dropout(dropout)
        )
    
    def forward(self, tokens, mask=None):
        x = self.embedding(tokens)
        x = x + self.pos_enc[:, :tokens.size(1), :]
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder_stack(x)
        cloud_state = self.cloud_proj(x).mean(dim=1)  # (B, Dim)
        return cloud_state


class MemoryContextModel(nn.Module):
    """
    Context model with Memory Network + Model Küçültme.
    Processes chunks sequentially, reading/writing to memory.
    """
    
    def __init__(self, vocab_size, d_model=128, d_sentence=256, max_tokens=256,
                 num_chunks=4, num_memory=64, dropout=0.2):
        super().__init__()
        self.num_chunks = num_chunks
        
        # Chunk encoder
        self.chunk_encoder = SimpleCloudEncoder(
            vocab_size, d_model=d_model, d_cloud=d_sentence, max_tokens=max_tokens,
            dropout=dropout
        )
        
        # Memory bank
        self.memory = MemoryBank(num_memory=num_memory, dim=d_sentence)
        
        # Final prediction head with dropout
        self.final_head = nn.Sequential(
            nn.Linear(d_sentence * 2, d_sentence),  # chunk_state + memory_read
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(d_sentence, vocab_size)
        )
    
    def forward(self, chunk_tokens, chunk_masks):
        """
        Process chunks sequentially with memory.
        
        Args:
            chunk_tokens: (B, NumChunks, MaxTokens)
            chunk_masks: (B, NumChunks, MaxTokens)
        
        Returns:
            logits: (B, Vocab)
        """
        B, NC, MT = chunk_tokens.shape
        
        # Initialize memory (clone for each batch)
        memory = self.memory.memory.unsqueeze(0).expand(B, -1, -1).clone()
        
        final_states = []
        
        for c in range(NC):
            chunk = chunk_tokens[:, c, :]  # (B, MaxTokens)
            mask = chunk_masks[:, c, :]
            
            # Encode chunk
            chunk_state = self.chunk_encoder(chunk, mask)  # (B, Dim)
            
            # Read from memory
            memory_read = self.memory.read(chunk_state)  # (B, Dim)
            
            # Combine chunk state and memory read
            combined = torch.cat([chunk_state, memory_read], dim=-1)  # (B, 2*Dim)
            final_states.append(combined)
            
            # Write to memory
            memory = self.memory.write(chunk_state, memory)  # (B, NumMemory, Dim)
        
        # Use last chunk's combined state for prediction
        final_state = final_states[-1]  # (B, 2*Dim)
        return self.final_head(final_state)


def train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=1.0):
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (chunks, masks) in enumerate(dataloader):
        chunks = chunks.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        chunks, masks, target = shift_target_3d(chunks, masks)
        logits = model(chunks, masks)

        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if batch_idx % 50 == 0:
            logger.info("Batch %d/%d | Loss: %.4f", batch_idx, len(dataloader), loss.item())
    
    return total_loss / num_batches


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for chunks, masks in dataloader:
            chunks = chunks.to(device)
            masks = masks.to(device)
            
            chunks, masks, target = shift_target_3d(chunks, masks)
            logits = model(chunks, masks)

            loss = criterion(logits, target)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Memory-context eğitimi")
    parser.add_argument("--vocab-size", type=int, default=0,
                        help="Vocab sınırı (0 = sınırsız); stable ile kontrollü kıyas için 10000")
    parser.add_argument("--ckpt-name", default="context_memory_best.pt")
    parser.add_argument("--d-model", type=int, default=64,
                        help="Gömme boyutu; stable boyut kontrolü için 128")
    parser.add_argument("--d-sentence", type=int, default=128,
                        help="Cümle boyutu; stable boyut kontrolü için 256")
    parser.add_argument("--seed", type=int, default=None,
                        help="torch RNG seedi (model init + shuffle); None = tohumlanmamış "
                             "(R5 orijinal davranışıyla birebir)")
    args = parser.parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)

    DATA_FILE = "data/turkish_sentences.txt"
    OUTPUT_DIR = "contextcloud/checkpoints"
    CHUNK_SIZE = 5
    MAX_TOKENS = 256
    NUM_CHUNKS = 4
    NUM_MEMORY = 32  # 64 → 32 (model küçültme)
    BATCH_SIZE = 16
    LEARNING_RATE = 5e-4  # 1e-3 → 5e-4 (daha stabil)
    EPOCHS = 30
    PATIENCE = 5
    DROPOUT = 0.2  # 0.4 → 0.2 (moderate)
    WEIGHT_DECAY = 0.02  # 0.1 → 0.02 (moderate L2)
    GRAD_CLIP = 1.0
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info("=== MODEL KÜÇÜLTME STRATEJİSİ ===")
    logger.info("Memory: %d, Dropout: %.2f, Weight Decay: %.2f, LR: %.5f", NUM_MEMORY, DROPOUT, WEIGHT_DECAY, LEARNING_RATE)
    logger.info("Torch seed: %s | ckpt: %s | veri bölme seed=42 sabit (AGENTS §4.1)",
                args.seed, args.ckpt_name)
    
    if not os.path.exists(DATA_FILE):
        logger.error("Data file not found: %s", DATA_FILE)
        return
    
    # Ayrık 80/10/10 bölme (AGENTS §4.1, seed=42); vocab yalnız train'den
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    train_lines, val_lines, test_lines = split_sentences(lines)
    logger.info("Split: train %d / val %d / test %d (seed=42)",
                len(train_lines), len(val_lines), len(test_lines))

    dataset = ChunkedVocabDataset(train_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS,
                                   vocab_size=args.vocab_size or None)
    val_dataset = ChunkedVocabDataset(val_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    test_dataset = ChunkedVocabDataset(test_lines, chunk_size=CHUNK_SIZE, max_tokens=MAX_TOKENS, num_chunks=NUM_CHUNKS, vocab=dataset.vocab)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_chunks)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_chunks)
    
    vocab_size = len(dataset.vocab)
    logger.info("Vocab size: %d", vocab_size)
    
    model = MemoryContextModel(
        vocab_size=vocab_size,
        d_model=args.d_model,  # varsayılan 64 (küçültme); kontrollü koşuda 128
        d_sentence=args.d_sentence,  # varsayılan 128 (küçültme); kontrollü koşuda 256
        max_tokens=MAX_TOKENS,
        num_chunks=NUM_CHUNKS,
        num_memory=NUM_MEMORY,
        dropout=DROPOUT
    )
    model = model.to(device)
    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(1, EPOCHS + 1):
        logger.info("\n=== Epoch %d/%d ===", epoch, EPOCHS)
        
        train_loss = train_epoch(model, dataloader, optimizer, criterion, device, grad_clip=GRAD_CLIP)
        val_loss = evaluate(model, val_loader, criterion, device)
        
        logger.info("Epoch %d | Train Loss: %.4f | Val Loss: %.4f", epoch, train_loss, val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint_path = os.path.join(OUTPUT_DIR, args.ckpt_name)
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "epoch": epoch,
                "val_loss": val_loss,
                "num_memory": NUM_MEMORY
            }, checkpoint_path)
            logger.info("Saved best checkpoint (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            logger.info("Patience: %d/%d", patience_counter, PATIENCE)
        
        if patience_counter >= PATIENCE:
            logger.info("Early stopping triggered")
            break
    
    logger.info("\n=== TRAINING COMPLETE ===")
    logger.info("Best val loss: %.4f", best_val_loss)

    checkpoint_path = os.path.join(OUTPUT_DIR, args.ckpt_name)
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    test_loss = evaluate(model, test_loader, criterion, device)
    logger.info("TEST loss (best ckpt, ayrık holdout): %.4f", test_loss)


if __name__ == "__main__":
    main()
