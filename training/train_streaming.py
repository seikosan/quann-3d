"""
CloudLM Eğitim - Hafıza Dostu 1.5M Versiyonu
=============================================
- Veriyi satır satır okur (memory'de tutmaz)
- Batch batch tokenize eder
- Shuffled iterator kullanır
"""

import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import random
from cloudlm import BPETokenizer, CloudLM

class StreamingTokenizer:
    """Veriyi satır satır okur, tokenize eder, yield eder."""
    
    def __init__(self, tokenizer, seq_len, batch_size):
        self.tok = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.buffer = []
        
    def stream(self, path, max_lines=None):
        """Veri dosyasından satır satır okur, tokenize edilmiş batch'ler yield eder."""
        batch = []
        line_count = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if max_lines and line_count >= max_lines:
                    break
                line_count += 1
                
                # Tokenize
                tokens = self.tok.encode(line.strip())
                
                # Yeterli uzunlukta mı?
                if len(tokens) >= self.seq_len + 1:
                    # Tüm pencere örneklerini ekle
                    for i in range(0, len(tokens) - self.seq_len):
                        batch.append((tokens[i:i+self.seq_len], tokens[i+self.seq_len]))
                        
                        # Batch dolu mu?
                        if len(batch) >= self.batch_size:
                            random.shuffle(batch)
                            yield batch
                            batch = []
        
        # Kalan batch
        if batch:
            random.shuffle(batch)
            yield batch

def load_bpe_tokenizer(save_dir):
    """BPE tokenizer'ı yükle."""
    bpe_path = os.path.join(save_dir, "wiki_bpe.json")
    if not os.path.exists(bpe_path):
        print(f"ERROR: BPE dosyası bulunamadı: {bpe_path}")
        print("Önce tokenizer oluştur: python3 -m training.train_wiki --step tokenizer")
        exit(1)
    return BPETokenizer.load(bpe_path)

def main():
    # Parametreler
    corpus = "data/turkish_wiki_1500k.txt"
    max_texts = 1500000  # 1.5M cümle
    epochs = 20
    batch_size = 64
    seq_len = 20
    d_model = 256
    d_cloud = 512
    d_dim = 64
    layers = 8
    save_dir = "checkpoints/wiki_checkpoints_r2_cloud_attn_1500k_d64_n20"
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    
    print("=" * 60)
    print("  CLOUDLM EĞİTİM - 1.5M HAFIZA DOSTU")
    print("=" * 60)
    
    # Tokenizer yükle
    print("\n[1/5] Tokenizer yükleniyor...")
    tok = load_bpe_tokenizer(save_dir)
    print(f"  Vocab: {tok.num_tokens}")
    
    # Model oluştur
    print("\n[2/5] Model oluşturuluyor...")
    model = CloudLM(
        tok.num_tokens,
        d_model=d_model,
        d_cloud=d_cloud,
        d_cloud_dim=d_dim,
        n_layers=layers,
        pool="max",
        cloud_attn=True,
        cloud_attn_heads=8,
        attn_scale_init=1.0,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    # Streaming tokenizer
    streamer = StreamingTokenizer(tok, seq_len, batch_size)
    
    print(f"\n[3/5] Eğitim başlıyor ({corpus})")
    print(f"  Max cümle: {max_texts}")
    print(f"  Batch: {batch_size}, Seq len: {seq_len}")
    print(f"  Device: {device}")
    print()
    
    t0 = time.time()
    best_ppl = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        
        # Streaming batch'ler
        for batch in streamer.stream(corpus, max_lines=max_texts):
            if len(batch) < 2:
                continue
            
            inputs = torch.tensor([b[0] for b in batch], device=device)
            targets = torch.tensor([b[1] for b in batch], device=device)
            eos_id = tok._tokenizer.token_to_id("<eos>") or 1
            mask = (inputs != 0) & (inputs != eos_id)
            
            optimizer.zero_grad()
            logits, _, _ = model(inputs, mask)
            loss = F.cross_entropy(logits, targets)
            
            if not torch.isfinite(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        epoch_time = time.time() - epoch_start
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {epoch_time:.0f}s | total={elapsed:.0f}s", flush=True)
        
        # Checkpoint
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {
                "d_model": d_model, "d_cloud": d_cloud, "d_cloud_dim": d_dim,
                "n_layers": layers, "cloud_attn": True, "cloud_attn_heads": 8
            },
            "ppl": ppl,
            "best_ppl": best_ppl,
        }
        
        # Her epoch checkpoint
        torch.save(checkpoint, os.path.join(save_dir, f"wiki_cloud_epoch{epoch+1}.pt"))
        
        # Best checkpoint
        if ppl < best_ppl:
            best_ppl = ppl
            checkpoint["best_ppl"] = best_ppl
            torch.save(checkpoint, os.path.join(save_dir, "wiki_cloud_best.pt"))
            print(f"    ✅ New best: {best_ppl:.2f}", flush=True)
    
    total = time.time() - t0
    print(f"\n  ✅ Eğitim tamamlandı: {total:.0f}s ({total/60:.1f}dk)")
    print(f"  Best PPL: {best_ppl:.2f}")

if __name__ == "__main__":
    main()
