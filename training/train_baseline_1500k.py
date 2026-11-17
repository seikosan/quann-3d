"""Baseline Transformer (CloudLM olmadan) - 1.5M Wikipedia"""
import sys, os, torch, torch.nn as nn, torch.nn.functional as F, math, time, random, argparse, numpy as np

ROOT = "/Users/serkankose/Documents/quann-3d"
sys.path.insert(0, ROOT)
from cloudlm import BPETokenizer

def iter_shuffled_batches(all_tokens, seq_len, batch_size):
    chunks = []
    chunk_start = None
    chunk_count = 0
    chunk_remaining = 8192
    for sentence_idx, tokens in enumerate(all_tokens):
        n_examples = max(0, len(tokens) - seq_len)
        local_start = 0
        while local_start < n_examples:
            take = min(chunk_remaining, n_examples - local_start)
            if chunk_count == 0:
                chunk_start = (sentence_idx, local_start)
            chunk_count += take
            local_start += take
            chunk_remaining -= take
            if chunk_remaining == 0:
                chunks.append((*chunk_start, chunk_count))
                chunk_start = None
                chunk_count = 0
                chunk_remaining = 8192
    if chunk_count:
        chunks.append((*chunk_start, chunk_count))
    random.shuffle(chunks)
    pending = []
    for sentence_idx, local_start, chunk_count in chunks:
        remaining = chunk_count
        examples = []
        while remaining:
            tokens = all_tokens[sentence_idx]
            n_examples = max(0, len(tokens) - seq_len)
            take = min(remaining, n_examples - local_start)
            if take:
                examples.extend(
                    (tokens[i:i+seq_len], tokens[i+seq_len])
                    for i in range(local_start, local_start + take)
                )
                remaining -= take
            sentence_idx += 1
            local_start = 0
        random.shuffle(examples)
        if pending:
            examples = pending + examples
        full_size = len(examples) // batch_size * batch_size
        for b in range(0, full_size, batch_size):
            yield examples[b:b+batch_size]
        pending = examples[full_size:]
    if pending:
        yield pending

class BaselineTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_layers=6, dim_feedforward=1024, dropout=0.1, max_seq=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(max_seq, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.decoder = nn.Linear(d_model, vocab_size)
        self._init_weights()
    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
    def forward(self, x, mask=None):
        b, s = x.shape
        pos = torch.arange(s, device=x.device).unsqueeze(0)
        h = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        h = h + self.pos_encoder(pos)
        h = self.transformer(h, src_key_padding_mask=(x==0))
        return self.decoder(h[:, -1, :])

def load_corpus(path, max_lines=None, min_len=20, max_len=300):
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines and i >= max_lines:
                break
            t = line.strip()
            if min_len <= len(t) <= max_len:
                texts.append(t)
    return texts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    parser.add_argument("--save-dir", default=os.path.join(ROOT, "checkpoints", "wiki_checkpoints_baseline_1500k"))
    parser.add_argument("--bpe-vocab", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    tok_path = os.path.join(args.save_dir, "wiki_bpe.json")
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        print(f"Loaded BPE: {tok.num_tokens} tokens")
    else:
        print("Training BPE tokenizer...")
        texts = load_corpus(args.corpus, 50000)
        tok = BPETokenizer(vocab_size=args.bpe_vocab)
        tok.train(texts)
        tok.save(tok_path)
        print(f"Saved BPE: {tok.num_tokens} tokens")
    
    texts = load_corpus(args.corpus)
    print(f"Corpus: {len(texts)} cümle")
    
    all_tokens = [tok.encode(t) for t in texts]
    all_tokens = [t for t in all_tokens if len(t) >= args.seq_len + 1]
    print(f"Valid: {len(all_tokens)} cümle (≥{args.seq_len+1} token)")
    
    model = BaselineTransformer(tok.num_tokens, d_model=args.d_model, num_layers=args.layers)
    model.to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        start_epoch = ckpt.get("epoch", 0)
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print(f"Resume: epoch {start_epoch}")
    
    best_ppl = ckpt.get("best_ppl", float('inf')) if args.resume else float('inf')
    
    model.train()
    t0 = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_loss, n_batches = 0, 0
        for batch in iter_shuffled_batches(all_tokens, args.seq_len, args.batch):
            inputs = torch.tensor([e[0] for e in batch], device=device)
            targets = torch.tensor([e[1] for e in batch], device=device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
            
            if not torch.isfinite(loss):
                continue
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        
        scheduler.step()
        avg = epoch_loss / max(n_batches, 1)
        ppl = math.exp(min(avg, 20))
        elapsed = time.time() - t0
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | loss={avg:.4f} | ppl={ppl:.2f} | {elapsed:.0f}s")
        
        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "vocab_size": tok.num_tokens,
            "config": {"d_model": args.d_model, "n_layers": args.layers},
            "ppl": ppl,
            "best_ppl": best_ppl,
        }
        
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(checkpoint, os.path.join(args.save_dir, "baseline_best.pt"))
        
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.save_dir, f"baseline_epoch{epoch+1}.pt"))
    
    print(f"\n✅ Best PPL: {best_ppl:.2f}")

if __name__ == "__main__":
    main()
