"""
QUANN-Cloud — Transformer vs QCloudNetwork (Hibrit)
=====================================================

ADİL KARŞILAŞTIRMA:
  - Aynı vocab, aynı d_model, aynı layer sayısı
  - Aynı veri, aynı epoch, aynı lr
  - FARK: QCloudNetwork'ün nokta bulut geometrisi (CloudLM)

Soru: Nokta bulut geometrisi, standart transformer'dan DAHA İYİ mi?
"""

import sys, os, time, math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tokenizers import Tokenizer
from qcloud_network import QCloudNetwork
from qcloud_weight import QCloudWeight


class StandardTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_layers=2, n_heads=4, d_ff=256, max_len=256):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=0.1, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens, mask=None):
        T = tokens.size(1)
        x = self.embedding(tokens) * math.sqrt(self.d_model)
        x = x + self.pos_enc[:, :T, :]
        if mask is not None:
            x = x.masked_fill(~mask.unsqueeze(-1), 0)
        x = self.encoder(x, src_key_padding_mask=~mask if mask is not None else None)
        x = self.norm(x)
        return self.head(x)
    
    def generate(self, tokens, max_new=20, temperature=0.8, eos_id=1):
        self.eval()
        generated = tokens.clone()
        with torch.no_grad():
            for _ in range(max_new):
                mask = (generated != 0) & (generated != eos_id)
                logits = self.forward(generated, mask)
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                nxt = torch.multinomial(probs, 1)
                if (nxt == eos_id).all(): break
                generated = torch.cat([generated, nxt], dim=1)
        return generated


def load_corpus(path, max_lines=100_000):
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines: break
            t = line.strip()
            if len(t) > 50: lines.append(t)
    return lines


def make_examples(tokenizer, texts, seq_len=16, max_ex=3000):
    examples = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        for i in range(0, len(ids)-1, seq_len):
            chunk = ids[i:i+seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [0]*(seq_len-len(chunk))
            tgt = ids[i+seq_len] if i+seq_len < len(ids) else 1
            examples.append((torch.tensor(chunk), torch.tensor(tgt)))
            if len(examples) >= max_ex: return examples
    return examples


def eval_ppl(model, examples, device, n=200, is_qcloud=False):
    model.eval()
    total = 0.0
    with torch.no_grad():
        for inp, tgt in examples[:n]:
            inp, tgt = inp.unsqueeze(0).to(device), tgt.unsqueeze(0).to(device)
            mask = (inp != 0).bool()
            if is_qcloud:
                logits, _, _ = model(inp, mask)
            else:
                logits = model(inp, mask)
                logits = logits[:, -1, :]
            total += F.cross_entropy(logits, tgt).item()
    return math.exp(min(total/n, 20))


def train_model(model, examples, device, epochs, lr=1e-3, batch_size=16,
                is_qcloud=False, name="M", token_weights=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    best = float('inf')
    for epoch in range(1, epochs+1):
        total = 0.0
        order = list(range(len(examples)))
        random.shuffle(order)
        for b in range(0, len(order), batch_size):
            idxs = order[b:b+batch_size]
            inps = torch.stack([examples[i][0] for i in idxs]).to(device)
            tgts = torch.stack([examples[i][1] for i in idxs]).to(device)
            mask = (inps != 0).bool()
            if is_qcloud:
                logits, _, _ = model(inps, mask)
            else:
                logits = model(inps, mask)[:, -1, :]
            loss = F.cross_entropy(logits, tgts)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += loss.item() * len(idxs)
        if is_qcloud and token_weights:
            for tid, w in list(token_weights.items())[:20]:
                old = w.value.copy()
                w.value = np.clip(old + np.random.uniform(-0.05, 0.05, w.D), w.low, w.high)
        ppl = math.exp(min(total/len(examples), 20))
        if ppl < best: best = ppl
        if epoch % 5 == 0 or epoch == 1:
            print(f"    {name} E{epoch:2d}: PPL={ppl:.2f}, best={best:.2f}")
    return best


def main():
    print("="*60)
    print("  TRANSFORMER vs QLOUDNETWORK (HİBRİT)")
    print("="*60)

    VOCAB, D_MODEL, N_LAYERS, N_HEADS, D_FF = 5000, 64, 2, 4, 256
    SEQ_LEN, EPOCHS, MAX_EX, BATCH, LR = 16, 20, 3000, 16, 1e-3

    print(f"\n  Config: vocab={VOCAB}, d={D_MODEL}, L={N_LAYERS}, H={N_HEADS}, FF={D_FF}")
    print(f"  Epochs={EPOCHS}, Examples={MAX_EX}, Batch={BATCH}, LR={LR}")

    tokenizer = Tokenizer.from_file(os.path.join(ROOT, "data/bpe_5k.json"))
    corpus = load_corpus(os.path.join(ROOT, "data/akademik_100k.txt"), 100_000)
    examples = make_examples(tokenizer, corpus, SEQ_LEN, MAX_EX)
    print(f"\n  Veri: {len(corpus)} makale, {len(examples)} örnek")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # MODEL A
    print(f"\n[1/2] STANDART TRANSFORMER")
    model_a = StandardTransformerLM(VOCAB, D_MODEL, N_LAYERS, N_HEADS, D_FF).to(device)
    pa = sum(p.numel() for p in model_a.parameters())
    print(f"  Params: {pa:,}")
    ba = eval_ppl(model_a, examples, device)
    print(f"  Baseline: {ba:.2f}")
    t0 = time.time()
    best_a = train_model(model_a, examples, device, EPOCHS, LR, BATCH, False, "Transf")
    ta = time.time()-t0
    fa = eval_ppl(model_a, examples, device)
    print(f"  Final: {fa:.2f}, {ta:.0f}s")

    # MODEL B
    print(f"\n[2/2] QLOUDNETWORK (HİBRİT)")
    model_b = QCloudNetwork(VOCAB, D_MODEL, D_FF, 8, N_LAYERS).to(device)
    pb = sum(p.numel() for p in model_b.parameters())
    unique = set()
    for inp, _ in examples:
        for t in inp.tolist():
            if t != 0: unique.add(t)
    for tid in list(unique)[:100]:
        model_b.get_weight(tid)
    print(f"  Params: {pb:,}, TokenW: {len(model_b.token_weights)}")
    bb = eval_ppl(model_b, examples, device, is_qcloud=True)
    print(f"  Baseline: {bb:.2f}")
    t0 = time.time()
    best_b = train_model(model_b, examples, device, EPOCHS, LR, BATCH, True, "QCloud", model_b.token_weights)
    tb = time.time()-t0
    fb = eval_ppl(model_b, examples, device, is_qcloud=True)
    print(f"  Final: {fb:.2f}, {tb:.0f}s")

    # SONUÇ
    print(f"\n{'='*60}")
    print(f"  📊 SONUÇ")
    print(f"{'='*60}")
    print(f"  {'':22s} {'Transformer':>13s} {'QCloudNet':>13s}")
    print(f"  {'-'*48}")
    print(f"  {'Params':22s} {pa:>12,} {pb:>12,}")
    print(f"  {'Baseline PPL':22s} {ba:>13.2f} {bb:>13.2f}")
    print(f"  {'Best PPL':22s} {best_a:>13.2f} {best_b:>13.2f}")
    print(f"  {'Final PPL':22s} {fa:>13.2f} {fb:>13.2f}")
    print(f"  {'İyileşme':22s} {ba/fa:>12.2f}x {bb/fb:>12.2f}x")
    print(f"  {'Süre':22s} {ta:>12.0f}s {tb:>12.0f}s")
    
    winner = "Transformer" if fa < fb else "QCloudNetwork"
    print(f"\n  🏆 Kazanan: {winner} (PPL: {min(fa,fb):.2f})")
    
    # Metin
    print(f"\n  --- Metin Üretimi ---")
    prompts = ['Türkiye bir coğrafyadır.', 'Bilim ve teknoloji gelişiyor.']
    full_vocab = len(tokenizer.get_vocab())
    for p in prompts:
        enc = tokenizer.encode(p)
        tokens = torch.tensor([enc.ids], device=device)
        gen_a = model_a.generate(tokens, max_new=15, temperature=0.8)
        gen_b = model_b.generate(tokens, max_new=15, temperature=0.8)
        print(f"  Prompt: {p}")
        print(f"    Transf: {tokenizer.decode(gen_a[0].tolist())[:80]}")
        print(f"    QCloud: {tokenizer.decode(gen_b[0].tolist())[:80]}")


if __name__ == "__main__":
    main()
