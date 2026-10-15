"""
QUANN-Cloud — QUANN Motoru ile Gerçek Eğitim
==============================================

QCloudNetwork'ün nn.Module parametrelerini (embedding, transformer, head)
QUANN motoru ile backprop'suz eğitir.

Bu, projenin ÇEKİRDEK deneyi:
  - Gradyan YOK
  - Sadece forward + ödül/ceza + aralık daraltma
  - Stokastik: her adımda K rastgele parametre daraltılır

Çalıştır:
    python3 experiments/quann_training.py
"""

import sys
import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from tokenizers import Tokenizer
from qcloud_network import QCloudNetwork
from qcloud_quann import QUANN, QUANNConfig


def load_corpus(path: str, max_lines: int = 20_000) -> list[str]:
    lines = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            text = line.strip()
            if len(text) > 50:
                lines.append(text)
    return lines


def make_examples(tokenizer: Tokenizer, texts: list[str], seq_len: int = 8,
                  max_examples: int = 500) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """(input_ids, target_id) örnekleri oluştur."""
    examples = []
    for text in texts:
        ids = tokenizer.encode(text).ids
        for i in range(0, len(ids) - 1, seq_len):
            chunk = ids[i:i + seq_len]
            if len(chunk) < seq_len:
                chunk = chunk + [0] * (seq_len - len(chunk))
            target = ids[i + seq_len] if i + seq_len < len(ids) else 1
            examples.append((torch.tensor(chunk), torch.tensor(target)))
            if len(examples) >= max_examples:
                return examples
    return examples


def main():
    print("=" * 60)
    print("  QUANN-Cloud — QUANN Motoru ile Eğitim (Backprop'suz)")
    print("=" * 60)

    # 1. Tokenizer
    print("\n[1/6] BPE Tokenizer...")
    tokenizer = Tokenizer.from_file(os.path.join(ROOT, "data/bpe_5k.json"))
    vocab_size = len(tokenizer.get_vocab())
    print(f"  Vocab: {vocab_size}")

    # 2. Corpus
    print("\n[2/6] Corpus (20K makale)...")
    corpus = load_corpus(os.path.join(ROOT, "data/akademik_100k.txt"), max_lines=20_000)
    print(f"  {len(corpus)} makale")

    # 3. Örnekler
    print("\n[3/6] Örnekler hazırlanıyor (500 örnek, seq_len=8)...")
    examples = make_examples(tokenizer, corpus, seq_len=8, max_examples=100)
    print(f"  {len(examples)} örnek")

    # 4. Model
    print("\n[4/6] Model...")
    model = QCloudNetwork(
        vocab_size=vocab_size,
        d_model=32,
        d_cloud=64,
        d_cloud_dim=4,
        n_layers=1,
    )
    params = sum(p.numel() for p in model.parameters())
    print(f"  {model}")
    print(f"  Parameters: {params:,}")

    # 5. QUANN Motoru
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    cfg = QUANNConfig(
        sigma0=0.3,
        k_weights=30,        # Adımda 30 parametre daralt (hızlı)
        minibatch=2,         # Mini-batch 2
        plateau_window=10,
        seed=42,
    )

    # Loss fonksiyonu: QCloudNetwork forward → cross_entropy
    def loss_fn(m: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        inputs = inputs.to(device)
        targets = targets.to(device)
        mask = (inputs != 0).bool()
        logits, _, _ = m(inputs, mask)
        return F.cross_entropy(logits, targets).item()

    quann = QUANN(model, device, cfg=cfg, loss_fn=loss_fn)
    print(f"\n[5/6] QUANN Motoru hazır")
    info = quann.info()
    print(f"  Total params: {info['total_params']:,}")
    print(f"  Param tensors: {info['n_param_tensors']}")
    print(f"  Initial sigma: {info['sigma_mean']:.3f}")

    # 6. Eğitim
    print(f"\n[6/6] QUANN EĞİTİMİ (5 epoch)...")
    print("-" * 60)

    # Baseline PPL (eğitilmemiş)
    with torch.no_grad():
        base_loss = 0.0
        for inp, tgt in examples[:50]:
            inp = inp.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            mask = (inp != 0).bool()
            logits, _, _ = model(inp, mask)
            base_loss += F.cross_entropy(logits, tgt).item()
        base_ppl = math.exp(base_loss / 50)
    print(f"  Baseline PPL (eğitilmemiş): {base_ppl:.2f}")
    print("-" * 60)

    result = quann.fit(examples, epochs=3, verbose=True)

    # Final PPL
    with torch.no_grad():
        final_loss = 0.0
        for inp, tgt in examples[:50]:
            inp = inp.unsqueeze(0).to(device)
            tgt = tgt.unsqueeze(0).to(device)
            mask = (inp != 0).bool()
            logits, _, _ = model(inp, mask)
            final_loss += F.cross_entropy(logits, tgt).item()
        final_ppl = math.exp(final_loss / 50)

    print("-" * 60)
    print(f"\n  📊 SONUÇ:")
    print(f"     Baseline PPL: {base_ppl:.2f}")
    print(f"     Final PPL:    {final_ppl:.2f}")
    print(f"     İyileşme:     {base_ppl/final_ppl:.2f}x")
    print(f"\n  QUANN Info:")
    info = quann.info()
    print(f"     Mean width:   {info['mean_width']:.6f}")
    print(f"     % Converged:  {info['pct_converged']*100:.1f}%")
    print(f"     Sigma:        {info['sigma_min']:.3f} - {info['sigma_max']:.3f} (mean {info['sigma_mean']:.3f})")

    # Metin üretimi
    print(f"\n  --- Metin Üretimi ---")
    prompt = "Türkiye bir coğrafyadır."
    enc = tokenizer.encode(prompt)
    tokens = torch.tensor([enc.ids], device=device)
    generated = model.generate(tokens, max_new=15, temperature=0.8)
    gen_text = tokenizer.decode(generated[0].tolist())
    print(f"  Prompt:  {prompt}")
    print(f"  Üretilen: {gen_text}")


if __name__ == "__main__":
    main()
