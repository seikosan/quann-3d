"""
Faz 2: Tiny Corpus (20K cümle) — QUANN-Cloud
=============================================

EXPERIMENTS.md'ye göre:
- Corpus: turkish_sentences.txt'in ilk 20K cümlesi
- Tokenizer: wiki BPE (20K vocab)
- Model: d_model=256, d_cloud=512, D=16, 4 katman (~21M param)
- Epoch: 10 | Seq len: 10 | Backprop YOK
- Eval: 20K-22K arası held-out cümleler

Kullanım:
  python3 tiny_corpus.py                                  # tam Faz 2 (20K, 10 epoch)
  python3 tiny_corpus.py --max-texts 2000 --epochs 1     # pilot (süre tahmini)
"""

import sys
import os
import argparse
import random

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.path.join(ROOT, 'quann_cloud', 'src'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "training"))

import torch
from qcloud_trainer import QCloudTrainer
from qcloud_network import QCloudNetwork
from train_wiki import BPETokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-texts", type=int, default=20000, help="Eğitim cümle sayısı")
    ap.add_argument("--eval-texts", type=int, default=2000, help="Held-out cümle sayısı")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--eval-interval", type=int, default=1)
    ap.add_argument("--corpus", type=str, default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--bpe", type=str, default=os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json"))
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--d-cloud", type=int, default=512)
    ap.add_argument("--d-dim", type=int, default=16, help="Bulut boyutu D")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=10)
    ap.add_argument("--hybrid", action="store_true",
                    help="Hibrit mod: ağ backprop ile, QUANN noktaları arama ile")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print(f"Loading BPE: {args.bpe}")
    tok = BPETokenizer.load(args.bpe)
    print(f"  vocab: {tok.num_tokens}")

    # Eğitim: ilk max_texts; eval: hemen sonraki eval_texts (held-out)
    print(f"Loading corpus: {args.corpus}")
    with open(args.corpus, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f]
    texts = [t for t in lines if 20 <= len(t) <= 300]
    corpus = texts[:args.max_texts]
    eval_corpus = texts[args.max_texts:args.max_texts + args.eval_texts]
    print(f"  train: {len(corpus)} | eval: {len(eval_corpus)}")

    vocab_size = tok.num_tokens
    model = QCloudNetwork(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_cloud=args.d_cloud,
        d_cloud_dim=args.d_dim,
        n_layers=args.layers,
        bidirectional=True,
    )
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params | D={args.d_dim} | device={device}")

    trainer = QCloudTrainer(
        model=model,
        corpus=corpus,
        device=device,
        tokenizer=tok,
        eval_corpus=eval_corpus,
        max_epochs=args.epochs,
        eval_interval=args.eval_interval,
        seq_len=args.seq_len,
        hybrid=args.hybrid,
        lr=args.lr,
    )

    mode = "hybrid" if args.hybrid else "pure"
    tag = f"tiny_{mode}_{args.max_texts}x{args.epochs}ep"
    save_path = os.path.join(ROOT, "quann_cloud", "logs", f"qcloud_{tag}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    result = trainer.train(verbose=True, save_path=save_path)

    print("\n=== SONUÇ ===")
    print(f"Best PPL: {result['best_ppl']:.2f} (epoch {result['best_epoch']})")
    print(f"Random baseline PPL ≈ {vocab_size} (untrained head)")
    print(f"Model: {save_path}")

    # Örnek üretimler
    print("\n=== ÜRETİM ÖRNEKLERİ (temp=0.8) ===")
    for prompt in ["Türkiye'nin başkenti", "bilgisayar bilimlerinde",
                   "sanat toplumun", "ekonomik büyüme"]:
        out = trainer.generate_text(prompt, max_new=10, temperature=0.8)
        print(f"  {prompt!r:30s} → {out[:70]}")


if __name__ == "__main__":
    main()
