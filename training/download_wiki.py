"""
Quann-3D — Büyük Türkçe Wikipedia Corpus İndirme + Cümleye Ayırma
==================================================================
HuggingFace: Alaeddin/wikipedia-turkish (1.65M paragraf, 2024 dump)
Paragrafları cümleye böler, 20-300 karakter filtresi uygular.

Kullanım:
  python3 download_wiki.py                 # tam corpus
  python3 download_wiki.py --max-rows 200000   # alt küme
"""

import re
import os

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import argparse
from datasets import load_dataset


def split_sentences(text: str):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p]


def pick_text_column(ds):
    """En uzun ortalama metin uzunluğuna sahip string kolonu seç."""
    best, best_len = None, -1
    for c in ds.column_names:
        try:
            sample = ds[c][:200]
            vals = [str(x) for x in sample if x is not None]
            if not vals:
                continue
            avg = sum(len(v) for v in vals) / len(vals)
            if avg > best_len:
                best, best_len = c, avg
        except Exception:
            continue
    return best, best_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf-repo", type=str, default="Alaeddin/wikipedia-turkish")
    ap.add_argument("--out", type=str,
                    default=os.path.join(ROOT, "data", "turkish_wiki_sentences.txt"))
    ap.add_argument("--min-len", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=300)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    print(f"İndiriliyor: {args.hf_repo}")
    ds = load_dataset(args.hf_repo, split="train")
    print(f"  Satır: {len(ds)} | Kolonlar: {ds.column_names}")

    col, avg = pick_text_column(ds)
    print(f"  Metin kolonu: {col!r} (ort. {avg:.0f} karakter)")

    n_in = n_out = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if args.max_rows and i >= args.max_rows:
                break
            text = row.get(col)
            if not text:
                continue
            n_in += 1
            for sent in split_sentences(text):
                if args.min_len <= len(sent) <= args.max_len:
                    f.write(sent + "\n")
                    n_out += 1
            if (i + 1) % 100000 == 0:
                print(f"  {i+1}/{len(ds)} paragraf → {n_out} cümle")

    print(f"\n✅ {n_in} paragraf → {n_out} cümle")
    print(f"   Çıktı: {args.out}")


if __name__ == "__main__":
    main()
