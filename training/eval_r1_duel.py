#!/usr/bin/env python3
"""Matched D32/D64 evaluation on fixed wiki rows [200000, 202000)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_llm import evaluate_ppl, get_device, load_cloud_ckpt, load_bpe


CORPUS = os.path.join(ROOT, "data", "turkish_wiki_1500k.txt")
BPE_PATH = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_200k", "wiki_bpe.json")
HOLDOUT_START = 200_000
HOLDOUT_END = 202_000
SEQ_LEN = 10
N = 2_000
SEED = 0


def load_fixed_holdout():
    texts = []
    with open(CORPUS, "r", encoding="utf-8") as source:
        for row, line in enumerate(source):
            if row >= HOLDOUT_END:
                break
            if row >= HOLDOUT_START:
                texts.append(line.strip())
    if len(texts) != N:
        raise RuntimeError(
            f"Expected {N} source rows [{HOLDOUT_START},{HOLDOUT_END}); got {len(texts)}"
        )
    return texts


def evaluate(label, checkpoint):
    model, state = load_cloud_ckpt(checkpoint)
    device = get_device()
    model.to(device)
    tokenizer = load_bpe(BPE_PATH)
    ppl, _, tokens = evaluate_ppl(
        model, tokenizer, load_fixed_holdout(), device, SEQ_LEN,
        "cloud", n=N, seed=SEED,
    )
    print(
        f"{label}_DUEL_RESULT | epoch={state['epoch']} | train_PPL={state['ppl']:.2f} "
        f"| heldout_PPL={ppl:.2f} | BPE={BPE_PATH} | n={N} | seed={SEED} "
        f"| rows=[{HOLDOUT_START},{HOLDOUT_END}) | seq_len={SEQ_LEN} | tokens={tokens}",
        flush=True,
    )
    return ppl, state


def main():
    d32_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_wiki200k_d32", "wiki_cloud_best.pt")
    d64_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_wiki200k_d64", "wiki_cloud_best.pt")
    d32_ppl, d32_state = evaluate("D32", d32_path)
    d64_ppl, d64_state = evaluate("D64", d64_path)
    delta = d32_ppl - d64_ppl
    winner = "D32" if delta < 0 else "D64" if delta > 0 else "tie"
    print(
        f"DUEL_SUMMARY | holdout_PPL_delta_D32_minus_D64={delta:.2f} "
        f"| winner={winner} | D32_epoch={d32_state['epoch']} | D64_epoch={d64_state['epoch']} "
        f"| n={N} | seed={SEED} | rows=[{HOLDOUT_START},{HOLDOUT_END}) | seq_len={SEQ_LEN}",
        flush=True,
    )


if __name__ == "__main__":
    main()
