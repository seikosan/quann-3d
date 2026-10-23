#!/usr/bin/env python3
"""R3 max vs attention pooling comparison on fixed wiki rows [200000, 202000)."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare_llm import evaluate_ppl, get_device, load_cloud_ckpt, load_bpe


CORPUS = os.path.join(ROOT, "data", "turkish_wiki_1500k.txt")
BPE_PATH = os.path.join(ROOT, "checkpoints", "r3_20k", "max", "wiki_bpe.json")
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
        f"{label}_R3_RESULT | epoch={state['epoch']} | train_PPL={state['ppl']:.2f} "
        f"| heldout_PPL={ppl:.2f} | BPE={BPE_PATH} | n={N} | seed={SEED} "
        f"| rows=[{HOLDOUT_START},{HOLDOUT_END}) | seq_len={SEQ_LEN} | tokens={tokens}",
        flush=True,
    )
    return ppl, state


def main():
    max_path = os.path.join(ROOT, "checkpoints", "r3_20k", "max", "wiki_cloud_best.pt")
    attn_path = os.path.join(ROOT, "checkpoints", "r3_20k", "attn", "wiki_cloud_best.pt")
    max_ppl, max_state = evaluate("MAX", max_path)
    attn_ppl, attn_state = evaluate("ATTN", attn_path)
    delta = attn_ppl - max_ppl
    winner = "ATTN" if delta < 0 else "MAX" if delta > 0 else "tie"
    print(
        f"R3_DUEL_SUMMARY | holdout_PPL_delta_ATN_minus_MAX={delta:.2f} "
        f"| winner={winner} | ATTN_epoch={attn_state['epoch']} | MAX_epoch={max_state['epoch']} "
        f"| n={N} | seed={SEED} | rows=[{HOLDOUT_START},{HOLDOUT_END}) | seq_len={SEQ_LEN}",
        flush=True,
    )


if __name__ == "__main__":
    main()
