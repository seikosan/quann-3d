#!/usr/bin/env python3
"""R4: güncel kodla yeniden eğitilen D32/D64 checkpoint'lerinin eşleşmiş değerlendirmesi.

Protokol eval_r1_duel ile birebir aynıdır (satırlar [200000,202000), n=2000,
seed=0, seq_len=10); yalnız checkpoint yolları R4 dizinlerini gösterir.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from eval_r1_duel import evaluate

N = 2_000
SEED = 0
HOLDOUT_START = 200_000
HOLDOUT_END = 202_000
SEQ_LEN = 10


def main():
    d32_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_r4_d32_200k", "wiki_cloud_best.pt")
    d64_path = os.path.join(ROOT, "checkpoints", "wiki_checkpoints_r4_d64_200k", "wiki_cloud_best.pt")
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
