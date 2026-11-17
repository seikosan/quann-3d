"""Measure baseline INT8-channel operating counts on the reproducible full test."""
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.abspath("fraud/src"))
import datasets
import edge_eval
import metrics
import train_quann as tq
from qcloud_fraud import QCloudFraudClassifier
from quantize import FeatureQuantizer


def main():
    torch.set_num_threads(4)
    ckpt = "fraud/results/quann_full_hybrid.pt"
    raw = torch.load(ckpt, map_location="cpu")
    fq = FeatureQuantizer.from_dict(raw["quantizer"])
    usecols = ["User", "Is Fraud?", "Amount", "Time", "Month", "Day", "Year",
               "Zip", "Use Chip", "MCC", "Merchant City", "Merchant Name",
               "Merchant State", "Errors?"]
    df = pd.read_csv(datasets.config.BIG_CSV, usecols=usecols)
    data = tq.build_data(df, 1_020_000, 100_000, 4_700_000, 0.02, 0.0013,
                         seed=datasets.config.SEED, quantizer=fq)
    y = data["y"][data["te_sel"]]
    ids = torch.as_tensor(data["tok_te"], dtype=torch.long)
    del data, df
    mask = (ids != 0) & (ids != 1)
    model = QCloudFraudClassifier.load_quann(ckpt, torch.device("cpu"))
    dp = edge_eval.DeployPath(model).eval()
    q = edge_eval._to_quant(__import__("copy").deepcopy(dp), weight_per_channel=True)
    edge_eval._calibrate(q, ids[:50000], mask[:50000])
    p = edge_eval._chunked_proba(q, ids, mask, chunk=8192)
    m = metrics.evaluate(y, p, threshold=0.9969438910484314)
    m["auc_pr"] = float(average_precision_score(y, p))
    m["auc_roc"] = float(roc_auc_score(y, p))
    out = {"model": "baseline quann_full_hybrid.pt", "threshold": 0.9969438910484314,
           "int8_channel": m}
    with open("fraud/results/online_cloud_int8_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"BASELINE INT8-channel: FP={m['fp']:,} TP={m['tp']:,} FN={m['fn']:,} "
          f"P={m['precision']:.4f} R={m['recall']:.4f} "
          f"PR={m['auc_pr']:.4f} ROC={m['auc_roc']:.4f}", flush=True)


if __name__ == "__main__":
    main()
