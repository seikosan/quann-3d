"""Unsupervised weekly drift audit over fixed train-window cloud prototypes."""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin

from config import FIGURES_DIR, N_REGIONS, RESULTS_DIR
from datasets_cfpb import load_cfpb
from explain_region import cloud_states, load_model


def weekly_distribution_drift(weeks, region_ids, n_regions, low_threshold=0.01,
                              new_threshold=0.05):
    """Return chronological weekly shares, TV/KL movement, and new-region flags."""
    grouped = {}
    for week, region in zip(weeks, region_ids):
        grouped.setdefault(str(week), np.zeros(n_regions, dtype=np.int64))[int(region)] += 1
    ordered_weeks = sorted(grouped)
    report = {}
    previous = None
    history = []
    for week in ordered_weeks:
        counts = grouped[week]
        shares = counts / max(int(counts.sum()), 1)
        tv = None
        kl = None
        if previous is not None:
            tv = float(0.5 * np.abs(shares - previous).sum())
            eps = 1e-8
            p = np.clip(shares, eps, 1.0)
            q = np.clip(previous, eps, 1.0)
            kl = float(np.sum(p * np.log(p / q)))
        baseline = np.mean(history, axis=0) if history else None
        new_regions = []
        if baseline is not None:
            new_regions = [int(index) for index in range(n_regions)
                           if baseline[index] < low_threshold and shares[index] > new_threshold]
        report[week] = {
            "n_records": int(counts.sum()),
            "region_shares": [float(value) for value in shares],
            "tv_vs_previous_week": tv,
            "kl_vs_previous_week": kl,
            "new_regions": new_regions,
        }
        history.append(shares)
        previous = shares
    return report


def draw_drift_figure(region_map, drift_report, output_path):
    regions = region_map["regions"]
    centers = np.asarray([regions[str(i)]["centroid"]
                          for i in range(region_map["n_regions"])], dtype=np.float32)
    coordinates = PCA(n_components=2, random_state=0).fit_transform(centers)
    weeks = list(drift_report)
    fig, (ax_map, ax_share) = plt.subplots(1, 2, figsize=(12, 5))
    for index, point in enumerate(coordinates):
        ax_map.scatter(point[0], point[1], s=90)
        ax_map.annotate(f"R{index}: {regions[str(index)]['name'][:24]}", point,
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax_map.set_title("Train-window region prototypes (PCA)")
    ax_map.set_xlabel("PC1")
    ax_map.set_ylabel("PC2")
    if weeks:
        shares = np.asarray([drift_report[week]["region_shares"] for week in weeks])
        x = np.arange(len(weeks))
        for region_id in range(shares.shape[1]):
            ax_share.plot(x, shares[:, region_id], linewidth=0.9, label=f"R{region_id}")
        ticks = np.linspace(0, len(weeks) - 1, min(8, len(weeks)), dtype=int)
        ax_share.set_xticks(ticks, [weeks[index] for index in ticks], rotation=45, ha="right")
        ax_share.set_ylabel("Weekly share")
        ax_share.legend(ncol=2, fontsize=7)
    ax_share.set_title("Drift-window region shares")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def audit(frame, cloud_states_array, region_map):
    drift = frame.loc[frame["split"] == "drift"].reset_index(drop=True)
    states = np.asarray(cloud_states_array, dtype=np.float32)
    if len(states) != len(drift):
        raise ValueError("drift states must align with drift rows")
    centers = np.asarray([region_map["regions"][str(i)]["centroid"]
                          for i in range(region_map["n_regions"])], dtype=np.float32)
    assignments = pairwise_distances_argmin(states, centers)
    report = weekly_distribution_drift(
        drift["week"].astype(str).tolist(), assignments, len(centers),
    )
    return {"method": "fixed train-window KMeans prototypes; no drift labels",
            "n_drift": len(drift), "weeks": report}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=None)
    parser.add_argument("--checkpoint", default=os.path.join(RESULTS_DIR, "reader.pt"))
    parser.add_argument("--tokenizer", default=os.path.join(RESULTS_DIR, "tokenizer.json"))
    parser.add_argument("--region-map", default=os.path.join(RESULTS_DIR, "region_map.json"))
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "drift_report.json"))
    parser.add_argument("--figure", default=os.path.join(FIGURES_DIR, "drift_region_map.png"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    with open(args.region_map, encoding="utf-8") as handle:
        region_map = json.load(handle)
    model, tokenizer, checkpoint = load_model(args.checkpoint, args.tokenizer)
    n_sample = args.n_sample or checkpoint["config"]["n_sample"]
    frame = load_cfpb(n_sample=n_sample, seed=checkpoint["config"]["seed"],
                      data_dir=args.data_dir)
    drift = frame.loc[frame["split"] == "drift"].reset_index(drop=True)
    states = cloud_states(model, tokenizer, drift["narrative"].tolist(),
                          checkpoint["config"]["max_len"])
    result = audit(frame, states, region_map)
    result["new_region_events"] = [
        {"week": week, "region": region,
         "theme": region_map["regions"][str(region)]["name"]}
        for week, detail in result["weeks"].items()
        for region in detail["new_regions"]
    ]
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    draw_drift_figure(region_map, result["weeks"], args.figure)
    print(json.dumps({"output": args.output, "figure": args.figure,
                      "n_weeks": len(result["weeks"]),
                      "new_region_events": result["new_region_events"][:10]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
