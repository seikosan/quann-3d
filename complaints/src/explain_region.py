"""Name learned complaint-cloud regions with their most common BPE tokens."""

import argparse
import json
import os
import re

import numpy as np
import torch
from sklearn.cluster import KMeans

from config import N_REGIONS, RESULTS_DIR, SEED
from datasets_cfpb import load_cfpb
from textcloud import ComplaintCloud

_STOP_TOKENS = {
    "the", "to", "i", "and", "a", "of", "in", "is", "it", "for", "my",
    "that", "was", "on", "with", "this", "have", "had", "be", "are", "as",
    "but", "not", "at", "by", "from", "me", "we", "they", "you", "he",
    "she", "their", "our", "your", "when", "what", "which", "who", "how",
    "or", "an", "so", "if", "can", "all", "do", "has", "were", "been",
    "would", "could", "should", "will", "just", "very", "also", "there",
    "then", "them", "than", "out", "up", "into", "about", "after", "before",
    "because", "one", "two", "said", "told", "get", "got", "make", "made",
    "like", "know", "think", "see", "go", "went", "back", "now", "XXXX",
}


def _is_interpretable(token):
    cleaned = token.replace("Ġ", "").replace("Ċ", "").strip()
    normalized = cleaned.casefold()
    return (
        len(cleaned) >= 3
        and bool(re.search(r"[A-Za-z]", cleaned))
        and not normalized.startswith("xxxx")
        and normalized not in {word.casefold() for word in _STOP_TOKENS}
    )


def make_region_map(frame, cloud_states, tokenizer, n_regions=N_REGIONS, seed=SEED,
                   example_limit=3):
    """Fit train-window prototypes and provide token-grounded readable names."""
    states = np.asarray(cloud_states, dtype=np.float32)
    if len(frame) != len(states) or len(frame) == 0:
        raise ValueError("frame and cloud_states must have the same nonzero length")
    n_clusters = min(int(n_regions), len(frame))
    clusterer = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    labels = clusterer.fit_predict(states)
    vocab = tokenizer.vocab
    id_to_token = {idx: token for token, idx in vocab.items()}
    special = {"<pad>", "<eos>", "<unk>"}
    regions = {}
    for region_id in range(n_clusters):
        positions = np.flatnonzero(labels == region_id)
        token_counts = {}
        examples = []
        for pos in positions:
            text = str(frame.iloc[pos]["narrative"])
            if len(examples) < example_limit:
                examples.append(text[:350])
            for token_id in tokenizer.encode(text):
                token = id_to_token.get(token_id, "")
                if token and token not in special:
                    token_counts[token] = token_counts.get(token, 0) + 1
        ranked_tokens = sorted(token_counts.items(), key=lambda item: (-item[1], item[0]))
        raw_top_tokens = ranked_tokens[:8]
        top_tokens = [item for item in ranked_tokens if _is_interpretable(item[0])][:8]
        if len(top_tokens) < 3:
            top_tokens = raw_top_tokens
        regions[str(region_id)] = {
            "name": " ".join(token for token, _ in top_tokens[:3]),
            "top_tokens": [{"token": token, "count": count}
                           for token, count in top_tokens],
            "raw_top_tokens": [{"token": token, "count": count}
                               for token, count in raw_top_tokens],
            "share": float(len(positions) / len(frame)),
            "n_records": int(len(positions)),
            "examples": examples,
            "centroid": clusterer.cluster_centers_[region_id].astype(float).tolist(),
        }
    return {"n_regions": n_clusters, "n_train": len(frame), "regions": regions}


def load_model(checkpoint_path, tokenizer_path, device="cpu"):
    from cloudlm.tokenizer import BPETokenizer

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ComplaintCloud(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    return model, BPETokenizer.load(tokenizer_path), checkpoint


def cloud_states(model, tokenizer, narratives, max_len, batch_size=32, device="cpu"):
    states = []
    for start in range(0, len(narratives), batch_size):
        rows = [tokenizer.encode(str(text))[:max_len] or [2]
                for text in narratives[start:start + batch_size]]
        width = max(map(len, rows))
        tokens = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            tokens[index, :len(row)] = torch.tensor(row, dtype=torch.long)
        mask = tokens.ne(0)
        with torch.no_grad():
            state, _ = model(tokens.to(device), mask.to(device))[:2]
        states.append(state.cpu().numpy())
    return np.concatenate(states, axis=0) if states else np.empty((0, 0), np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=None,
                        help="sample size; defaults to the checkpoint's training sample")
    parser.add_argument("--checkpoint", default=os.path.join(RESULTS_DIR, "reader.pt"))
    parser.add_argument("--tokenizer", default=os.path.join(RESULTS_DIR, "tokenizer.json"))
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "region_map.json"))
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    model, tokenizer, checkpoint = load_model(args.checkpoint, args.tokenizer)
    n_sample = args.n_sample or checkpoint["config"]["n_sample"]
    frame = load_cfpb(n_sample=n_sample, seed=checkpoint["config"]["seed"],
                      data_dir=args.data_dir)
    train = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    states = cloud_states(model, tokenizer, train["narrative"].tolist(),
                          checkpoint["config"]["max_len"])
    result = make_region_map(train, states, tokenizer, n_regions=N_REGIONS,
                             seed=checkpoint["config"]["seed"])
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps({"output": args.output, "n_regions": result["n_regions"],
                      "names": {key: value["name"]
                                for key, value in result["regions"].items()}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
