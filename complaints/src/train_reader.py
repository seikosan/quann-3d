"""Train the light issue-supervision proxy and save its portable artifacts."""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset

from config import (
    BATCH_SIZE, BPE_VOCAB_SIZE, D, D_CLOUD, D_MODEL, EPOCHS, HIDDEN,
    LEARNING_RATE, LOGS_DIR, MAX_LEN, N_SAMPLE, RESULTS_DIR, SEED,
)
from datasets_cfpb import load_cfpb
from textcloud import ComplaintCloud


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_rows(tokenizer, narratives, max_len):
    rows = [tokenizer.encode(text)[:max_len] or [2] for text in narratives]
    return pad_sequence(
        [torch.tensor(row, dtype=torch.long) for row in rows],
        batch_first=True, padding_value=0,
    )


def run(n_sample=N_SAMPLE, epochs=EPOCHS, max_len=MAX_LEN, batch_size=BATCH_SIZE,
        data_dir=None, output_dir=RESULTS_DIR, device=None):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    device = device or choose_device()
    frame = load_cfpb(n_sample=n_sample, seed=SEED, data_dir=data_dir)
    frame = frame.dropna(subset=["issue"]).copy()
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    drift_frame = frame.loc[frame["split"] == "drift"].reset_index(drop=True)
    if len(train_frame) < 2:
        raise ValueError("At least two labeled train-window complaints are required")

    # Keep a chronological validation slice inside the training window.
    validation_at = max(1, int(len(train_frame) * 0.85))
    fit_frame = train_frame.iloc[:validation_at].copy()
    val_frame = train_frame.iloc[validation_at:].copy()
    if val_frame.empty:
        val_frame = fit_frame.tail(1).copy()

    from cloudlm.tokenizer import BPETokenizer
    tokenizer = BPETokenizer(vocab_size=BPE_VOCAB_SIZE).train(
        fit_frame["narrative"].astype(str).tolist(), vocab_size=BPE_VOCAB_SIZE,
    )
    all_texts = pd_text(frame)
    token_tensor = encode_rows(tokenizer, all_texts, max_len)
    row_index = {idx: pos for pos, idx in enumerate(frame.index)}
    frame_indices = frame.index.to_list()
    train_indices = [row_index[i] for i in fit_frame.index]
    val_indices = [row_index[i] for i in val_frame.index]
    labels_sorted = sorted(fit_frame["issue"].astype(str).unique())
    issue_vocab = {label: i for i, label in enumerate(labels_sorted)}
    # Rare validation-only labels are excluded from the validation metric.
    label_ids = frame["issue"].astype(str).map(issue_vocab).fillna(-1).astype("int64").to_numpy()
    mask_tensor = token_tensor.ne(0)

    def loader(indices, shuffle):
        rows = torch.tensor(indices, dtype=torch.long)
        usable = rows[label_ids[indices] >= 0]
        return DataLoader(
            TensorDataset(token_tensor[usable], mask_tensor[usable],
                          torch.tensor(label_ids[usable], dtype=torch.long)),
            batch_size=batch_size, shuffle=shuffle,
        )

    train_loader = loader(train_indices, True)
    val_loader = loader(val_indices, False)
    if len(train_loader.dataset) == 0:
        raise ValueError("No train rows match the issue vocabulary")

    model = ComplaintCloud(
        vocab_size=tokenizer.num_tokens, d_model=D_MODEL, d_cloud_dim=D,
        d_cloud=D_CLOUD, hidden=HIDDEN, d_z=D, n_issue=len(issue_vocab),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    history = []
    for epoch in range(1, epochs + 1):
        started = time.time()
        model.train()
        losses = []
        for tokens, mask, labels in train_loader:
            tokens, mask, labels = tokens.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            _, _, logits = model(tokens, mask)
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        model.eval()
        actual, predicted = [], []
        with torch.no_grad():
            for tokens, mask, labels in val_loader:
                _, _, logits = model(tokens.to(device), mask.to(device))
                actual.extend(labels.tolist())
                predicted.extend(logits.argmax(-1).cpu().tolist())
        metrics = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)) if losses else None,
            "issue_accuracy": float(accuracy_score(actual, predicted)) if actual else None,
            "issue_balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ) if actual else None,
            "validation_n": len(actual),
            "seconds": round(time.time() - started, 2),
        }
        history.append(metrics)
        print(json.dumps(metrics, ensure_ascii=False), flush=True)

    model = model.to("cpu").eval()
    tok_path = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tok_path)
    checkpoint_path = os.path.join(output_dir, "reader.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "model_config": {
            "vocab_size": tokenizer.num_tokens, "d_model": D_MODEL,
            "d_cloud_dim": D, "d_cloud": D_CLOUD, "hidden": HIDDEN,
            "d_z": D, "n_issue": len(issue_vocab),
        },
        "issue_vocab": issue_vocab,
        "history": history,
        "data_counts": {"sampled": len(frame), "train": len(train_frame),
                        "drift": len(drift_frame), "fit": len(fit_frame),
                        "validation": len(val_frame)},
        "config": {"n_sample": n_sample, "seed": SEED, "max_len": max_len,
                   "epochs": epochs, "batch_size": batch_size,
                   "proxy_target": "issue", "drift_supervision": False},
    }, checkpoint_path)
    print(json.dumps({
        "checkpoint": checkpoint_path,
        "tokenizer": tok_path,
        "device": str(device),
        "counts": {"sampled": len(frame), "train": len(train_frame),
                   "drift": len(drift_frame), "fit": len(fit_frame),
                   "validation": len(val_frame)},
    }, ensure_ascii=False), flush=True)
    return model, tokenizer, issue_vocab, frame


def pd_text(frame):
    return frame["narrative"].astype(str).tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sample", type=int, default=N_SAMPLE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--max-len", type=int, default=MAX_LEN)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()
    run(args.n_sample, args.epochs, args.max_len, args.batch_size, args.data_dir)


if __name__ == "__main__":
    main()
