"""
Quann-3D — Kapsamlı LLM Karşılaştırma ve Edge Benchmark
=========================================================

Kalite metrikleri:
  - Held-out PPL (her model için ayrı)
  - Generation quality (exact match, word overlap)
  - Sentence completion accuracy
  - Length-dependent quality analysis
  - Grammatical validity proxy (EOS rate, vocab coverage)

Edge verimlilik:
  - Inference latency (ms/token, Apple Silicon native)
  - Memory footprint (params + activations + KV cache)
  - FLOPS estimate (per token, per sentence)
  - Energy proxy (Joules)
  - Batch vs streaming comparison

Kullanım:
  python3 compare_llm.py --step all
  python3 compare_llm.py --step quality
  python3 compare_llm.py --step edge
  python3 compare_llm.py --step full_report \
      --cloud-ckpt checkpoints/wiki_checkpoints/wiki_cloud_best.pt \
      --base-ckpt checkpoints/baseline_200k/baseline_best.pt --arch causal
"""

import torch
import torch.nn.functional as F
import math
import time
import random
import json
import argparse
import os

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys
import numpy as np
from typing import Dict

# ─────────────────────────────────────────────
# IMPORTS FROM EXISTING CODE
# ─────────────────────────────────────────────

# We import the models directly via importlib to avoid circular deps
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from train_wiki import BPETokenizer, CloudLM
from train_baseline import BidirNoCloud, build_model, count_params


# ─────────────────────────────────────────────
# UTILITY
# ─────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def load_cloud_ckpt(path: str):
    """Load CloudLM checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CloudLM(ckpt["vocab_size"], **{k: v for k, v in ckpt["config"].items()})
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def load_baseline_ckpt(path: str):
    """Load baseline checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]["arch"], ckpt["vocab_size"],
                        ckpt["config"]["d_model"], ckpt["config"]["n_layers"])
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def load_model(ckpt_path: str):
    """Auto-detect and load model from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "config" in ckpt and "d_cloud_dim" in ckpt.get("config", {}):
        return load_cloud_ckpt(ckpt_path)
    else:
        return load_baseline_ckpt(ckpt_path)


def load_bpe(bpe_path: str):
    return BPETokenizer.load(bpe_path)


def load_corpus(corpus_path: str, max_texts=None):
    texts = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    if max_texts:
        texts = texts[:max_texts]
    return texts


# ─────────────────────────────────────────────
# QUALITY METRICS
# ─────────────────────────────────────────────

def evaluate_ppl(model, tok, texts, device, seq_len, arch, n=2000, seed=42):
    """Held-out PPL evaluation."""
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()
    total_loss, total_tokens = 0.0, 0
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    with torch.no_grad():
        for text in sample:
            ids = tok.encode(text)
            if len(ids) < seq_len + 1:
                continue
            for i in range(0, len(ids) - seq_len, seq_len):
                inp = torch.tensor([ids[i:i+seq_len]], device=device)
                mask = (inp != 0) & (inp != eos_id)

                if arch == "cloud":
                    logits, _, _ = model(inp, mask)
                    tgt1 = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits, tgt1, reduction="sum").item()
                    total_tokens += 1
                elif arch == "causal":
                    logits = model(inp, mask)  # (1, T, V)
                    # Eğitimle tutarlı: son pozisyon → sonraki tek token
                    tgt1 = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits[0, -1, :], tgt1[0].unsqueeze(0),
                                                  reduction="sum").item()
                    total_tokens += 1
                else:  # bidir
                    logits = model(inp, mask)
                    tgt1 = torch.tensor([ids[i+seq_len]], device=device)
                    total_loss += F.cross_entropy(logits, tgt1, reduction="sum").item()
                    total_tokens += 1

    ppl = math.exp(total_loss / max(total_tokens, 1))
    return ppl, total_loss, total_tokens


def evaluate_generation_quality(model, tok, texts, device, seq_len, arch, n=500, seed=42):
    """
    Generation quality: given a prefix, how close is the generated suffix to the true suffix?
    Metrics: exact match, word overlap (Jaccard), word-level F1.
    """
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()
    results = {"exact_match": 0, "total": 0,
               "word_jaccard_sum": 0.0, "word_f1_sum": 0.0,
               "valid_eos": 0, "avg_gen_len": 0, "avg_token_prob_sum": 0.0}
    token_prob_count = 0

    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    with torch.no_grad():
        for text in sample:
            ids = tok.encode(text)
            if len(ids) < seq_len + 3:  # need at least 3 tokens after prefix
                continue

            prefix = ids[:seq_len]
            true_suffix = ids[seq_len:seq_len + 5]  # predict next 5 tokens
            true_words = tok.decode(true_suffix)

            inp = torch.tensor([prefix], device=device)
            mask = (inp != 0) & (inp != eos_id)

            # Generate next 5 tokens
            if arch == "cloud":
                generated_ids = []
                cloud_inp = inp
                for step in range(5):
                    logits, p_next, _ = model(cloud_inp, mask)
                    probs = F.softmax(logits / 0.7, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).item()
                    generated_ids.append(pred_id)
                    # Update for next step
                    if pred_id == eos_id:
                        break
                    new_token = torch.tensor([[pred_id]], device=device)
                    if cloud_inp.size(1) == 1:
                        cloud_inp = new_token
                    else:
                        cloud_inp = torch.cat([cloud_inp, new_token], dim=1)
                    mask = (cloud_inp != 0) & (cloud_inp != eos_id)

                gen_words = tok.decode(generated_ids)

            else:  # causal / bidir
                generated_ids = []
                gen_tokens = inp.clone()
                for step in range(5):
                    if arch == "causal":
                        logits = model(gen_tokens, mask)  # (B, T, V)
                        next_logits = logits[:, -1, :]
                    else:
                        logits = model(gen_tokens, mask)
                        next_logits = logits

                    probs = F.softmax(next_logits / 0.7, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).item()
                    generated_ids.append(pred_id)

                    if pred_id == eos_id:
                        break

                    new_token = torch.tensor([[pred_id]], device=device)
                    gen_tokens = torch.cat([gen_tokens, new_token], dim=1)
                    mask = (gen_tokens != 0) & (gen_tokens != eos_id)

                gen_words = tok.decode(generated_ids)

            results["total"] += 1
            results["avg_gen_len"] += len(generated_ids)

            # Exact match of suffix
            # true_suffix is a list (from tok.encode -> list of ints)
            if generated_ids[:len(true_suffix)] == true_suffix:
                results["exact_match"] += 1

            # EOS rate
            if eos_id in generated_ids:
                results["valid_eos"] += 1

            # Word-level Jaccard similarity
            true_set = set(tok.decode(true_suffix).split())
            gen_set = set(gen_words.split())
            if true_set and gen_set:
                intersection = true_set & gen_set
                union = true_set | gen_set
                jaccard = len(intersection) / len(union)
                results["word_jaccard_sum"] += jaccard

                # Word-level F1
                precision = len(intersection) / len(gen_set)
                recall = len(intersection) / len(true_set)
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    results["word_f1_sum"] += f1

            # Token-level probability of correct tokens
            # (EOS'ta erken biten üretilmiş diziler için adım sınırı: len(generated_ids))
            if arch == "cloud":
                cloud_inp = inp
                mask = (cloud_inp != 0) & (cloud_inp != eos_id)  # reset mask
                for step in range(min(5, len(true_suffix), len(generated_ids))):
                    logits, _, _ = model(cloud_inp, mask)
                    probs = F.softmax(logits / 0.7, dim=-1)
                    true_token = true_suffix[step]  # already an int
                    results["avg_token_prob_sum"] += probs[0, true_token].item()
                    token_prob_count += 1
                    new_token = torch.tensor([[generated_ids[step]]], device=device)
                    if cloud_inp.size(1) == 1:
                        cloud_inp = new_token
                    else:
                        cloud_inp = torch.cat([cloud_inp, new_token], dim=1)
                    mask = (cloud_inp != 0) & (cloud_inp != eos_id)
            else:
                gen_tokens = inp.clone()
                mask = (gen_tokens != 0) & (gen_tokens != eos_id)  # reset mask
                for step in range(min(5, len(true_suffix), len(generated_ids))):
                    if arch == "causal":
                        logits = model(gen_tokens, mask)  # (1, T, V)
                        last_logits = logits[0, -1, :]
                    else:
                        logits = model(gen_tokens, mask)
                        last_logits = logits[0]
                    probs = F.softmax(last_logits / 0.7, dim=-1)
                    true_token = true_suffix[step]  # already an int
                    results["avg_token_prob_sum"] += probs[true_token].item()
                    token_prob_count += 1
                    new_token = torch.tensor([[generated_ids[step]]], device=device)
                    gen_tokens = torch.cat([gen_tokens, new_token], dim=1)
                    mask = (gen_tokens != 0) & (gen_tokens != eos_id)

    results["avg_gen_len"] /= max(results["total"], 1)
    results["word_jaccard_sum"] /= max(results["total"], 1)
    results["word_f1_sum"] /= max(results["total"], 1)
    results["exact_match"] = results["exact_match"] / max(results["total"], 1)
    results["valid_eos"] = results["valid_eos"] / max(results["total"], 1)
    results["avg_token_prob_sum"] /= max(token_prob_count, 1)

    return results


def evaluate_factual_completion(model, tok, prompts, device, max_new=10, seed=42):
    """
    Test model on known-fact completion prompts.
    Returns list of (prompt, completion, expected_keyword, hit).
    """
    model.eval()
    results = []

    with torch.no_grad():
        for prompt in prompts:
            text = prompt["text"]
            expected = prompt.get("expected_keyword", "")

            ids = torch.tensor([tok.encode(text)], device=device)
            generated_ids = []
            eos_id = tok._tokenizer.token_to_id("<eos>") or 1

            # Use the generate method from baseline
            if hasattr(model, 'generate'):
                gen_out = model.generate(ids, max_new=max_new, temperature=0.7, eos_id=eos_id)
                generated_ids = gen_out[0].tolist()
            else:
                # CloudLM: custom generation
                generated_ids = []
                cloud_inp = ids
                mask = (cloud_inp != 0) & (cloud_inp != eos_id)
                for step in range(max_new):
                    logits, _, _ = model(cloud_inp, mask)
                    probs = F.softmax(logits / 0.7, dim=-1)
                    pred_id = torch.argmax(probs, dim=-1).item()
                    if pred_id == eos_id:
                        break
                    generated_ids.append(pred_id)
                    new_token = torch.tensor([[pred_id]], device=device)
                    if cloud_inp.size(1) == 1:
                        cloud_inp = new_token
                    else:
                        cloud_inp = torch.cat([cloud_inp, new_token], dim=1)
                    mask = (cloud_inp != 0) & (cloud_inp != eos_id)

            completion = tok.decode(generated_ids)

            # Check keyword hit
            completion_lower = completion.lower()
            hit = expected.lower() in completion_lower or \
                  any(w.lower() in completion_lower for w in expected.split())

            results.append({
                "prompt": text,
                "completion": completion,
                "expected": expected,
                "hit": hit
            })

    hit_rate = sum(1 for r in results if r["hit"]) / max(len(results), 1)
    return results, hit_rate


def length_dependent_quality(model, tok, texts, device, seq_len, arch,
                              n=1000, seed=42):
    """
    Analyze quality as a function of input sentence length.
    Returns dict: {length_bucket: PPL}
    """
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()

    # Bucket by input token length
    buckets = {}
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    with torch.no_grad():
        for text in sample:
            ids = tok.encode(text)
            if len(ids) < seq_len + 1:
                continue

            bucket_size = len(ids)  # total sentence tokens
            if bucket_size not in buckets:
                buckets[bucket_size] = {"loss": 0.0, "tokens": 0}

            for i in range(0, len(ids) - seq_len, seq_len):
                inp = torch.tensor([ids[i:i+seq_len]], device=device)
                mask = (inp != 0) & (inp != eos_id)

                if arch == "cloud":
                    logits, _, _ = model(inp, mask)
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    loss = F.cross_entropy(logits, tgt).item()
                    buckets[bucket_size]["loss"] += loss
                    buckets[bucket_size]["tokens"] += 1

                elif arch == "causal":
                    logits = model(inp, mask)  # (1, T, V)
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    loss = F.cross_entropy(logits[0, -1, :], tgt[0].unsqueeze(0)).item()
                    buckets[bucket_size]["loss"] += loss
                    buckets[bucket_size]["tokens"] += 1

                else:
                    logits = model(inp, mask)
                    tgt = torch.tensor([ids[i+seq_len]], device=device)
                    loss = F.cross_entropy(logits, tgt).item()
                    buckets[bucket_size]["loss"] += loss
                    buckets[bucket_size]["tokens"] += 1

    # Compute PPL per bucket
    bucket_ppl = {}
    for b, data in buckets.items():
        ppl = math.exp(data["loss"] / max(data["tokens"], 1))
        bucket_ppl[b] = ppl

    return bucket_ppl


# ─────────────────────────────────────────────
# EDGE EFFICIENCY METRICS
# ─────────────────────────────────────────────

def measure_inference_latency(model, tok, texts, device, seq_len, arch,
                               n=100, seed=42):
    """
    Measure inference latency in ms/token for both models.
    Uses timeit for stable measurement.
    """
    rng = random.Random(seed)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()

    eos_id = tok._tokenizer.token_to_id("<eos>") or 1
    latencies = []  # list of (tokens_generated, time_ms) pairs

    for text in sample:
        ids = tok.encode(text)
        if len(ids) < seq_len + 1:
            continue

        inp = torch.tensor([ids[:seq_len]], device=device)
        mask = (inp != 0) & (inp != eos_id)

        if arch == "cloud":
            # CloudLM: generate until EOS or max 20 tokens
            start = time.perf_counter()
            gen_tokens = inp.clone()
            for step in range(20):
                logits, p_next, _ = model(gen_tokens, mask)
                probs = F.softmax(logits / 0.8, dim=-1)
                pred_id = torch.multinomial(probs, 1).item()
                if pred_id == eos_id:
                    break
                new_token = torch.tensor([[pred_id]], device=device)
                if gen_tokens.size(1) == 1:
                    gen_tokens = new_token
                else:
                    gen_tokens = torch.cat([gen_tokens, new_token], dim=1)
                mask = (gen_tokens != 0) & (gen_tokens != eos_id)
            end = time.perf_counter()

            gen_count = gen_tokens.size(1)
            time_ms = (end - start) * 1000

            if time_ms > 0:  # valid run
                latencies.append((gen_count, time_ms))

        else:
            start = time.perf_counter()
            gen_tokens = model.generate(inp, max_new=15, temperature=0.8, eos_id=eos_id)
            end = time.perf_counter()

            time_ms = (end - start) * 1000
            gen_count = gen_tokens.size(1)

            if time_ms > 0:
                latencies.append((gen_count, time_ms))

    # Compute stats
    if not latencies:
        return {"error": "No valid runs", "ms_per_token": float('inf')}

    all_ms_per_token = [ms / max(cnt, 1) for cnt, ms in latencies]
    all_total_ms = [ms for _, ms in latencies]
    all_gen_counts = [cnt for cnt, _ in latencies]

    return {
        "ms_per_token": np.mean(all_ms_per_token),
        "ms_per_token_std": np.std(all_ms_per_token),
        "median_ms_per_token": np.median(all_ms_per_token),
        "total_avg_ms": np.mean(all_total_ms),
        "total_std_ms": np.std(all_total_ms),
        "avg_gen_tokens": np.mean(all_gen_counts),
        "samples": len(latencies),
        "ms_per_token_sorted": sorted(all_ms_per_token),
        "p95_ms_per_token": np.percentile(all_ms_per_token, 95),
        "p99_ms_per_token": np.percentile(all_ms_per_token, 99),
    }


def measure_memory_footprint(model, device, arch, seq_len=10):
    """
    Measure memory footprint: parameters + activations + KV cache.
    Returns bytes breakdown.
    """
    n_params = count_params(model)

    # Parameter memory (assuming float32)
    param_bytes = n_params * 4  # float32

    # Activation memory estimate
    # Context encoder: L layers, d_model, seq_len
    d_model = getattr(model, 'd_model', 256)
    n_layers = getattr(model, 'encoder', None)
    n_layers = n_layers.num_layers if hasattr(n_layers, 'num_layers') else 4

    # Per-token activation: embedding + encoder output + intermediate
    # Rough estimate: 2 * d_model * seq_len * n_layers (forward + backward)
    activation_bytes = 2 * d_model * seq_len * n_layers * 4  # float32

    # KV cache (causal only, grows with sequence)
    if arch == "causal":
        # Each layer: KV = 2 * d_model * max_seq_len
        max_seq = 256
        kv_bytes = 2 * d_model * max_seq * n_layers * 4
    else:
        kv_bytes = 0

    # Cloud-specific: point cloud + pair features
    cloud_bytes = 0
    if hasattr(model, 'd_cloud_dim') and model.d_cloud_dim:
        D = model.d_cloud_dim
        N = seq_len
        # Points: N * D * 4 (float32)
        cloud_bytes += N * D * 4
        # Pair features: N*N * (D+4) * 4 (float32) — only during inference we recompute
        cloud_bytes += N * N * (D + 4) * 4

    total_bytes = param_bytes + activation_bytes + kv_bytes + cloud_bytes

    return {
        "n_params": n_params,
        "param_bytes": param_bytes,
        "param_mb": param_bytes / (1024 * 1024),
        "activation_bytes": activation_bytes,
        "activation_mb": activation_bytes / (1024 * 1024),
        "kv_cache_bytes": kv_bytes,
        "kv_cache_mb": kv_bytes / (1024 * 1024),
        "cloud_bytes": cloud_bytes,
        "cloud_mb": cloud_bytes / (1024 * 1024),
        "total_bytes": total_bytes,
        "total_mb": total_bytes / (1024 * 1024),
        "arch": arch,
    }


def estimate_flops(model, arch, seq_len=10, vocab_size=None):
    """
    Estimate FLOPS per token and per sentence.
    Based on transformer arithmetic complexity.
    """
    n_params = count_params(model)

    if arch == "cloud":
        # Context encoder: 2/3 * params (forward) + 4/3 * params (backward for training)
        # For inference: just forward
        context_flops = 2 * n_params * 0.5  # ~half params for inference forward
        # ND projection: trivial (MLP)
        nd_flops = seq_len * 3 * 256 * 256  # point MLP: 2 MLP layers
        # Cloud encoder: point MLP + pair MLP (N^2 pairs)
        D = getattr(model, 'd_cloud_dim', 16)
        d_cloud = 512
        point_flops = seq_len * (D * 256 + 256 * 256 + 256 * d_cloud)
        pair_flops = seq_len * seq_len * ((D + 4) * 256 + 256 * 256 + 256 * d_cloud)
        # Prediction head
        pred_flops = d_cloud * 256 + 256 * (vocab_size or 20000)
        total_flops = context_flops + nd_flops + point_flops + pair_flops + pred_flops
    else:
        # Standard transformer: 6 * params * seq_len (forward)
        total_flops = 6 * n_params * seq_len

    return {
        "flops_per_token": total_flops / max(seq_len, 1),
        "flops_per_sentence": total_flops,
        "flops_tf": total_flops / 1e12,
    }


def estimate_energy(model, arch, seq_len=10, device_str="mps"):
    """
    Estimate energy consumption in Joules.
    Uses rough PUE and device-specific power draw.
    """
    flops = estimate_flops(model, arch, seq_len)

    # Device-specific power efficiency (FLOPS/Watt)
    # Apple M-series: ~10-20 TFLOPS/W (performance/Watt)
    # NVIDIA A100: ~25 TFLOPS/W (mixed precision)
    # CPU: ~0.5 TFLOPS/W

    if device_str == "mps":
        flops_per_watt = 15e12  # ~15 TFLOPS/W for Apple Silicon
    elif device_str.startswith("cuda"):
        flops_per_watt = 25e12  # A100 mixed precision
    else:
        flops_per_watt = 0.5e12  # CPU

    watts = flops["flops_per_sentence"] / flops_per_watt
    joules = watts  # 1 second at 1 Watt = 1 Joule

    # PUE (Power Usage Effectiveness) for data center
    pue = 1.5  # typical modern data center

    return {
        "joules": joules,
        "joules_pue": joules * pue,
        "flops_per_watt": flops_per_watt,
        "device": device_str,
    }


def batch_vs_streaming(model, tok, texts, device, seq_len, arch, n=50):
    """
    Compare batch inference vs streaming (token-by-token) latency.
    """
    rng = random.Random(42)
    sample = rng.sample(texts, min(n, len(texts)))
    model.eval()

    # Batch: process all sentences at once (single forward pass)
    eos_id = tok._tokenizer.token_to_id("<eos>") or 1

    batch_text = sample[:n]
    batch_tokens = [tok.encode(t) for t in batch_text]
    # Pad to max length (her satırın ilk seq_len token'ı)
    max_len = max(len(t) for t in batch_tokens)
    padded = torch.zeros(len(batch_tokens), max_len, dtype=torch.long, device=device)
    for i, t in enumerate(batch_tokens):
        n_tok = min(len(t), seq_len)
        padded[i, :n_tok] = torch.tensor(t[:n_tok])
    mask = (padded != 0) & (padded != eos_id)

    start = time.perf_counter()
    with torch.no_grad():
        if arch == "cloud":
            logits, _, _ = model(padded, mask)
        elif arch == "causal":
            logits = model(padded, mask)  # (B, T, V)
        else:
            logits = model(padded, mask)
    batch_time = (time.perf_counter() - start) * 1000

    # Streaming: process each sentence individually
    stream_total = 0.0
    for text in batch_text:
        ids = tok.encode(text)
        if len(ids) < seq_len + 1:
            continue
        inp = torch.tensor([ids[:seq_len]], device=device)
        mask = (inp != 0) & (inp != eos_id)
        start = time.perf_counter()
        with torch.no_grad():
            if arch == "cloud":
                logits, _, _ = model(inp, mask)
            elif arch == "causal":
                logits = model(inp, mask)  # (1, T, V)
            else:
                logits = model(inp, mask)
        stream_total += (time.perf_counter() - start) * 1000

    return {
        "batch_time_ms": batch_time,
        "stream_total_ms": stream_total,
        "speedup": stream_total / max(batch_time, 1),
        "batch_size": len(batch_text),
    }


# ─────────────────────────────────────────────
# REPORTING
# ─────────────────────────────────────────────

def print_quality_report(name, metrics, arch, n_params):
    """Print a formatted quality report."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — QUALITY REPORT")
    print(f"{'='*60}")
    print(f"  Model: {name}")
    print(f"  Architecture: {arch}")
    print(f"  Parameters: {n_params:,}")

    if "ppl" in metrics:
        print(f"  PPL: {metrics['ppl']:.2f}")
    if "total_loss" in metrics:
        print(f"  Total loss: {metrics['total_loss']:.4f}")
    if "total_tokens" in metrics:
        print(f"  Evaluated tokens: {metrics['total_tokens']:,}")
    if "exact_match" in metrics:
        print(f"  Exact suffix match: {metrics['exact_match']:.3%}")
    if "word_jaccard_sum" in metrics:
        print(f"  Word Jaccard similarity: {metrics['word_jaccard_sum']:.3f}")
    if "word_f1_sum" in metrics:
        print(f"  Word F1 score: {metrics['word_f1_sum']:.3f}")
    if "valid_eos" in metrics:
        print(f"  Valid EOS rate: {metrics['valid_eos']:.3%}")
    if "avg_token_prob_sum" in metrics:
        print(f"  Avg correct-token prob: {metrics['avg_token_prob_sum']:.4f}")
    if "factual_hit_rate" in metrics:
        print(f"  Factual hit rate: {metrics['factual_hit_rate']:.3%}")


def print_edge_report(name, metrics, arch, device_str="MPS"):
    """Print a formatted edge efficiency report."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — EDGE EFFICIENCY")
    print(f"{'='*60}")
    print(f"  Model: {name}")
    print(f"  Architecture: {arch}")
    print(f"  Device: {device_str}")

    print(f"\n  {'Metric':30s} {'Value':>20s}")
    print(f"  {'-'*50}")

    if "n_params" in metrics:
        print(f"  {'Parameters':30s} {metrics['n_params']:>20,}")
        print(f"  {'Param memory (float32)':30s} {metrics['param_mb']:>19.2f} MB")
        print(f"  {'Activation memory':30s} {metrics['activation_mb']:>19.2f} MB")
        print(f"  {'KV cache memory':30s} {metrics['kv_cache_mb']:>19.2f} MB")
        print(f"  {'Cloud representation':30s} {metrics['cloud_mb']:>19.2f} MB")
        print(f"  {'Total memory':30s} {metrics['total_mb']:>19.2f} MB")

    if "ms_per_token" in metrics:
        print(f"  {'Latency (ms/token, mean)':30s} {metrics['ms_per_token']:>19.4f}")
        print(f"  {'Latency (ms/token, std)':30s} {metrics['ms_per_token_std']:>19.4f}")
        print(f"  {'Latency (ms/token, median)':30s} {metrics['median_ms_per_token']:>19.4f}")
        print(f"  {'P95 latency (ms/token)':30s} {metrics['p95_ms_per_token']:>19.4f}")
        print(f"  {'P99 latency (ms/token)':30s} {metrics['p99_ms_per_token']:>19.4f}")
        print(f"  {'Avg generation time':30s} {metrics['total_avg_ms']:>19.2f} ms")
        print(f"  {'Avg tokens generated':30s} {metrics['avg_gen_tokens']:>19.1f}")
        print(f"  {'Samples':30s} {metrics['samples']:>19}")

    if "flops_per_sentence" in metrics:
        print(f"  {'FLOPS per sentence':30s} {metrics['flops_tf']:>19.4f} TF")
        print(f"  {'FLOPS per token':30s} {metrics['flops_per_token']:>19.4f}")

    if "joules" in metrics:
        print(f"  {'Energy (Joules)':30s} {metrics['joules']:>19.6f}")
        print(f"  {'Energy with PUE 1.5':30s} {metrics['joules_pue']:>19.6f}")

    if "speedup" in metrics:
        print(f"  {'Batch streaming speedup':30s} {metrics['speedup']:>19.2f}x")
        print(f"  {'Batch time (ms)':30s} {metrics['batch_time_ms']:>19.2f}")
        print(f"  {'Streaming total (ms)':30s} {metrics['stream_total_ms']:>19.2f}")


def print_comparison_summary(quality_results: Dict[str, Dict],
                              edge_results: Dict[str, Dict],
                              arch_info: Dict[str, str]):
    """Print a side-by-side comparison table."""
    print(f"\n{'#'*70}")
    print("  QUANN-3D — LLM KARŞILAŞTIRMA ÖZETİ")
    print(f"  {'#'*66}")

    models = list(quality_results.keys())

    # Quality summary table
    print(f"\n{'='*70}")
    print("  KALİTE KARŞILAŞTIRMASI")
    print(f"{'='*70}")
    header = f"  {'Model':20s}"
    metrics_to_show = ["ppl", "exact_match", "word_jaccard", "word_f1", "valid_eos", "factual_hit_rate"]
    for m in metrics_to_show:
        header += f" {m:>16s}"
    print(header)
    print(f"  {'-'*70}")

    for name in models:
        line = f"  {name:20s}"
        m = quality_results[name]
        line += f" {m.get('ppl', 'N/A'):>16}" if isinstance(m.get('ppl'), (int, float)) else f" {m.get('ppl', 'N/A'):>16}"
        line += f" {m.get('exact_match', 0):>16.3%}"
        line += f" {m.get('word_jaccard', 0):>16.3f}"
        line += f" {m.get('word_f1', 0):>16.3f}"
        line += f" {m.get('valid_eos', 0):>16.3%}"
        line += f" {m.get('factual_hit_rate', 0):>16.3%}"
        print(line)

    # Edge summary table
    print(f"\n{'='*70}")
    print("  EDGE VERİMLİLİK KARŞILAŞTIRMASI")
    print(f"{'='*70}")
    header = f"  {'Model':20s}"
    edge_metrics = ["param_mb", "total_mb", "ms_per_token", "flops_tf", "joules", "speedup"]
    for m in edge_metrics:
        header += f" {m:>16s}"
    print(header)
    print(f"  {'-'*70}")

    for name in models:
        line = f"  {name:20s}"
        m = edge_results[name]
        line += f" {m.get('param_mb', 'N/A'):>16}" if isinstance(m.get('param_mb'), (int, float)) else f" {m.get('param_mb', 'N/A'):>16}"
        line += f" {m.get('total_mb', 'N/A'):>16}" if isinstance(m.get('total_mb'), (int, float)) else f" {m.get('total_mb', 'N/A'):>16}"
        line += f" {m.get('ms_per_token', 'N/A'):>16}" if isinstance(m.get('ms_per_token'), (int, float)) else f" {m.get('ms_per_token', 'N/A'):>16}"
        line += f" {m.get('flops_tf', 'N/A'):>16}" if isinstance(m.get('flops_tf'), (int, float)) else f" {m.get('flops_tf', 'N/A'):>16}"
        line += f" {m.get('joules', 'N/A'):>16}" if isinstance(m.get('joules'), (int, float)) else f" {m.get('joules', 'N/A'):>16}"
        line += f" {m.get('speedup', 'N/A'):>16}" if isinstance(m.get('speedup'), (int, float)) else f" {m.get('speedup', 'N/A'):>16}"
        print(line)

    # Quality-per-parameter analysis
    print(f"\n{'='*70}")
    print("  KALİTE / PARAMETRE VERİMLİLİĞİ")
    print(f"{'='*70}")

    for name in models:
        ppl = quality_results[name].get("ppl", float('inf'))
        n_params = edge_results[name].get("n_params", 0)
        if isinstance(ppl, (int, float)) and ppl > 0 and n_params > 0:
            eff = n_params / (ppl - 1)  # higher is better (more params per PPL-excess-1)
            print(f"  {name:20s} PPL={ppl:.2f} | {n_params:>10,} params | PPL-bağımlı verimlilik: {eff:>12,.0f}")

    # Edge efficiency summary
    print(f"\n{'='*70}")
    print("  EDGE KARARNAME")
    print(f"{'='*70}")

    best_latency_model = None
    best_latency = float('inf')
    best_memory_model = None
    best_memory = float('inf')

    for name in models:
        ms_pt = edge_results[name].get("ms_per_token", float('inf'))
        total_mb = edge_results[name].get("total_mb", float('inf'))
        if isinstance(ms_pt, (int, float)) and ms_pt < best_latency:
            best_latency = ms_pt
            best_latency_model = name
        if isinstance(total_mb, (int, float)) and total_mb < best_memory:
            best_memory = total_mb
            best_memory_model = name

    if best_latency_model:
        print(f"  En hızlı inference: {best_latency_model} ({best_latency:.4f} ms/token)")
    if best_memory_model:
        print(f"  En düşük bellek: {best_memory_model} ({best_memory:.2f} MB)")

    # Find best quality/performance ratio
    print("\n  Öneri: Hangi model hangi senaryoda öne çıkar —")
    for name in models:
        ppl = quality_results[name].get("ppl", float('inf'))
        ms_pt = edge_results[name].get("ms_per_token", float('inf'))
        total_mb = edge_results[name].get("total_mb", float('inf'))
        print(f"  {name:20s} PPL={ppl:.2f} | {ms_pt:.4f} ms/token | {total_mb:.2f} MB")

    print()


def print_length_quality(bucket_ppl, name):
    """Print length-dependent quality analysis."""
    print(f"\n{'='*60}")
    print(f"  {name.upper()} — UZUNLUĞA BAĞLI KALİTE")
    print(f"{'='*60}")
    print(f"  {'Sentence Length (tokens)':25s} {'PPL':>10s}")
    print(f"  {'-'*35}")
    for length in sorted(bucket_ppl.keys()):
        ppl = bucket_ppl[length]
        print(f"  {'~' + str(length) + ' tokens':25s} {ppl:>10.2f}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_quality(args, tok, model, device, seq_len, arch, name, corpus_path=None):
    """Run all quality metrics for a single model."""
    texts = load_corpus(corpus_path or args.corpus, args.max_texts)

    # PPL
    ppl, total_loss, total_tokens = evaluate_ppl(
        model, tok, texts, device, seq_len, arch, args.eval_n, seed=42)

    # Generation quality
    gen_metrics = evaluate_generation_quality(
        model, tok, texts, device, seq_len, arch, args.gen_n, seed=42)

    # Factual completion (if prompts provided)
    factual_result = {"factual_hit_rate": 0}
    if args.factual_prompts:
        prompts = load_corpus(args.factual_prompts)
        # Parse prompts: format "keyword: expected"
        fact_prompts = []
        for p in prompts:
            if ": " in p:
                kw, text = p.split(": ", 1)
                fact_prompts.append({"text": text, "expected_keyword": kw})
        if fact_prompts:
            _, fact_rate = evaluate_factual_completion(
                model, tok, fact_prompts, device)
            factual_result["factual_hit_rate"] = fact_rate

    metrics = {
        "ppl": ppl, "total_loss": total_loss, "total_tokens": total_tokens,
        "exact_match": gen_metrics["exact_match"],
        "word_jaccard": gen_metrics["word_jaccard_sum"],
        "word_f1": gen_metrics["word_f1_sum"],
        "valid_eos": gen_metrics["valid_eos"],
        "avg_token_prob": gen_metrics["avg_token_prob_sum"],
        "factual_hit_rate": factual_result["factual_hit_rate"],
        "n_params": count_params(model),
        "arch": arch,
    }

    print_quality_report(name, metrics, arch, count_params(model))

    # Length-dependent analysis (optional)
    bucket_ppl = None
    if args.length_analysis:
        bucket_ppl = length_dependent_quality(
            model, tok, texts, device, seq_len, arch, args.length_n, seed=42)
        print_length_quality(bucket_ppl, name)

    return metrics, bucket_ppl


def run_edge(args, model, device, arch, name, seq_len=10, tok=None, corpus_path=None):
    """Run all edge efficiency metrics for a single model."""
    # Latency
    # Load corpus for latency measurement
    if corpus_path and tok:
        corpus_texts = load_corpus(corpus_path, max(args.edge_n * 3, 100))
    else:
        corpus_texts = None
    latency = measure_inference_latency(
        model, tok=tok, texts=corpus_texts, device=device, seq_len=seq_len,
        arch=arch, n=args.edge_n, seed=42)

    # Memory
    memory = measure_memory_footprint(model, device, arch, seq_len)

    # FLOPS
    flops = estimate_flops(model, arch, seq_len)

    # Energy
    device_str = "mps" if device.type == "mps" else device.type
    energy = estimate_energy(model, arch, seq_len, device_str)

    metrics = {
        **latency, **memory, **flops, **energy,
        "n_params": memory["n_params"],
    }

    print_edge_report(name, metrics, arch, device_str)

    return metrics


def run_full(args):
    """Run full comparison between CloudLM and baseline."""
    device = get_device()

    print(f"\n{'#'*70}")
    print("  QUANN-3D — TAM KARŞILAŞTIRMA")
    print(f"  Device: {device}")
    print(f"  {'#'*66}")

    # Load models
    models_info = []

    # CloudLM
    if args.cloud_ckpt:
        cloud, cloud_ckpt = load_cloud_ckpt(args.cloud_ckpt)
        cloud.to(device)
        models_info.append({
            "model": cloud, "name": "CloudLM", "arch": "cloud",
            "ckpt": cloud_ckpt,
        })

    # Baseline — causal
    if args.base_ckpt:
        base, base_ckpt = load_baseline_ckpt(args.base_ckpt)
        base.to(device)
        models_info.append({
            "model": base, "name": f"Baseline({args.arch.upper()})",
            "arch": args.arch,
            "ckpt": base_ckpt,
        })

    # Load tokenizer and corpus FIRST (needed for bidir creation)
    bpe_path = args.bpe or os.path.join(
        os.path.dirname(args.cloud_ckpt) if args.cloud_ckpt
        else os.path.join(ROOT, "checkpoints", "wiki_checkpoints"),
        "wiki_bpe.json"
    )
    tok = load_bpe(bpe_path)

    # Also run bidirectional ablation if requested
    # NOT: Bu model bu script içinde EĞİTİLMEZ — taze (rastgele) ağırlıklarla
    # çalışır ve metrikleri alt-sın (untrained lower bound) olarak okunmalı.
    if args.include_bidir and args.cloud_ckpt:
        print("\n  ⚠ --include-bidir: BidirNoCloud EĞİTİLMEMİŞ (rastgele ağırlık) —")
        print("    sonuçlar alt-sın referansıdır, eğitilmiş ablation DEĞİL.")
        bidir = BidirNoCloud(cloud_ckpt["vocab_size"], d_model=256, n_layers=4)
        bidir.to(device)
        models_info.append({
            "model": bidir, "name": "Bidir(NoCloud, untrained)",
            "arch": "bidir",
        })

    # Run quality for each model
    quality_results = {}
    all_bucket_ppl = {}

    for info in models_info:
        print(f"\n{'#'*60}")
        print(f"  MODEL: {info['name']} ({info['arch']})")
        print(f"  {'#'*56}")

        metrics, bucket_ppl = run_quality(
            args, tok, info["model"], device, args.seq_len,
            info["arch"], info["name"], args.corpus)
        quality_results[info["name"]] = metrics
        if bucket_ppl:
            all_bucket_ppl[info["name"]] = bucket_ppl

    # Run edge for each model
    edge_results = {}
    for info in models_info:
        print(f"\n{'#'*60}")
        print(f"  EDGE TEST: {info['name']}")
        print(f"  {'#'*56}")

        metrics = run_edge(
            args, info["model"], device, info["arch"], info["name"],
            tok=tok, corpus_path=args.corpus)
        edge_results[info["name"]] = metrics

    # Print summary
    print_comparison_summary(quality_results, edge_results, {})

    # Save results to JSON
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        report = {
            "quality": quality_results,
            "edge": edge_results,
            "arch_info": {info["name"]: info["arch"] for info in models_info},
        }
        report_path = os.path.join(args.save_dir, "comparison_report.json")
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\n  📊 Rapor kaydedildi: {report_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Quann-3D — Kapsamlı LLM Karşılaştırma ve Edge Benchmark")

    # Model paths
    ap.add_argument("--step", type=str, default="all",
                    choices=["quality", "edge", "full_report", "all"])
    ap.add_argument("--cloud-ckpt", type=str,
                    default=os.path.join(ROOT, "checkpoints", "wiki_checkpoints", "wiki_cloud_best.pt"),
                    help="CloudLM checkpoint path")
    ap.add_argument("--base-ckpt", type=str, default=None,
                    help="Baseline checkpoint path")
    ap.add_argument("--arch", type=str, default="causal",
                    choices=["causal", "bidir"])

    # Data
    ap.add_argument("--corpus", type=str,
                    default=os.path.join(ROOT, "data", "turkish_sentences.txt"))
    ap.add_argument("--bpe", type=str, default=None,
                    help="BPE json path (default: <cloud-ckpt dizini>/wiki_bpe.json)")
    ap.add_argument("--max-texts", type=int, default=20000,
                    help="Max corpus sentences")
    ap.add_argument("--factual-prompts", type=str, default=None,
                    help="Path to factual prompt file (format: 'keyword: sentence')")

    # Quality settings
    ap.add_argument("--eval-n", type=int, default=5000,
                    help="Number of sentences for PPL evaluation")
    ap.add_argument("--gen-n", type=int, default=500,
                    help="Number of sentences for generation quality")
    ap.add_argument("--length-analysis", action="store_true",
                    help="Run length-dependent quality analysis")
    ap.add_argument("--length-n", type=int, default=2000,
                    help="Sentences for length analysis")

    # Edge settings
    ap.add_argument("--edge-n", type=int, default=100,
                    help="Number of samples for edge benchmark")
    ap.add_argument("--seq-len", type=int, default=10,
                    help="Sequence length for comparisons")
    ap.add_argument("--include-bidir", action="store_true",
                    help="Include bidirectional ablation model")

    # Output
    ap.add_argument("--save-dir", type=str, default="comparison",
                    help="Output directory for reports")
    ap.add_argument("--verbose", action="store_true", help="Verbose output")

    args = ap.parse_args()

    if args.step in ("full_report", "all"):
        run_full(args)
    elif args.step == "quality":
        if not args.cloud_ckpt:
            print("Hata: --cloud-ckpt gerekli")
            sys.exit(1)
        device = get_device()
        cloud, ckpt = load_cloud_ckpt(args.cloud_ckpt)
        cloud.to(device)
        bpe_path = args.bpe or os.path.join(os.path.dirname(args.cloud_ckpt), "wiki_bpe.json")
        tok = load_bpe(bpe_path)
        run_quality(args, tok, cloud, device, args.seq_len, "cloud", "CloudLM", args.corpus)

        if args.base_ckpt:
            base, b_ckpt = load_baseline_ckpt(args.base_ckpt)
            base.to(device)
            run_quality(args, tok, base, device, args.seq_len, args.arch,
                        f"Baseline({args.arch.upper()})", args.corpus)

    elif args.step == "edge":
        if not args.cloud_ckpt:
            print("Hata: --cloud-ckpt gerekli")
            sys.exit(1)
        device = get_device()
        cloud, ckpt = load_cloud_ckpt(args.cloud_ckpt)
        cloud.to(device)
        bpe_path = args.bpe or os.path.join(os.path.dirname(args.cloud_ckpt), "wiki_bpe.json")
        tok = load_bpe(bpe_path)
        run_edge(args, cloud, device, "cloud", "CloudLM", tok=tok, corpus_path=args.corpus)

        if args.base_ckpt:
            base, b_ckpt = load_baseline_ckpt(args.base_ckpt)
            base.to(device)
            run_edge(args, base, device, args.arch, f"Baseline({args.arch.upper()})", tok=tok, corpus_path=args.corpus)


if __name__ == "__main__":
    main()
