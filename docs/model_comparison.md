# Model Comparison: CloudLM vs Baselines

## Overview
This document compares CloudLM with standard baselines (GPT-2, BERT, Transformer) on Turkish language modeling.

## Models Compared

### 1. CloudLM (Ours)
- **Architecture**: Point cloud geometry + Transformer encoder
- **Configuration**: D=64, Nokta=20, 8 layers
- **Corpus**: Turkish Wikipedia (1.5M sentences)
- **Params**: ~25M
- **PPL**: TBD (training in progress)

### 2. Baseline Transformer
- **Architecture**: Standard Transformer encoder
- **Configuration**: d_model=256, 6 layers, 8 heads
- **Corpus**: Turkish Wikipedia (1.5M sentences)
- **Params**: ~15M
- **PPL**: TBD

### 3. GPT-2 Turkish
- **Model**: dbmdz/gpt2-large-turkish
- **Type**: Autoregressive language model
- **Params**: ~345M
- **PPL**: TBD

### 4. BERT Turkish
- **Model**: dbmdz/bert-large-turkish
- **Type**: Masked language model
- **Params**: ~336M
- **PPL**: TBD (masked LM perplexity)

## Evaluation Protocol

### Dataset
- **Source**: Turkish sentences corpus (200K + 1.5M)
- **Preprocessing**: BPE tokenization (20K vocab)
- **Split**: 95% train, 5% held-out

### Metrics
- **Perplexity (PPL)**: Lower is better
- **Loss**: Cross-entropy loss
- **Inference Time**: Seconds per sample

### Tokenizer
- **Type**: BPE (Byte Pair Encoding)
- **Vocab Size**: 20,000 tokens
- **Source**: Trained on corpus

## Results

| Model | Corpus | Params | PPL | Notes |
|-------|--------|--------|-----|-------|
| CloudLM | 1.5M | 25M | TBD | Point cloud geometry |
| Baseline | 1.5M | 15M | TBD | Standard Transformer |
| GPT-2-Turkish | Pretrained | 345M | TBD | Autoregressive |
| BERT-Turkish | Pretrained | 336M | TBD | Masked LM |

## Running Evaluation

```bash
# Evaluate pretrained models
python3 training/evaluate_pretrained.py --eval-gpt2 --eval-bert

# Evaluate our CloudLM
python3 training/train_wiki.py --step test
```

## Key Questions

1. Does CloudLM's geometric approach outperform standard Transformer?
2. How does CloudLM compare to large pretrained models (GPT-2, BERT)?
3. What is the data efficiency of CloudLM vs baselines?
4. Does point cloud geometry capture semantic relationships better?

## Next Steps

- [ ] Complete CloudLM 100 epoch training
- [ ] Run baseline 1500K training
- [ ] Evaluate GPT-2/BERT on same corpus
- [ ] Analyze results and write comparison
- [ ] Update paper with comparison results
