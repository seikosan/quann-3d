# ContextCloudLM

ContextCloudLM is a transformer-free language model for multi-sentence input. It
encodes each sentence with PureCloudLM's O(N) `GlobalGeometryEncoder`, then
encodes overlapping groups of neighboring sentence states as window clouds. A
linear-time geometry stage summarizes each window's distance and angle relative
to the global window centroid; it does not construct all window pairs.

## Usage

```python
import torch
from contextcloud import ContextCloudLM

model = ContextCloudLM(vocab_size=30_000)
tokens = torch.randint(1, 30_000, (2, 8, 64))  # batch, sentences, tokens
token_mask = torch.ones_like(tokens, dtype=torch.bool)
logits = model(tokens, sentence_mask=token_mask)
print(logits.shape)  # (2, 30_000)
```

For one sentence per example, pass `(batch, tokens)` instead. `sentence_mask`
can be a token mask shaped like `tokens`, or a sentence-validity mask shaped
`(batch, sentences)`. Padding defaults to token ID `0`.

### Initialize from a PureCloudLM checkpoint

The embedding, position encoding, `WordToND`, and compatible global-geometry
weights can be transferred from an existing PureCloudLM checkpoint. The new
window encoder and prediction head retain their fresh initialization.

```python
checkpoint = torch.load("purecloudlm_checkpoint.pt", map_location="cpu")
model.load_purecloudlm_checkpoint(checkpoint)
```

The loader accepts a raw state dict or a checkpoint containing `model_state`,
`model_state_dict`, or `state_dict`; tensors are copied only when their names and
shapes match the configured sentence encoder.

## Main hyperparameters

| Parameter | Default | Meaning |
|---|---:|---|
| `d_model` | 256 | Token embedding width |
| `d_cloud_dim` | 16 | Sentence point dimension |
| `d_sentence` | 512 | Sentence-cloud state width |
| `d_window_dim` | 32 | Projected point dimension within a window |
| `d_window` | 512 | Window-cloud state width |
| `window_size` | 5 | Sentences per window |
| `window_stride` | 2 | Window movement in sentences |
| `hidden` | 256 | Geometry encoder hidden width |
| `padding_idx` | 0 | Padding token ID |

If an input has fewer sentences than `window_size`, its window is zero-padded
and masked. Both sentence and window encoders reuse the PureCloudLM
global-statistics encoder; no Transformer or quadratic all-pairs window
computation is used.
