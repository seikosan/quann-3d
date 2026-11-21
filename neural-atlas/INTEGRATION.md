# Neural Atlas Integration for CloudLM

## Overview
Neural Atlas visualization architecture extracted from https://bbycroft.net/llm and integrated into Quann-3D project.

## Files Created

### 1. `neural-atlas/generate_cloudlm_json.py`
Python script to generate Neural Atlas JSON from CloudLM checkpoints.

**Usage:**
```bash
python3 neural-atlas/generate_cloudlm_json.py \
  --ckpt checkpoints/wiki_checkpoints_wiki200k_d32/wiki_cloud_best.pt \
  --output neural-atlas/neural-atlas/cloudlm_d32.json
```

> Not (2026-11-21 refactor C2): canlı uygulamanın web-kökü `neural-atlas/neural-atlas/`
> (start_viewer.sh oradan servis eder; viewer `cloudlm_d32.json`'u kendi dizininden okur).
> Bayat üst-düzey kopyalar (`cloudlm_viewer.html`, `cloudlm_d*.json`, `dist/`) silindi.

**Features:**
- Extracts model architecture from checkpoint
- Generates nodes for all layers (embedding, transformer, cloud encoder, etc.)
- Calculates parameter counts
- Validates JSON structure

### 2. `neural-atlas/neural-atlas/cloudlm_d32.json` & `cloudlm_d64.json`
Generated Neural Atlas JSON files for D32 and D64 models.

**Structure:**
- 36 nodes per model
- 43 edges per model
- D32: 17.5M parameters
- D64: 17.5M parameters (slightly more due to higher d_cloud_dim)

### 3. `neural-atlas/neural-atlas/cloudlm_viewer.html`
Interactive 3D viewer for CloudLM architectures.

**Features:**
- 3D visualization of neural network architecture
- Node selection with info panel
- Export to PNG
- Multiple model support (D32, D64, example Transformer)

**Usage:**
```bash
cd neural-atlas
python3 -m http.server 8080 --bind 127.0.0.1 --directory neural-atlas
# Open http://127.0.0.1:8080/cloudlm_viewer.html
```

## Architecture Details

### CloudLM Layers (D32/D64):
1. **Input** - Tokenized sequence (seq_len=10)
2. **Embedding** - 20000×256 (5.12M params)
3. **Pos Encoding** - Positional embeddings
4. **Transformer Layers** (4 layers):
   - Layer Norm 1
   - Self Attention
   - Residual Add
   - Layer Norm 2
   - FFN Up (256→1024)
   - FFN Down (1024→256)
   - Residual Add
5. **Final Layer Norm**
6. **Word → ND** - Project to D-dimensional cloud space
7. **Cloud Encoder** - PointNet-style encoder
8. **Word Head** - Predict next token
9. **Output** - Logits

### Neural Atlas Format
```json
{
  "name": "CloudLM (D=32)",
  "description": "...",
  "nodes": [
    {
      "id": "embedding",
      "label": "Embedding (20000×256)",
      "type": "embedding",
      "shape": [10, 256],
      "params": 5120000,
      "description": "..."
    }
  ],
  "edges": [
    {"source": "input", "target": "embedding"}
  ]
}
```

## Integration with Quann-3D

The Neural Atlas visualization can be used to:
1. Visualize CloudLM architecture for documentation
2. Compare D32 vs D64 architectures
3. Export high-quality PNG figures for papers
4. Interactive exploration of model structure

## References
- Original: https://bbycroft.net/llm
- Neural Atlas source: `neural-atlas/neural-atlas/src/`
