#!/usr/bin/env python3
"""CloudLM model architecture to Neural Atlas JSON converter."""

import torch
import json
import sys
import os

# Add repo root to path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)



def get_model_architecture(ckpt_path):
    """Extract CloudLM architecture from checkpoint."""
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    
    nodes = []
    edges = []
    node_id = 0
    
    # Input layer
    nodes.append({
        "id": "input",
        "label": "Input (tokens)",
        "type": "input",
        "shape": [10],  # seq_len
        "description": "Tokenized input sequence"
    })
    
    # Embedding layer
    d_model = config['d_model']
    vocab_size = ckpt['vocab_size']
    nodes.append({
        "id": "embedding",
        "label": f"Embedding ({vocab_size}×{d_model})",
        "type": "embedding",
        "shape": [10, d_model],
        "params": vocab_size * d_model,
        "description": f"Token embeddings: {vocab_size} vocab × {d_model} dim"
    })
    edges.append({"source": "input", "target": "embedding"})
    
    # Positional encoding (not a learnable layer, just visual)
    nodes.append({
        "id": "pos_enc",
        "label": "Pos Encoding",
        "type": "custom",
        "shape": [10, d_model],
        "description": "Learned positional embeddings"
    })
    edges.append({"source": "embedding", "target": "pos_enc"})
    
    # Transformer encoder layers
    n_layers = config['n_layers']
    d_cloud = config['d_cloud']
    d_cloud_dim = config['d_cloud_dim']
    
    for layer_idx in range(n_layers):
        layer_id = f"layer_{layer_idx}"
        
        # Transformer encoder layer
        nodes.append({
            "id": f"{layer_id}_norm1",
            "label": "Layer Norm 1",
            "type": "norm",
            "shape": [10, d_model],
            "description": f"Transformer layer {layer_idx+1} normalization"
        })
        
        # Multi-head attention
        nodes.append({
            "id": f"{layer_id}_attn",
            "label": "Self Attention",
            "type": "attention",
            "shape": [10, d_model],
            "description": f"Multi-head self-attention (layer {layer_idx+1})"
        })
        
        # Residual add
        nodes.append({
            "id": f"{layer_id}_add1",
            "label": "Residual Add",
            "type": "add",
            "shape": [10, d_model],
            "description": f"Residual connection (layer {layer_idx+1})"
        })
        
        # Feed-forward network
        nodes.append({
            "id": f"{layer_id}_norm2",
            "label": "Layer Norm 2",
            "type": "norm",
            "shape": [10, d_model],
            "description": f"Feed-forward normalization (layer {layer_idx+1})"
        })
        
        # FFN up projection
        ff_dim = d_model * 4
        nodes.append({
            "id": f"{layer_id}_ff_up",
            "label": "FFN Up",
            "type": "linear",
            "shape": [10, ff_dim],
            "params": d_model * ff_dim,
            "description": f"Feed-forward up-projection (layer {layer_idx+1})"
        })
        
        # FFN down projection
        nodes.append({
            "id": f"{layer_id}_ff_down",
            "label": "FFN Down",
            "type": "linear",
            "shape": [10, d_model],
            "params": ff_dim * d_model,
            "description": f"Feed-forward down-projection (layer {layer_idx+1})"
        })
        
        # Residual add
        nodes.append({
            "id": f"{layer_id}_add2",
            "label": "Residual Add",
            "type": "add",
            "shape": [10, d_model],
            "description": f"Residual connection (layer {layer_idx+1})"
        })
        
        # Connect transformer layer
        if layer_idx == 0:
            prev_id = "pos_enc"
        else:
            prev_id = f"layer_{layer_idx-1}_add2"
        
        # Layer norm 1
        edges.append({"source": prev_id, "target": f"{layer_id}_norm1"})
        
        # Self attention
        edges.append({"source": f"{layer_id}_norm1", "target": f"{layer_id}_attn"})
        
        # Residual add 1
        edges.append({"source": f"{layer_id}_attn", "target": f"{layer_id}_add1"})
        edges.append({"source": prev_id, "target": f"{layer_id}_add1"})
        
        # Layer norm 2
        edges.append({"source": f"{layer_id}_add1", "target": f"{layer_id}_norm2"})
        
        # FFN up
        edges.append({"source": f"{layer_id}_norm2", "target": f"{layer_id}_ff_up"})
        
        # FFN down
        edges.append({"source": f"{layer_id}_ff_up", "target": f"{layer_id}_ff_down"})
        
        # Residual add 2
        edges.append({"source": f"{layer_id}_ff_down", "target": f"{layer_id}_add2"})
        edges.append({"source": f"{layer_id}_add1", "target": f"{layer_id}_add2"})
    
    # Final layer norm
    final_norm_id = "final_norm"
    nodes.append({
        "id": final_norm_id,
        "label": "Final Layer Norm",
        "type": "norm",
        "shape": [10, d_model],
        "description": "Final layer normalization"
    })
    edges.append({"source": f"layer_{n_layers-1}_add2", "target": final_norm_id})
    
    # Cloud projection (WordToND)
    nodes.append({
        "id": "to_nd",
        "label": "Word → ND",
        "type": "custom",
        "shape": [10, d_cloud_dim],
        "params": d_model * d_cloud_dim,
        "description": f"Project hidden to {d_cloud_dim}-D cloud space"
    })
    edges.append({"source": final_norm_id, "target": "to_nd"})
    
    # Cloud encoder
    nodes.append({
        "id": "cloud_encoder",
        "label": "Cloud Encoder",
        "type": "custom",
        "shape": [10, d_cloud_dim],
        "params": d_cloud_dim * d_cloud * 2 + d_cloud * 4,  # Approximate
        "description": f"PointNet-style cloud encoder ({d_cloud} hidden)"
    })
    edges.append({"source": "to_nd", "target": "cloud_encoder"})
    
    # Word head
    nodes.append({
        "id": "word_head",
        "label": "Word Head",
        "type": "linear",
        "shape": [d_cloud, vocab_size],
        "params": d_cloud * vocab_size,
        "description": f"Predict next token ({vocab_size} vocab)"
    })
    edges.append({"source": "cloud_encoder", "target": "word_head"})
    
    # Output
    nodes.append({
        "id": "output",
        "label": "Output (logits)",
        "type": "output",
        "shape": [vocab_size],
        "description": f"Next token logits ({vocab_size} vocab)"
    })
    edges.append({"source": "word_head", "target": "output"})
    
    return {
        "name": f"CloudLM (D={d_cloud_dim})",
        "description": f"CloudLM architecture with {n_layers} transformer layers, d_model={d_model}, d_cloud={d_cloud}, d_cloud_dim={d_cloud_dim}, vocab={vocab_size}",
        "nodes": nodes,
        "edges": edges
    }


def main():
    """Generate Neural Atlas JSON from CloudLM checkpoint."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate Neural Atlas JSON from CloudLM checkpoint')
    parser.add_argument('--ckpt', required=True, help='Path to CloudLM checkpoint')
    parser.add_argument('--output', required=True, help='Output JSON path')
    args = parser.parse_args()
    
    print(f"Loading checkpoint: {args.ckpt}")
    arch = get_model_architecture(args.ckpt)
    
    print(f"Generating JSON with {len(arch['nodes'])} nodes and {len(arch['edges'])} edges")
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(arch, f, indent=2, ensure_ascii=False)
    
    print(f"Saved to: {args.output}")
    
    # Print summary
    total_params = sum(n.get('params', 0) for n in arch['nodes'])
    print(f"Total parameters: {total_params:,}")


if __name__ == '__main__':
    main()
