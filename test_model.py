"""
Basit model test - wiki_cloud_best.pt
"""
import torch
from cloudlm import CloudLM, BPETokenizer

def test_model(prompt, ckpt_path, bpe_path, max_new=15):
    # Yükle
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    tok = BPETokenizer.load(bpe_path)
    
    # Model oluştur (config eksikse default değerler)
    cfg = ckpt.get('config', {})
    model = CloudLM(
        ckpt['vocab_size'],
        d_model=cfg.get('d_model', 256),
        d_cloud=cfg.get('d_cloud', 512),
        d_cloud_dim=cfg.get('d_cloud_dim', 32),
        n_layers=cfg.get('n_layers', 4),
        pool=cfg.get('pool', 'max'),
        cloud_attn=cfg.get('cloud_attn', False),
        cloud_attn_heads=cfg.get('cloud_attn_heads', 4),
        attn_scale_init=1.0,
    )
    
    # State dict yükle (strict=False ile eksik key'leri görmezden gel)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    
    # Tokenize
    tokens = torch.tensor([tok.encode(prompt)])
    
    # Generate
    out = model.generate(tokens, max_new=max_new, temperature=0.7)
    
    return tok.decode(out[0].tolist())

if __name__ == "__main__":
    ckpt_path = "checkpoints/wiki_checkpoints_wiki1500k/wiki_cloud_best.pt"
    bpe_path = "checkpoints/wiki_checkpoints_wiki1500k/wiki_bpe.json"
    
    prompts = [
        "Türkiye'nin başkenti",
        "bilgisayar bilimlerinde",
        "sanat tarihi açısından",
        "ekonomik büyüme",
        "demokrasi ve insan",
    ]
    
    print("Testing wiki_cloud_best.pt\n")
    for p in prompts:
        result = test_model(p, ckpt_path, bpe_path)
        print(f"  {p:30s} → {result}")
