"""
Quann-3D — Model Test & Etkileşim
====================================
Eğitilmiş checkpoint'i yükleyip test et.

Kullanım:
  python3 eval_model.py                    # otomatik testler
  python3 eval_model.py --interactive      # etkileşimli mod
  python3 eval_model.py --prompt "kedi bahçede"  # tek prompt
"""

import torch
import torch.nn.functional as F
import math
import argparse
import os
import sys

# Repozitör kökü (CWD'den bağımsız)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from cloudlm import CloudLM, WordToND, CloudEncoderND, Tok  # noqa: F401

import numpy as np
import random
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def build_model_from_checkpoint(ckpt):
    """Construct a CloudLM using every architecture field saved in its checkpoint."""
    config = ckpt.get("config", {})
    model = CloudLM(
        vocab_size=ckpt["vocab_size"],
        d_model=config.get("d_model", 128),
        d_cloud=config.get("d_cloud", 256),
        d_cloud_dim=config.get("d_cloud_dim", 16),
        n_layers=config.get("n_layers", 2),
        nhead=config.get("nhead"),
        max_pos_len=config.get("max_pos_len", config.get("pos_len", 2048)),
        hidden=config.get("hidden", 256),
        proj_hidden=config.get("proj_hidden"),
        point_hidden=config.get("point_hidden", 128),
        pool=config.get("pool", "max"),
    )
    model.load_state_dict(ckpt["model_state"])
    return model


# ─────────────────────────────────────────────
# TEST FONKSİYONLARI
# ─────────────────────────────────────────────

def test_generation(model, tok, device, prompts, max_new=10, temp=0.7):
    """Çeşitli promptlarla üretim testi."""
    print(f"\n{'='*60}")
    print(f"  🧪 ÜRETİM TESTİ (temp={temp}, max_new={max_new})")
    print(f"{'='*60}")
    
    model.to(device)
    results = []
    for p in prompts:
        tokens = torch.tensor([tok.encode(p)], device=device)
        result = model.generate(tokens, max_new=max_new, temperature=temp)
        out = tok.decode(result[0].tolist())
        results.append((p, out))
        print(f"  {p:35s} → {out}")
    
    return results


def test_geometry(model, tok, device, text):
    """Bulut geometrisi analizi."""
    model.to(device)
    tokens = torch.tensor([tok.encode(text)], device=device)
    mask = (tokens != 0) & (tokens != 1)
    
    pts = model.get_cloud(tokens, mask)
    words = text.lower().split()
    
    print(f"\n  \"{text}\"")
    print(f"  Noktalar ({pts.shape[1]}D):")
    for w, p in zip(words, pts):
        coords = ", ".join(f"{v:+.3f}" for v in p[:4])
        print(f"    {w:15s} ({coords}{'...' if pts.shape[1]>4 else ''})")
    
    # Mesafe matrisi
    n = len(words)
    if n <= 8:
        print("  Mesafe matrisi:")
        print(f"    {'':12s}", end="")
        for w in words:
            print(f"{w[:9]:>11s}", end="")
        print()
        for i in range(n):
            print(f"    {words[i][:10]:12s}", end="")
            for j in range(n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                print(f"{d:11.3f}", end="")
            print()
    
    # Küme istatistikleri
    centroid = pts.mean(axis=0)
    spread = float(np.linalg.norm(pts - centroid, axis=1).mean())
    all_dists = [float(np.linalg.norm(pts[i]-pts[j])) for i in range(n) for j in range(n) if i!=j]
    print(f"  Yayılım: {spread:.3f} | Min/Max dist: {min(all_dists):.3f}/{max(all_dists):.3f}")
    
    return pts


def test_cloud_evolution(model, tok, device, text, save_path=None):
    """Kademeli cümle inşası — bulut evrimi."""
    model.to(device)
    words = text.lower().split()
    n_steps = min(len(words), 6)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), subplot_kw={'projection': '3d'})
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(words), 2)))
    
    for step in range(n_steps):
        partial = " ".join(words[:step+1])
        tokens = torch.tensor([tok.encode(partial)], device=device)
        mask = (tokens != 0) & (tokens != 1)
        pts = model.get_cloud(tokens, mask)[:, :3]
        
        ax = axes[step]
        for i, (w, p) in enumerate(zip(words[:step+1], pts)):
            ax.scatter(*p, s=180, color=colors[i % 10], edgecolors='black', linewidth=0.5)
            ax.text(p[0], p[1], p[2], f" {w[:7]}", fontsize=8, color=colors[i % 10])
        for i in range(len(pts)):
            for j in range(i+1, len(pts)):
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        [pts[i][2], pts[j][2]], 'gray', alpha=0.3, linewidth=0.8)
        ax.set_title(f"Adım {step+1}", fontsize=9)
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    
    for step in range(n_steps, len(axes)):
        axes[step].set_visible(False)
    
    plt.suptitle(f"Bulut Evrimi: {text!r}", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📸 {save_path}")
    plt.close()


def test_word_regions(model, tok, device, texts, n_words=20):
    """Kelime bölge analizi."""
    model.to(device)
    word_positions = {}
    
    for sent in texts:
        words = sent.split()[:8]
        tokens = torch.tensor([tok.encode(' '.join(words))], device=device)
        mask = (tokens != 0) & (tokens != 1)
        pts = model.get_cloud(tokens, mask)[:, :3]
        for w, p in zip(words, pts):
            if w not in word_positions:
                word_positions[w] = []
            word_positions[w].append(p)
    
    word_counts = Counter(w for t in texts for w in t.split()[:8])
    top_words = [w for w, _ in word_counts.most_common(n_words)]
    
    print(f"\n  {'Kelime':15s} {'X':>8s} {'Y':>8s} {'Z':>8s}  {'n':>4s}  Bölge")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}  {'-'*4}  {'-'*12}")
    
    for w in top_words:
        if w in word_positions and len(word_positions[w]) > 3:
            positions = np.array(word_positions[w])
            avg = positions.mean(axis=0)
            std = positions.std(axis=0).mean()
            region = ""
            if avg[0] > 0.2: region += "doğu"
            elif avg[0] < -0.2: region += "batı"
            else: region += "merkez"
            if avg[1] > 0.2: region += "+K"
            elif avg[1] < -0.2: region += "+G"
            print(f"  {w:15s} {avg[0]:+8.3f} {avg[1]:+8.3f} {avg[2]:+8.3f}  {len(positions):4d}  {region} (σ={std:.2f})")
    
    return word_positions


def test_ppl(model, tok, device, texts, seq_len=6, n_test=100):
    """Perplexity testi."""
    model.to(device)
    model.eval()
    all_tokens = [tok.encode(t) for t in texts if len(tok.encode(t)) >= seq_len + 1]
    random.seed(999)
    samples = random.sample(all_tokens, min(n_test, len(all_tokens)))
    
    total_loss, count = 0, 0
    with torch.no_grad():
        for tokens in samples:
            for i in range(0, len(tokens) - seq_len, 2):
                if i + seq_len >= len(tokens): break
                inp = torch.tensor([tokens[i:i+seq_len]], device=device)
                tgt = torch.tensor([tokens[i+seq_len]], device=device)
                mask = (inp != 0) & (inp != 1)
                logits, _, _ = model(inp, mask)
                total_loss += F.cross_entropy(logits, tgt, reduction='sum').item()
                count += 1
    return math.exp(min(total_loss / max(count, 1), 20))


def interactive_mode(model, tok, device):
    """Etkileşimli mod — cümle gir, bulut + tahmin gör."""
    print(f"\n{'='*60}")
    print("  🎮 ETKİLEŞİMLİ MOD")
    print("  Cümle yaz → bulut + bir sonraki kelime tahmini")
    print("  'q' ile çık")
    print(f"{'='*60}")
    
    model.to(device)
    model.eval()
    
    while True:
        try:
            text = input("\n  Cümle: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        
        if text.lower() in ('q', 'quit', 'exit', 'çık'):
            print("  Hoşça kal!")
            break
        if not text:
            continue
        
        tokens = torch.tensor([tok.encode(text)], device=device)
        mask = (tokens != 0) & (tokens != 1)
        
        # Bulut
        pts = model.get_cloud(tokens, mask)
        words = text.lower().split()
        
        print(f"\n  🌩️  Bulut ({pts.shape[1]}D, {len(words)} nokta):")
        for w, p in zip(words, pts):
            coords = " ".join(f"{v:+.2f}" for v in p[:3])
            print(f"    {w:15s} → ({coords})")
        
        # Mesafeler
        n = len(words)
        if 2 <= n <= 8:
            print("\n  🔗 Bağlar:")
            dists = []
            for i in range(n):
                for j in range(i+1, n):
                    d = float(np.linalg.norm(pts[i] - pts[j]))
                    dists.append((d, words[i], words[j]))
            dists.sort()
            for d, w1, w2 in dists[:3]:
                print(f"    {w1} ↔ {w2} = {d:.3f}")
        
        # Tahmin
        with torch.no_grad():
            logits, next_pt, _ = model(tokens, mask)
            top5_idx = torch.argsort(logits[0], descending=True)[:5].tolist()
            top5 = [(tok.itos.get(i, '?'), float(logits[0, i])) for i in top5_idx]
        
        print("\n  🔮 Bir sonraki kelime (top-5):")
        for w, score in top5:
            bar = "█" * int(max(0, (score - top5[-1][1]) / max(top5[0][1] - top5[-1][1], 1e-8) * 20))
            print(f"    {w:15s} {score:+.2f} {bar}")
        
        # Üretim
        result = model.generate(tokens, max_new=5, temperature=0.7)
        out = tok.decode(result[0].tolist())
        print(f"\n  ✍️  Tamamlama: {out}")


# ─────────────────────────────────────────────
# ANA
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Quann-3D Model Test")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(ROOT, "checkpoints", "cloudlm", "cloud_lm_final.pt"))
    parser.add_argument("--tokenizer", type=str,
                        default=os.path.join(ROOT, "checkpoints", "cloudlm", "tokenizer.json"))
    parser.add_argument("--prompt", type=str, default=None, help="Tek prompt")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--max-new", type=int, default=10)
    args = parser.parse_args()
    
    # Device
    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else \
                 "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"  Device: {device}")
    
    # Tokenizer
    print(f"  Tokenizer: {args.tokenizer}")
    tok = Tok.load(args.tokenizer)
    print(f"  Vocab: {tok.vocab_size}")
    
    # Model
    print(f"  Checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    
    config = ckpt.get("config", {})
    model = build_model_from_checkpoint(ckpt)
    print(f"  Model: d_model={config.get('d_model')}, d_cloud={config.get('d_cloud')}, "
          f"D={config.get('d_cloud_dim')}, layers={config.get('n_layers')}")
    print(f"  Epoch: {ckpt.get('epoch', '?')} | PPL: {ckpt.get('ppl', '?')}")
    
    # ── TEST MODU ──
    
    if args.prompt:
        # Tek prompt
        model.to(device)
        tokens = torch.tensor([tok.encode(args.prompt)], device=device)
        mask = (tokens != 0) & (tokens != 1)
        
        # Bulut
        pts = model.get_cloud(tokens, mask)
        words = args.prompt.lower().split()
        print("\n  🌩️  Bulut:")
        for w, p in zip(words, pts):
            coords = " ".join(f"{v:+.3f}" for v in p[:4])
            print(f"    {w:15s} ({coords}{'...' if pts.shape[1]>4 else ''})")
        
        # Tahmin
        model.to(device)
        with torch.no_grad():
            logits, _, _ = model(tokens, mask)
            top5 = torch.argsort(logits[0], descending=True)[:5].tolist()
            print("\n  🔮 Top-5:")
            for i in top5:
                print(f"    {tok.itos.get(i, '?'):15s} {logits[0, i]:+.2f}")
        
        result = model.generate(tokens, max_new=args.max_new, temperature=args.temp)
        print(f"\n  ✍️  {tok.decode(result[0].tolist())}")
    
    elif args.interactive:
        interactive_mode(model, tok, device)
    
    else:
        # Otomatik testler
        print(f"\n{'='*60}")
        print("  🧪 OTOMATİK TESTLER")
        print(f"{'='*60}")
        
        # 1. Üretim
        prompts = [
            "kedi bahçede",
            "çocuk parkta topu",
            "kadın pazara domatesi",
            "öğrenci kütüphanede",
            "müzisyen sahne de",
            "her duygunun bir",
            "bu cümleyi kurduğumda",
        ]
        test_generation(model, tok, device, prompts, max_new=args.max_new, temp=args.temp)
        
        # 2. Geometri
        print(f"\n{'='*60}")
        print("  🔬 GEOMETRİ TESTİ")
        print(f"{'='*60}")
        
        test_sentences = [
            "kedi bahçede uyudu",
            "çocuk parkta topu kovaladı",
            "her duygunun bir zıttı var",
            "bu cümleyi kurduğumda fark ettim",
        ]
        for sent in test_sentences:
            test_geometry(model, tok, device, sent)
            print()
        
        # 3. Bulut evrimi
        print(f"\n{'='*60}")
        print("  🌩️  BULUT EVRİMİ")
        print(f"{'='*60}")
        test_cloud_evolution(model, tok, device, "çocuk parkta topu kovaladı",
                           save_path="test_evolution.png")
        
        # 4. Kelime bölgeleri (küçük örnek)
        print(f"\n{'='*60}")
        print("  📐 KELİME BÖLGELERİ")
        print(f"{'='*60}")
        
        sample_texts = [
            "bir gün bu cümleyi yazdım",
            "ve sonra ne olduğunu anladım",
            "çok güzel bir gündü",
            "daha önce hiç böyle hissetmemiştim",
            "için teşekkür ederim",
            "gibi bir şey olduğunu fark ettim",
            "bu yüzden daha dikkatli olmalıyız",
            "her şey bir anda değişti",
            "bu kararı vermem çok zor oldu",
            "gün batımını izlerken düşündüm",
        ] * 10
        test_word_regions(model, tok, device, sample_texts, n_words=15)
        
        print(f"\n\n{'='*60}")
        print("  ✅ TESTLER TAMAM")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
