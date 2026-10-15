import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "quann_cloud", "src"))

from qcloud_trainer import QCloudTrainer
import torch
from qcloud_network import QCloudNetwork

# Tam corpus (şablon — gerçek corpus'a göre düzenlenecek)
corpus = [
    line.strip() for line in open(os.path.join(ROOT, "data", "turkish_sentences.txt"), 'r')  # Geçici: tiny corpus
]

# Tam model (gerçek eğitim için)
model = QCloudNetwork(
    vocab_size=1000,  # Gerçek vocab boyutu
    d_model=256,
    d_cloud_dim=16,
    n_layers=4
)

device = torch.device('cpu')  # Test için CPU
trainer = QCloudTrainer(
    model=model,
    corpus=corpus,
    device=device,
    max_epochs=20,  # Gerçek eğitim için
    eval_interval=5
)

# Eğitim
print("=== FULL CORPUS TRAINING ===")
result = trainer.train(verbose=True)
print(f"\nBest PPL: {result['best_ppl']:.2f} (epoch {result['best_epoch']})")
