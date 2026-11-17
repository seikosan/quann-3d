"""Configuration for the CFPB complaint-cloud pilot."""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

COMPLAINTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "data", "complaints", "cfpb_narrative_snapshot")
PARQUET_DIR = os.path.join(DATA_DIR, "data")
RESULTS_DIR = os.path.join(COMPLAINTS_DIR, "results")
LOGS_DIR = os.path.join(COMPLAINTS_DIR, "logs")
FIGURES_DIR = os.path.join(COMPLAINTS_DIR, "figures")

D = 32
D_CLOUD = 128
D_MODEL = 128
HIDDEN = 128
N_SAMPLE = 100_000
SEED = 0
MAX_LEN = 64
N_REGIONS = 8
BPE_VOCAB_SIZE = 8_000
EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
