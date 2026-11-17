"""Maintenance E0 paths and deterministic defaults."""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
LOGS_DIR = os.path.join(PROJECT_DIR, "logs")
SEED = 20261016
DEVICE = "cpu"
FEATURE_COUNT = 36
WINDOW_SIZE = 10
STAGES = ("critical_0_30", "high_31_60", "moderate_61_90", "healthy_gt_90")
