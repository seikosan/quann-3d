"""Path + scale configuration for the fraud sub-project.

Data sources (single source of truth = ~/Downloads/archive, IBM Credit Card
Fraud Detection set):
  - BIG_CSV   : full 24.4M-row dataset (2.35 GB). Referenced IN PLACE — not
                copied into the repo (disk is ~96% full; keeps a single copy).
  - SMALL_*   : small subsets copied into fraud/data/ for self-contained dev
                and the (slow) QUANN 0th-order layer.

Override the archive location with the FRAUD_DATA_DIR env var if the files
are moved.
"""
import os
import sys

# --- import bootstrap -------------------------------------------------------
# This module lives in fraud/src. Add the REPO ROOT to sys.path so the
# canonical `cloudlm` and `quann_cloud` packages (research core) are importable
# from any fraud/src script run as `python3 fraud/src/<script>.py`.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# The user-provided directory holding the raw CSVs (big file lives here).
DATA_DIR = os.environ.get(
    "FRAUD_DATA_DIR",
    os.path.expanduser("~/Downloads/archive"),
)

# Small bundled subsets (copied into the repo, safe for git-less dev use).
_BUNDLED = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

BIG_CSV = os.path.join(DATA_DIR, "credit_card_transactions-ibm_v2.csv")
SMALL_CSV = os.path.join(_BUNDLED, "User0_credit_card_transactions.csv")
CARDS_CSV = os.path.join(_BUNDLED, "sd254_cards.csv")
USERS_CSV = os.path.join(_BUNDLED, "sd254_users.csv")

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FIGURES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures")
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")

# Label / feature column names in the raw IBM schema.
LABEL_COL = "Is Fraud?"
GROUP_COL = "User"          # identity -> used for GroupKFold (no leakage)
AMOUNT_COL = "Amount"

# Random seed used across the sub-project for reproducibility.
SEED = 20261015
