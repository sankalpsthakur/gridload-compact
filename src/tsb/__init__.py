"""Compact energy-load forecaster + TS-Arena-faithful backtest harness."""
import os

# Repo root; TSB_ROOT overrides it (Kaggle kernels extract the repo to a scratch directory).
ROOT = os.environ.get("TSB_ROOT") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
