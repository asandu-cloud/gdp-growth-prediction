"""
layer5_models.py -- Orchestrator for all Layer 5 sub-modules.

Usage:
    python thesis_pipeline/layer5_models.py

Runs all five estimation sub-modules in sequence, then the
evaluation module that produces metrics, statistical tests,
LaTeX tables, and figures.
"""
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thesis_pipeline.layer5a_ols import main as run_ols
from thesis_pipeline.layer5b_quantile import main as run_quantile
from thesis_pipeline.layer5c_rf import main as run_rf
from thesis_pipeline.layer5d_midas import main as run_midas
from thesis_pipeline.layer5e_probit import main as run_probit
from thesis_pipeline.layer5_evaluation import main as run_evaluation


def main():
    t0 = time.time()
    print("=" * 60)
    print("LAYER 5: Full Model Pipeline")
    print("=" * 60)

    run_ols()
    run_probit()
    run_quantile()
    run_rf()
    run_midas()
    run_evaluation()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"LAYER 5 COMPLETE — {elapsed / 60:.1f} min total")
    print("=" * 60)


if __name__ == "__main__":
    main()
