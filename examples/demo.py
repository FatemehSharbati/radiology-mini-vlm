"""End-to-end fairness-audit demo on synthetic data.

Runs with no downloads and no model weights (CPU only):
    python examples/demo.py        # from the repo root
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fairness_audit import audit_csv, make_demo_csv, print_gap_table

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)


def main():
    csv_path = OUT / "demo_audit.csv"
    make_demo_csv(csv_path, n=2000, seed=0)
    print(f"demo CSV written to: {csv_path}")

    plot_path = OUT / "subgroup_auroc.png"
    results = audit_csv(csv_path, n_boot=500, plot_path=plot_path)
    print()
    print_gap_table(results)
    print()
    print(f"plot saved to: {plot_path}")
    print("Note: all data here is synthetic — it represents no real patients.")


if __name__ == "__main__":
    main()
