"""Smoke tests — CPU only, no downloads, no model weights.

The CLIP retrieval functions that need weights are exercised only through
``rank_by_cosine``, which is pure tensor math over caller-supplied embeddings.

Run from the repo root:
    python tests/test_smoke.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.fairness_audit import audit_csv, make_demo_csv

try:
    import torch

    from src.retrieval import rank_by_cosine

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def test_audit_detects_subgroup_gap():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "demo.csv")
        make_demo_csv(csv_path, n=600, seed=1)
        assert os.path.exists(csv_path)
        results = audit_csv(csv_path, n_boot=100)

    assert set(results["subgroup"]) == {"A", "B"}
    assert results["auroc"].between(0.5, 1.0).all(), "AUROC out of range"
    assert ((results["ci_lo"] <= results["auroc"]) & (results["auroc"] <= results["ci_hi"])).all()

    a = results.loc[results["subgroup"] == "A", "auroc"].iloc[0]
    b = results.loc[results["subgroup"] == "B", "auroc"].iloc[0]
    assert a > b, "demo data should give subgroup A a higher AUROC than B"
    gap_b = results.loc[results["subgroup"] == "B", "gap_vs_best"].iloc[0]
    assert gap_b > 0, "subgroup B should show a positive gap vs best"


def test_audit_rejects_bad_input():
    with tempfile.TemporaryDirectory() as tmp:
        bad = os.path.join(tmp, "bad.csv")
        pd.DataFrame({"y_true": [0, 1], "y_score": [0.1, 0.9]}).to_csv(bad, index=False)
        try:
            audit_csv(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for missing 'subgroup' column")

        single_class = os.path.join(tmp, "single.csv")
        pd.DataFrame(
            {
                "y_true": [1, 1, 1],
                "y_score": [0.8, 0.9, 0.7],
                "subgroup": ["A", "A", "A"],
            }
        ).to_csv(single_class, index=False)
        try:
            audit_csv(single_class)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for single-class subgroup")


def test_rank_by_cosine_fake_embeddings():
    if not _TORCH_AVAILABLE:
        print("SKIP test_rank_by_cosine_fake_embeddings (torch not installed)")
        return
    # Fake precomputed embeddings: query is identical to image 1.
    query = torch.tensor([[1.0, 0.0, 0.0]])
    images = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    idx, scores = rank_by_cosine(query, images, top_k=2)
    assert idx[0] == 1, f"expected image 1 first, got {idx}"
    assert len(idx) == 2 and len(scores) == 2
    assert scores[0] > scores[1], "scores should be descending"
    assert abs(scores[0] - 1.0) < 1e-6, "identical vectors => cosine 1.0"

    # top_k larger than N is clamped, not an error
    idx_all, _ = rank_by_cosine(query, images, top_k=99)
    assert len(idx_all) == 3


if __name__ == "__main__":
    test_audit_detects_subgroup_gap()
    test_audit_rejects_bad_input()
    test_rank_by_cosine_fake_embeddings()
    print("smoke tests passed")
