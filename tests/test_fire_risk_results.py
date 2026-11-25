from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_results" / "fire_risk"


def test_fire_research_files_exist():
    required = [
        "dataset_summary.csv", "model_comparison.csv", "evaluation_details.csv",
        "fire_risk_results.md", "reproducibility.md", "research_paper_section.md",
        "limitations.md", "fire_risk_analysis.json",
    ]
    for name in required:
        assert (OUT / name).exists(), name


def test_fire_figures_exist_at_presentation_resolution():
    figures = sorted((OUT / "figures").glob("*.png"))
    assert len(figures) == 10
    for p in figures:
        assert p.stat().st_size > 10_000, p


def test_reproduced_reference_metrics():
    analysis = json.loads((OUT / "fire_risk_analysis.json").read_text(encoding="utf-8"))
    models = analysis["existing_reference_models"]
    assert abs(models["persistence (fire yesterday)"]["AUPRC"] - 0.7408789489250701) < 1e-12
    assert abs(models["logistic regression"]["AUPRC"] - 0.8530347910413624) < 1e-12
    assert abs(models["gradient-boosted trees"]["AUPRC"] - 0.872330398328095) < 1e-12


def test_split_counts_are_consistent():
    with (OUT / "evaluation_details.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    counts = {r["split"]: r for r in rows}
    assert int(counts["train"]["rows"]) == 13176
    assert int(counts["validation"]["rows"]) == 3294
    assert int(counts["test"]["rows"] ) == 6588
