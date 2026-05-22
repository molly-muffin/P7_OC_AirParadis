#!/usr/bin/env python3
"""Generate local screenshot substitutes for blog/slides (metrics chart, API traces)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)


def write_metrics_chart() -> None:
    results_path = ROOT / "models" / "production" / "model_comparison.json"
    if not results_path.exists():
        print("Skip chart: model_comparison.json missing")
        return

    data = json.loads(results_path.read_text())
    models = [r["model"] for r in data]
    test_f1 = [float(r.get("test_f1", 0)) * 100 for r in data]

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ["#4C72B0" if m != "distilbert_finetuned" else "#C44E52" for m in models]
        ax.barh(models, test_f1, color=colors)
        ax.set_xlabel("Test F1 (%)")
        ax.set_title("Model comparison (50k sample)")
        ax.set_xlim(0, 100)
        for i, v in enumerate(test_f1):
            ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "mlflow_model_comparison.png", dpi=150)
        plt.close(fig)
        print(f"Wrote {OUT / 'mlflow_model_comparison.png'}")
    except ImportError:
        (OUT / "mlflow_model_comparison.txt").write_text(
            "\n".join(f"{m}: {v:.1f}%" for m, v in zip(models, test_f1))
        )
        print("matplotlib missing; wrote text fallback")


def capture_pytest() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    (OUT / "pytest_ci_output.txt").write_text(proc.stdout + proc.stderr)
    print(f"pytest exit {proc.returncode} -> pytest_ci_output.txt")


def capture_api_curl() -> None:
    lines = ["# Local API smoke test (run API first: uvicorn src.api.main:app --port 8000)\n"]
    for url, method, body in [
        ("http://localhost:8000/health", "GET", None),
        (
            "http://localhost:8000/predict",
            "POST",
            '{"text":"I love Air Paradis!"}',
        ),
    ]:
        cmd = ["curl", "-s", "-m", "5", url]
        if method == "POST":
            cmd.extend(["-X", "POST", "-H", "Content-Type: application/json", "-d", body])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        lines.append(f"$ curl {' '.join(cmd[1:])}\n{proc.stdout or proc.stderr}\n")
    (OUT / "api_curl_local.txt").write_text("\n".join(lines))
    print("Wrote api_curl_local.txt")


if __name__ == "__main__":
    write_metrics_chart()
    capture_pytest()
    capture_api_curl()
