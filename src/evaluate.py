"""Evaluate Pyrenex Crédit risk model on the holdout dataset.

Usage:
    python src/evaluate.py --model models/pyrenex_risk_v2.joblib \
                           --data data/lending_club_holdout.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)

from preprocess import load_dataset


def evaluate(model_path: Path, data_path: Path) -> dict:
    pipeline = joblib.load(model_path)
    X_holdout, y_holdout = load_dataset(data_path)

    y_pred = pipeline.predict(X_holdout)
    y_proba = pipeline.predict_proba(X_holdout)[:, 1]

    return {
        "f1_macro": round(f1_score(y_holdout, y_pred, average="macro"), 4),
        "f1_default": round(f1_score(y_holdout, y_pred, pos_label=1), 4),
        "roc_auc": round(roc_auc_score(y_holdout, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_holdout, y_pred).tolist(),
        "classification_report": classification_report(
            y_holdout, y_pred, target_names=["Remboursé", "Défaut"], output_dict=True
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Pyrenex risk model on holdout")
    parser.add_argument("--model", default="models/pyrenex_risk_v2.joblib", type=Path)
    parser.add_argument("--data", default="data/lending_club_holdout.csv", type=Path)
    args = parser.parse_args()

    metrics = evaluate(args.model, args.data)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
