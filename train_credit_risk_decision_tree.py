#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


@dataclass
class TrainConfig:
    input_csv: Optional[Path]
    target_column: str
    test_size: float
    random_state: int
    max_depth: Optional[int]
    class_weight: Optional[str]
    model_path: Path
    metrics_path: Path
    use_demo: bool
    demo_samples: int


@dataclass
class PredictConfig:
    input_csv: Path
    model_path: Path
    output_csv: Optional[Path]
    include_proba: bool


def infer_feature_types(df: pd.DataFrame, target_column: str) -> Tuple[List[str], List[str]]:
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not in input data columns: {list(df.columns)}")

    candidate_df = df.drop(columns=[target_column])

    numeric_features: List[str] = list(candidate_df.select_dtypes(include=[np.number]).columns)
    categorical_features: List[str] = list(
        candidate_df.select_dtypes(include=["object", "category", "bool"]).columns
    )

    if not numeric_features and not categorical_features:
        raise ValueError("No features found after excluding target column; check your input data.")

    return numeric_features, categorical_features


def build_pipeline(
    numeric_features: List[str],
    categorical_features: List[str],
    max_depth: Optional[int],
    class_weight: Optional[str],
    random_state: int,
) -> Pipeline:
    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
    )

    model = Pipeline(steps=[("preprocess", preprocessor), ("classifier", classifier)])
    return model


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
) -> Dict[str, object]:
    labels = np.unique(y_true)
    average = "binary" if len(labels) == 2 else "macro"

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    metrics: Dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "support": int(support) if np.isscalar(support) else [int(s) for s in support],
        "classification_report": classification_report(y_true, y_pred, digits=4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and len(labels) == 2:
        try:
            proba_positive = y_proba[:, 1]
            metrics["roc_auc"] = float(roc_auc_score(y_true, proba_positive))
        except Exception:
            # Silently skip ROC-AUC if computation fails
            pass

    return metrics


def train_and_evaluate(config: TrainConfig) -> Tuple[Pipeline, Dict[str, object]]:
    if config.use_demo:
        df = generate_demo_dataset(config.demo_samples, config.random_state)
        target_column = "credit_risk"
    else:
        if config.input_csv is None:
            raise ValueError("--input is required when not using --demo")
        df = pd.read_csv(config.input_csv)
        target_column = config.target_column

    numeric_features, categorical_features = infer_feature_types(df, target_column)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.test_size, random_state=config.random_state, stratify=y if y.nunique() > 1 else None
    )

    model = build_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        max_depth=config.max_depth,
        class_weight=config.class_weight,
        random_state=config.random_state,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None

    metrics = compute_metrics(y_true=y_test.to_numpy(), y_pred=y_pred, y_proba=y_proba)

    return model, metrics


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def save_metrics(metrics: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def load_model(path: Path) -> Pipeline:
    return joblib.load(path)


def run_prediction(config: PredictConfig) -> pd.DataFrame:
    model = load_model(config.model_path)
    df = pd.read_csv(config.input_csv)

    predictions = model.predict(df)

    output = pd.DataFrame({"prediction": predictions})

    if config.include_proba and hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(df)
            if proba.ndim == 2 and proba.shape[1] == 2:
                output["probability_positive"] = proba[:, 1]
            else:
                for i in range(proba.shape[1]):
                    output[f"probability_class_{i}"] = proba[:, i]
        except Exception:
            # If probabilities fail, continue with predictions only
            pass

    if config.output_csv is not None:
        config.output_csv.parent.mkdir(parents=True, exist_ok=True)
        output.to_csv(config.output_csv, index=False)

    return output


def generate_demo_dataset(n_samples: int, random_state: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=random_state)

    age = rng.integers(21, 70, size=n_samples)
    income = rng.normal(loc=60000, scale=20000, size=n_samples).clip(15000, 200000)
    loan_amount = rng.normal(loc=20000, scale=10000, size=n_samples).clip(1000, 150000)
    credit_history_years = rng.integers(0, 30, size=n_samples)
    num_delinquencies = rng.poisson(1.0, size=n_samples)
    employment_status = rng.choice(
        ["employed", "self_employed", "unemployed", "student", "retired"], size=n_samples, p=[0.6, 0.15, 0.1, 0.05, 0.1]
    )
    home_ownership = rng.choice(["rent", "own", "mortgage", "other"], size=n_samples, p=[0.4, 0.2, 0.35, 0.05])

    debt_to_income = (loan_amount / np.maximum(income, 1.0)).clip(0, 5)

    # Risk score: higher is riskier
    risk_score = (
        0.03 * (70 - age)
        + 1.5 * debt_to_income
        + 0.4 * (2 - (credit_history_years / 15.0))
        + 0.5 * (num_delinquencies > 0).astype(float)
        + 0.2 * (employment_status == "unemployed").astype(float)
        + 0.1 * (home_ownership == "rent").astype(float)
        + rng.normal(0, 0.3, size=n_samples)
    )

    # Convert risk score into binary outcome via logistic function
    prob_default = 1 / (1 + np.exp(-(risk_score - 1.0)))
    credit_risk = (rng.random(n_samples) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "credit_history_years": credit_history_years,
            "num_delinquencies": num_delinquencies,
            "employment_status": employment_status,
            "home_ownership": home_ownership,
            "debt_to_income": debt_to_income,
            "credit_risk": credit_risk,
        }
    )

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and predict credit risk using a Decision Tree with preprocessing."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a Decision Tree model")
    train_parser.add_argument("--input", type=Path, default=None, help="Path to input CSV file for training")
    train_parser.add_argument(
        "--target",
        type=str,
        default="credit_risk",
        help="Name of the target column in the training CSV",
    )
    train_parser.add_argument("--test-size", type=float, default=0.2, help="Holdout test size fraction")
    train_parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    train_parser.add_argument("--max-depth", type=int, default=6, help="Max depth for the decision tree")
    train_parser.add_argument(
        "--class-weight",
        type=str,
        choices=["balanced"],
        default=None,
        help="Class weighting strategy (use 'balanced' for imbalanced data)",
    )
    train_parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/decision_tree_credit_risk.joblib"),
        help="Path to save the trained model",
    )
    train_parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/metrics.json"),
        help="Path to save evaluation metrics as JSON",
    )
    train_parser.add_argument(
        "--demo",
        action="store_true",
        help="Use a generated demo dataset instead of reading a CSV",
    )
    train_parser.add_argument(
        "--demo-samples",
        type=int,
        default=2000,
        help="Number of samples for the demo dataset",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict credit risk for applicants")
    predict_parser.add_argument("--input", type=Path, required=True, help="Path to applicant CSV")
    predict_parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/decision_tree_credit_risk.joblib"),
        help="Path to a trained model",
    )
    predict_parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write predictions CSV",
    )
    predict_parser.add_argument(
        "--proba",
        action="store_true",
        help="Include predicted probabilities in the output",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "train":
        config = TrainConfig(
            input_csv=args.input,
            target_column=args.target,
            test_size=args.test_size,
            random_state=args.random_state,
            max_depth=args.max_depth,
            class_weight=args.class_weight,
            model_path=args.model,
            metrics_path=args.metrics,
            use_demo=args.demo,
            demo_samples=args.demo_samples,
        )

        model, metrics = train_and_evaluate(config)
        save_model(model, config.model_path)
        save_metrics(metrics, config.metrics_path)

        print("Training complete. Model saved to:", config.model_path)
        print("Metrics saved to:", config.metrics_path)
        print("\n=== Evaluation Metrics ===")
        print(json.dumps({k: v for k, v in metrics.items() if k != "classification_report"}, indent=2))
        print("\nClassification report:\n", metrics["classification_report"])  # type: ignore[index]

    elif args.command == "predict":
        config = PredictConfig(
            input_csv=args.input,
            model_path=args.model,
            output_csv=args.output,
            include_proba=args.proba,
        )
        output_df = run_prediction(config)

        print("Predictions head:\n", output_df.head().to_string(index=False))
        if config.output_csv is not None:
            print("\nPredictions written to:", config.output_csv)


if __name__ == "__main__":
    main()
