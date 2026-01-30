from __future__ import annotations

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Iterable
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from anon import anonimization_clustering
from constraints import BINARY_CLASS_NAMES, ORIGINAL_CLASS_NAMES
from file_utils import (
    binary_path as RELATIVE_BINARY_PATH,
    ensure_binary_dataset,
    load_and_analyze_dataset,
    load_and_split_data,
    path as RELATIVE_ORIGINAL_PATH,
)
from distribution_metrics import per_feature_shift_metrics, summarize_shift_metrics
from mia import evaluate_mia

ALPHA_VALUES: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_K = 3
RANDOM_STATE = 42
DEFAULT_MODE = "both"
VICTIM_MODEL_CHOICES = [
    "knn",
    "mlp",
    "decision_tree",
    "gaussian_nb",
    "logistic_regression",
]
DEFAULT_STATS_BINS = 30


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run anonymization experiments with classical ML models and/or membership inference attacks."
        )
    )
    parser.add_argument(
        "--mode",
        choices=["ml", "mia", "stats", "both"],
        default=DEFAULT_MODE,
        help=(
            "Select whether to run ML experiments (ml), membership inference (mia), "
            "distribution-shift stats only (stats), or everything (both)."
        ),
    )
    parser.add_argument(
        "--victim-models",
        choices=VICTIM_MODEL_CHOICES,
        nargs="+",
        default=VICTIM_MODEL_CHOICES,
        help="Victim models to run during membership inference (default: all training models).",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        help="Optional override for alpha values used during anonymization.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_K,
        help="Number of KMeans clusters used inside the anonymization routine.",
    )
    parser.add_argument(
        "--no-stats",
        action="store_true",
        help="Disable distribution-shift statistics (KL/Wasserstein).",
    )
    parser.add_argument(
        "--stats-bins",
        type=int,
        default=DEFAULT_STATS_BINS,
        help="Number of histogram bins used for KL divergence (default: 30).",
    )
    return parser.parse_args()


def _resolve_path(relative_path: str) -> str:
    """Resolve paths relative to the project root regardless of cwd."""
    project_root = Path(__file__).resolve().parent.parent
    return str(project_root / relative_path)


def _load_dataset_numpy(dataset_path: str, target_column: str = "y") -> tuple[np.ndarray, np.ndarray]:
    X_df, y_series = load_and_split_data(dataset_path, target_column=target_column)
    non_numeric_cols = X_df.select_dtypes(exclude=["number"]).columns.tolist()
    if non_numeric_cols:
        print(
            "Converting non-numeric feature columns to categorical codes:",
            ", ".join(non_numeric_cols),
        )
        for column in non_numeric_cols:
            X_df[column], _ = pd.factorize(X_df[column])
    X = X_df.to_numpy(dtype=float)
    if pd.api.types.is_numeric_dtype(y_series):
        y = y_series.to_numpy()
    else:
        y, _ = pd.factorize(y_series)
    return X, y


def _parallel_anonymizations(
    X: np.ndarray,
    y: np.ndarray,
    alphas: Iterable[float],
    k: int = DEFAULT_K,
) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    alpha_list = list(alphas)
    max_workers = min(len(alpha_list), os.cpu_count() or 1)
    results: dict[float, tuple[np.ndarray, np.ndarray]] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_alpha = {
            executor.submit(anonimization_clustering, X, y, k, alpha): alpha for alpha in alpha_list
        }
        for future in as_completed(future_to_alpha):
            alpha = future_to_alpha[future]
            try:
                X_anon, y_anon = future.result()
            except Exception as exc:
                print(f"Alpha {alpha} failed during anonymization: {exc}")
                continue
            results[alpha] = (X_anon, y_anon)
    return results


def _model_builders(random_state: int = RANDOM_STATE) -> dict[str, Pipeline | GaussianNB | DecisionTreeClassifier]:
    return {
        "KNN": Pipeline(
            [("scaler", StandardScaler()), ("model", KNeighborsClassifier(n_neighbors=5))]
        ),
        "MLP": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state),
                ),
            ]
        ),
        "DecisionTree": DecisionTreeClassifier(random_state=random_state),
        "GaussianNB": GaussianNB(),
        "LogisticRegression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, random_state=random_state),
                ),
            ]
        ),
    }


def _evaluate_models(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = RANDOM_STATE,
) -> list[dict[str, float | str]]:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=random_state,
        stratify=y,
    )
    results: list[dict[str, float | str]] = []
    builders = _model_builders(random_state=random_state)
    unique_classes = np.unique(y)
    average = "binary" if len(unique_classes) == 2 else "macro"
    for model_name, model in builders.items():
        model_clone = model
        train_start = perf_counter()
        model_clone.fit(X_train, y_train)
        train_duration = perf_counter() - train_start

        predict_start = perf_counter()
        y_pred = model_clone.predict(X_test)
        predict_duration = perf_counter() - predict_start

        results.append(
            {
                "model": model_name,
                "train_time": train_duration,
                "predict_time": predict_duration,
                "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
                "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average=average, zero_division=0),
                "accuracy": accuracy_score(y_test, y_pred),
                "cohen_kappa": cohen_kappa_score(y_test, y_pred),
                "mcc": matthews_corrcoef(y_test, y_pred),
            }
        )
    return results


def _display_ml_results(results: list[dict[str, float | str]]) -> None:
    if not results:
        print("No evaluation results to display.")
        return
    df = pd.DataFrame(results)
    ordered_columns = [
        "dataset",
        "alpha",
        "model",
        "train_time",
        "predict_time",
        "precision",
        "recall",
        "f1_score",
        "accuracy",
        "cohen_kappa",
        "mcc",
    ]
    df = df[ordered_columns].sort_values(["dataset", "alpha", "model"]).reset_index(drop=True)
    print("\nModel evaluation summary:")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def _run_ml_pipeline_for_dataset(
    dataset_name: str,
    anonymized_batches: dict[float, tuple[np.ndarray, np.ndarray]],
) -> list[dict[str, float | str]]:
    collected_rows: list[dict[str, float | str]] = []
    for alpha in sorted(anonymized_batches.keys()):
        print(f"Evaluating ML models for {dataset_name} | alpha={alpha:.2f}")
        X_anon, y_anon = anonymized_batches[alpha]
        try:
            alpha_results = _evaluate_models(X_anon, y_anon)
        except Exception as exc:
            print(f"Model training failed for {dataset_name} | alpha={alpha:.2f}: {exc}")
            continue
        for row in alpha_results:
            row["alpha"] = alpha
            row["dataset"] = dataset_name
            collected_rows.append(row)
    return collected_rows


def _run_mia_pipeline_for_dataset(
    dataset_name: str,
    anonymized_batches: dict[float, tuple[np.ndarray, np.ndarray]],
    victim_models: list[str],
) -> list[dict[str, float | str]]:
    mia_rows: list[dict[str, float | str]] = []
    for alpha in sorted(anonymized_batches.keys()):
        X_anon, y_anon = anonymized_batches[alpha]
        for victim_name in victim_models:
            print(f"Evaluating MIA for {dataset_name} | alpha={alpha:.2f} | victim={victim_name}")
            try:
                mia_result = evaluate_mia(
                    X_anon,
                    y_anon,
                    victim_model_name=victim_name,
                    random_state=RANDOM_STATE,
                )
            except Exception as exc:
                print(
                    f"MIA failed for {dataset_name} | alpha={alpha:.2f} | victim={victim_name}: {exc}"
                )
                continue
            mia_rows.append(
                {
                    "dataset": dataset_name,
                    "alpha": alpha,
                    "victim_model": mia_result.victim_model,
                    "auc": mia_result.auc,
                    "accuracy": mia_result.accuracy,
                    "roc_points": (mia_result.fpr, mia_result.tpr, mia_result.thresholds),
                }
            )
    return mia_rows


def _display_mia_results(results: list[dict[str, float | str]]) -> None:
    if not results:
        print("No membership inference results to display.")
        return
    printable_rows = [{k: v for k, v in row.items() if k != "roc_points"} for row in results]
    df = pd.DataFrame(printable_rows)
    ordered_columns = ["dataset", "alpha", "victim_model", "auc", "accuracy"]
    df = df[ordered_columns].sort_values(["dataset", "alpha", "victim_model"]).reset_index(drop=True)
    print("\nMembership inference summary:")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("AUC ~0.50 => robust; AUC >0.70 => potential leakage.")


def _display_shift_results(results: list[dict[str, float | str]]) -> None:
    if not results:
        print("No distribution-shift results to display.")
        return
    df = pd.DataFrame(results)
    ordered_columns = [
        "dataset",
        "alpha",
        "kl_mean",
        "kl_median",
        "wasserstein_mean",
        "wasserstein_median",
    ]
    df = df[ordered_columns].sort_values(["dataset", "alpha"]).reset_index(drop=True)
    print("\nDistribution-shift summary (per-feature KL/Wasserstein; mean/median):")
    print(df.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def main() -> None:
    args = _parse_args()
    alphas = args.alphas if args.alphas else ALPHA_VALUES
    if not alphas:
        raise ValueError("Alpha list cannot be empty.")
    alphas = sorted(set(alphas))
    k_clusters = args.k

    original_dataset_path = _resolve_path(RELATIVE_ORIGINAL_PATH)
    binary_dataset_path = _resolve_path(RELATIVE_BINARY_PATH)

    load_and_analyze_dataset(
        dataset_path=original_dataset_path,
        target_column="y",
        dataset_name="Original multi-class dataset",
        class_names=ORIGINAL_CLASS_NAMES,
    )

    ensured_binary_path = ensure_binary_dataset(
        original_dataset_path=original_dataset_path,
        binary_dataset_path=binary_dataset_path,
        target_column="y",
        positive_class=1,
    )

    load_and_analyze_dataset(
        dataset_path=ensured_binary_path,
        target_column="y",
        dataset_name="Binary seizure dataset",
        class_names=BINARY_CLASS_NAMES,
    )

    load_and_analyze_dataset(
        dataset_path=_resolve_path("data/Synthetic_patient-HealthCare-Monitoring_dataset.csv"),
        target_column="Temperature Alert",
        dataset_name="Synthetic healthcare dataset (Temperature Alert)",
        class_names=None,
    )

    load_and_analyze_dataset(
        dataset_path=_resolve_path("data/patients_data_with_alerts.csv"),
        target_column="Temperature Alert",
        dataset_name="D4",
        class_names=None,
    )

    dataset_runs = [
        # D2
        {
            "name": "Original multi-class dataset",
            "path": original_dataset_path,
            "class_names": ORIGINAL_CLASS_NAMES,
            "target_column": "y",
        },
        # d1
        {
            "name": "Binary seizure dataset",
            "path": ensured_binary_path,
            "class_names": BINARY_CLASS_NAMES,
            "target_column": "y",
        },
        # D3
        {
            "name": "Synthetic healthcare dataset (Temperature Alert)",
            "path": _resolve_path("data/Synthetic_patient-HealthCare-Monitoring_dataset.csv"),
            "class_names": None,
            "target_column": "Temperature Alert",
        },
        # D4
        {
            "name": "D4",
            "path": _resolve_path("data/patients_data_with_alerts.csv"),
            "class_names": None,
            "target_column": "Temperature Alert",
        },
    ]

    ml_results: list[dict[str, float | str]] = []
    mia_results: list[dict[str, float | str]] = []
    shift_results: list[dict[str, float | str]] = []

    for config in dataset_runs:
        print(f"\nRunning anonymization + training for {config['name']}...")
        X_data, y_data = _load_dataset_numpy(config["path"], target_column=config.get("target_column", "y"))
        anonymized_batches = _parallel_anonymizations(
            X_data,
            y_data,
            alphas=alphas,
            k=k_clusters,
        )

        missing_alphas = set(alphas) - set(anonymized_batches.keys())
        for alpha in missing_alphas:
            print(f"Alpha {alpha} missing from parallel stage; retrying sequentially...")
            anonymized_batches[alpha] = anonimization_clustering(X_data, y_data, k_clusters, alpha)

        if args.mode in {"stats", "both"} and not args.no_stats:
            stats_bins = int(args.stats_bins)
            if stats_bins < 2:
                raise ValueError("--stats-bins must be >= 2")
            for alpha in sorted(anonymized_batches.keys()):
                X_anon, _ = anonymized_batches[alpha]
                try:
                    kl_values, wass_values = per_feature_shift_metrics(
                        X_reference=X_data,
                        X_candidate=X_anon,
                        bins=stats_bins,
                    )
                    summary = summarize_shift_metrics(kl_values, wass_values)
                except Exception as exc:
                    print(f"Shift stats failed for {config['name']} | alpha={alpha:.2f}: {exc}")
                    continue
                shift_results.append(
                    {
                        "dataset": config["name"],
                        "alpha": alpha,
                        "kl_mean": summary.kl_mean,
                        "kl_median": summary.kl_median,
                        "wasserstein_mean": summary.wasserstein_mean,
                        "wasserstein_median": summary.wasserstein_median,
                    }
                )

        if args.mode in {"ml", "both"}:
            ml_results.extend(
                _run_ml_pipeline_for_dataset(
                    dataset_name=config["name"],
                    anonymized_batches=anonymized_batches,
                )
            )

        if args.mode in {"mia", "both"}:
            mia_results.extend(
                _run_mia_pipeline_for_dataset(
                    dataset_name=config["name"],
                    anonymized_batches=anonymized_batches,
                    victim_models=args.victim_models,
                )
            )

    if ml_results:
        _display_ml_results(ml_results)

    if mia_results:
        _display_mia_results(mia_results)

    if shift_results:
        _display_shift_results(shift_results)


if __name__ == "__main__":
    main()
