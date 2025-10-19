import argparse
import os
from pathlib import Path
from typing import List, Tuple
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from constraints import BINARY_CLASS_NAMES, ORIGINAL_CLASS_NAMES
from file_utils import (
    binary_path,
    ensure_binary_dataset,
    generate_binary_dataset,
    load_and_analyze_dataset,
    load_and_split_data,
    path,
)
from ml.training import train_and_evaluate


RESULTS_DIR = Path("results")
MI_DIR = RESULTS_DIR / "mi"
RANDOM_STATE = 42


def ensure_results_dir() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)


def ensure_mi_dir() -> None:
    ensure_results_dir()
    MI_DIR.mkdir(exist_ok=True)


def prepare_binary_dataset(regenerate: bool) -> str:
    if regenerate or not os.path.exists(binary_path):
        return generate_binary_dataset(original_dataset_path=path, binary_dataset_path=binary_path)
    ensure_binary_dataset(path, binary_path)
    return binary_path


def get_dataset_metadata(dataset_key: str) -> Tuple[str, str, dict[int, str]]:
    if dataset_key == "five":
        return path, "Epileptic Seizure Recognition (5-class)", ORIGINAL_CLASS_NAMES
    if dataset_key == "binary":
        return binary_path, "Epileptic Seizure Recognition (Binary)", BINARY_CLASS_NAMES
    raise ValueError(f"Unsupported dataset key: {dataset_key}")


def resolve_dataset_path(dataset_key: str, regenerate_binary: bool) -> Tuple[str, str, dict[int, str]]:
    dataset_path, dataset_name, class_names = get_dataset_metadata(dataset_key)
    if dataset_key == "binary":
        dataset_path = prepare_binary_dataset(regenerate_binary)
    else:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Original dataset not found at {dataset_path}")
    return dataset_path, dataset_name, class_names


def display_dataset(dataset_key: str, regenerate_binary: bool) -> None:
    try:
        dataset_path, dataset_name, class_names = resolve_dataset_path(dataset_key, regenerate_binary)
    except FileNotFoundError as exc:
        print(str(exc))
        return

    load_and_analyze_dataset(
        dataset_path=dataset_path,
        target_column="y",
        dataset_name=dataset_name,
        class_names=class_names,
    )


def load_dataset_frames(dataset_key: str, regenerate_binary: bool) -> Tuple[str, pd.DataFrame, pd.Series]:
    dataset_path, dataset_name, _ = resolve_dataset_path(dataset_key, regenerate_binary)
    X, y = load_and_split_data(dataset_path, target_column="y")
    return dataset_name, X, y


def compute_mutual_information(
    dataset_key: str,
    dataset_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    top_n: int,
    alpha: float,
) -> None:
    ensure_mi_dir()
    print(f"\nMutual information scores for {dataset_name} ({dataset_key}) dataset:")

    X_processed = X.copy()
    categorical_cols = X_processed.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        X_processed[col] = X_processed[col].astype("category").cat.codes

    scores = mutual_info_classif(X_processed.values, y.values, random_state=RANDOM_STATE)
    mi_df = pd.DataFrame({"feature": X.columns, "mutual_information": scores})
    mi_df.sort_values(by="mutual_information", ascending=False, inplace=True)

    weights = mi_df["mutual_information"].clip(lower=0).pow(alpha)
    total_weight = weights.sum()
    if total_weight > 0:
        mi_df["weight_alpha"] = weights / total_weight
    else:
        mi_df["weight_alpha"] = 0.0
    mi_df["alpha"] = alpha

    display_count = min(top_n, len(mi_df))
    for _, row in mi_df.head(display_count).iterrows():
        print(
            f"  {row['feature']}: MI={row['mutual_information']:.6f} | weight(alpha={alpha:g})={row['weight_alpha']:.6f}"
        )

    output_path = MI_DIR / f"{dataset_key}_mutual_information.csv"
    mi_df.to_csv(output_path, index=False)
    print(f"Full mutual information scores saved to {output_path}")


def parse_dataset_selection(selection: str) -> List[str]:
    if selection == "both":
        return ["five", "binary"]
    return [selection]


def handle_view_command(args: argparse.Namespace) -> None:
    dataset_keys = parse_dataset_selection(args.dataset)
    regenerate_binary = not args.keep_binary
    for dataset_key in dataset_keys:
        regen_flag = regenerate_binary if dataset_key == "binary" else False
        display_dataset(dataset_key, regen_flag)


def handle_mutual_information_command(args: argparse.Namespace) -> None:
    dataset_keys = parse_dataset_selection(args.dataset)
    regenerate_binary = not args.keep_binary
    for dataset_key in dataset_keys:
        regen_flag = regenerate_binary if dataset_key == "binary" else False
        if args.show_summary:
            display_dataset(dataset_key, regen_flag)
            regen_flag = False if dataset_key == "binary" else regen_flag

        try:
            dataset_name, X, y = load_dataset_frames(dataset_key, regen_flag)
        except FileNotFoundError as exc:
            print(str(exc))
            continue

        compute_mutual_information(
            dataset_key=dataset_key,
            dataset_name=dataset_name,
            X=X,
            y=y,
            top_n=args.top,
            alpha=args.alpha,
        )


def handle_train_command(args: argparse.Namespace) -> None:
    dataset_keys = parse_dataset_selection(args.dataset)
    regenerate_binary = not args.keep_binary
    verbose = not args.quiet

    for dataset_key in dataset_keys:
        regen_flag = regenerate_binary if dataset_key == "binary" else False
        try:
            dataset_path, dataset_name, _ = resolve_dataset_path(dataset_key, regen_flag)
        except FileNotFoundError as exc:
            print(str(exc))
            continue

        results_path = train_and_evaluate(
            dataset_key=dataset_key,
            dataset_path=dataset_path,
            dataset_label=dataset_name,
            is_binary=(dataset_key == "binary"),
            classical_trials=args.classical_trials,
            dnn_trials=args.dnn_trials,
            verbose=verbose,
        )
        print(f"{dataset_name} metrics saved to {results_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Epileptic Seizure Recognition dataset utilities",
    )
    subparsers = parser.add_subparsers(dest="command")

    view_parser = subparsers.add_parser(
        "view",
        help="Display dataset statistics for the five-class or binary dataset",
    )
    view_parser.add_argument(
        "dataset",
        choices=["five", "binary", "both"],
        help="Which dataset summary to display",
    )
    view_parser.add_argument(
        "--keep-binary",
        action="store_true",
        help="Do not regenerate the binary dataset before viewing it",
    )

    mi_parser = subparsers.add_parser(
        "mi",
        help="Compute mutual information between features and target",
    )
    mi_parser.add_argument(
        "--dataset",
        choices=["five", "binary", "both"],
        default="both",
        help="Dataset(s) to process",
    )
    mi_parser.add_argument(
        "--top",
        type=int,
        default=10,
        metavar="N",
        help="Number of top features to display",
    )
    mi_parser.add_argument(
        "--alpha",
        type=float,
        default=2.0,
        help="Exponent for adaptive weight normalization (alpha > 1)",
    )
    mi_parser.add_argument(
        "--keep-binary",
        action="store_true",
        help="Do not regenerate the binary dataset before computing MI",
    )
    mi_parser.add_argument(
        "--show-summary",
        action="store_true",
        help="Also display dataset summary before computing MI",
    )

    train_parser = subparsers.add_parser(
        "train",
        help="Run Optuna-tuned experiments and save evaluation metrics",
    )
    train_parser.add_argument(
        "dataset",
        choices=["five", "binary", "both"],
        help="Which dataset(s) to train",
    )
    train_parser.add_argument(
        "--keep-binary",
        action="store_true",
        help="Do not regenerate the binary dataset before training",
    )
    train_parser.add_argument(
        "--classical-trials",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of Optuna trials for classical models",
    )
    train_parser.add_argument(
        "--dnn-trials",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of Optuna trials for the DNN",
    )
    train_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress tuning progress logs",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "view":
        handle_view_command(args)
        return

    if args.command == "mi":
        handle_mutual_information_command(args)
        return

    if args.command == "train":
        handle_train_command(args)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
