from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from file_utils import load_and_split_data
from ml.models import instantiate_model, suggest_model_params

DEFAULT_RANDOM_STATE = 42
DEFAULT_RESULTS_DIR = Path("results") / "ml"
np.random.seed(DEFAULT_RANDOM_STATE)
tf.random.set_seed(DEFAULT_RANDOM_STATE)
optuna.logging.set_verbosity(optuna.logging.WARNING)
tf.get_logger().setLevel("ERROR")


def _log(message: str, *, verbose: bool) -> None:
    if verbose:
        print(message)


def preprocess_features(X: pd.DataFrame) -> pd.DataFrame:
    X_processed = X.copy()
    categorical_cols = X_processed.select_dtypes(include=["object", "category"]).columns
    for col in categorical_cols:
        X_processed[col] = X_processed[col].astype("category").cat.codes
    return X_processed


def load_dataset(
    dataset_path: str,
    *,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    is_binary: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, y = load_and_split_data(dataset_path, target_column="y")
    X_processed = preprocess_features(X)
    y_array = y.to_numpy()
    if not is_binary:
        y_array = y_array - y_array.min()
    return train_test_split(
        X_processed.to_numpy(dtype=np.float32),
        y_array,
        test_size=test_size,
        stratify=y_array,
        random_state=random_state,
    )


def scale_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled.astype(np.float32), X_test_scaled.astype(np.float32)


def evaluate_predictions(y_true, y_pred, average: str) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_true, y_pred, average=average, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average=average, zero_division=0),
    }


def tune_classical_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    average: str,
    is_binary: bool,
    num_classes: int,
    n_trials: int,
    random_state: int,
    verbose: bool,
) -> Tuple[object, Dict[str, object], float, float]:
    _log(
        f"    -> Tuning {model_name} with {n_trials} Optuna trials (dataset={'binary' if is_binary else 'five-class'})",
        verbose=verbose,
    )
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def objective(trial: optuna.Trial) -> float:
        params = suggest_model_params(model_name, trial, is_binary)
        scores: List[float] = []
        try:
            for _, (train_idx, valid_idx) in enumerate(cv.split(X_train, y_train), start=1):
                fold_model = instantiate_model(
                    model_name,
                    params,
                    is_binary,
                    num_classes,
                    random_state=random_state,
                )
                fold_model.fit(X_train[train_idx], y_train[train_idx])
                preds = fold_model.predict(X_train[valid_idx])
                score = f1_score(y_train[valid_idx], preds, average=average, zero_division=0)
                scores.append(score)
        except Exception:
            return 0.0

        mean_score = float(np.mean(scores)) if scores else 0.0
        _log(
            f"       Trial {trial.number + 1}/{n_trials}: val_f1={mean_score:.4f} (cv over {len(scores)} folds)",
            verbose=verbose,
        )
        return mean_score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    best_params = study.best_trial.params

    best_model = instantiate_model(
        model_name,
        best_params,
        is_binary,
        num_classes,
        random_state=random_state,
    )
    start_train = time.perf_counter()
    best_model.fit(X_train, y_train)
    training_time = time.perf_counter() - start_train

    _log(
        f"    -> Best {model_name} val_f1={study.best_value:.4f} with params={json.dumps(best_params)}",
        verbose=verbose,
    )
    return best_model, best_params, training_time, study.best_value


def build_dnn(
    input_dim: int,
    num_classes: int,
    is_binary: bool,
    params: Dict[str, object],
) -> tf.keras.Model:
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(input_dim,)))
    model.add(tf.keras.layers.Dense(params["units1"], activation="relu"))
    model.add(tf.keras.layers.Dropout(params["dropout"]))
    model.add(tf.keras.layers.Dense(params["units2"], activation="relu"))
    model.add(tf.keras.layers.Dropout(params["dropout"]))
    if is_binary:
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        loss = "binary_crossentropy"
    else:
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        loss = "sparse_categorical_crossentropy"

    optimizer = tf.keras.optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model


def tune_dnn_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    average: str,
    is_binary: bool,
    n_trials: int,
    random_state: int,
    verbose: bool,
) -> Tuple[tf.keras.Model, Dict[str, object], float, float]:
    _log(
        f"    -> Tuning Deep Neural Network with {n_trials} Optuna trials (dataset={'binary' if is_binary else 'five-class'})",
        verbose=verbose,
    )
    X_sub, X_valid, y_sub, y_valid = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=random_state,
    )

    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))

    def objective(trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()
        params = {
            "units1": trial.suggest_int("units1", 128, 512, step=64),
            "units2": trial.suggest_int("units2", 64, 256, step=32),
            "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 25, 60),
        }
        model = build_dnn(input_dim, num_classes, is_binary, params)
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        model.fit(
            X_sub,
            y_sub,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            verbose=0,
            validation_data=(X_valid, y_valid),
            callbacks=[early_stop],
        )
        probabilities = model.predict(X_valid, verbose=0)
        if is_binary:
            preds = (probabilities >= 0.5).astype(int).flatten()
        else:
            preds = probabilities.argmax(axis=1)
        score = f1_score(y_valid, preds, average=average, zero_division=0)
        _log(
            f"       Trial {trial.number + 1}/{n_trials}: val_f1={score:.4f}",
            verbose=verbose,
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)
    best_params = study.best_trial.params

    tf.keras.backend.clear_session()
    best_model = build_dnn(input_dim, num_classes, is_binary, best_params)
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    start_train = time.perf_counter()
    best_model.fit(
        X_train,
        y_train,
        epochs=best_params["epochs"],
        batch_size=best_params["batch_size"],
        verbose=0,
        validation_split=0.1,
        callbacks=[early_stop],
    )
    training_time = time.perf_counter() - start_train

    _log(
        f"    -> Best DNN val_f1={study.best_value:.4f} with params={json.dumps(best_params)}",
        verbose=verbose,
    )
    return best_model, best_params, training_time, study.best_value


def ensure_results_dir(results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir


def save_results(dataset_key: str, results: List[Dict[str, object]], *, results_dir: Path) -> Path:
    ensure_results_dir(results_dir)
    df = pd.DataFrame(results)
    output_path = results_dir / f"{dataset_key}_results.csv"
    df.sort_values(by="accuracy", ascending=False).to_csv(output_path, index=False)
    return output_path


def train_and_evaluate(
    dataset_key: str,
    dataset_path: str,
    *,
    dataset_label: str | None = None,
    is_binary: bool,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    classical_trials: int | None = None,
    dnn_trials: int | None = None,
    verbose: bool = True,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Path:
    label = dataset_label or dataset_key
    print(f"\nTraining models for {label} dataset")

    X_train, X_test, y_train, y_test = load_dataset(
        dataset_path,
        is_binary=is_binary,
        random_state=random_state,
    )
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)

    average = "binary" if is_binary else "macro"
    num_classes = len(np.unique(y_train))

    classical_trials_count = classical_trials if classical_trials is not None else (12 if is_binary else 15)
    dnn_trials_count = dnn_trials if dnn_trials is not None else (8 if is_binary else 10)

    results: List[Dict[str, object]] = []
    for model_name in ["SVM", "Random Forest", "XGBoost", "MLP"]:
        tuned_model, best_params, training_time, best_validation = tune_classical_model(
            model_name,
            X_train_scaled,
            y_train,
            average=average,
            is_binary=is_binary,
            num_classes=num_classes,
            n_trials=classical_trials_count,
            random_state=random_state,
            verbose=verbose,
        )
        start_pred = time.perf_counter()
        y_pred = tuned_model.predict(X_test_scaled)
        prediction_time = time.perf_counter() - start_pred

        metrics = evaluate_predictions(y_test, y_pred, average)
        metrics.update(
            {
                "model": model_name,
                "training_time": training_time,
                "prediction_time": prediction_time,
                "best_validation_score": best_validation,
                "best_params": json.dumps(best_params),
            }
        )
        results.append(metrics)
        print(
            f"  {model_name}: accuracy={metrics['accuracy']:.4f}, f1={metrics['f1_score']:.4f}, val_f1={best_validation:.4f}"
        )

    dnn_model, dnn_params, dnn_train_time, dnn_val = tune_dnn_model(
        X_train_scaled,
        y_train,
        average=average,
        is_binary=is_binary,
        n_trials=dnn_trials_count,
        random_state=random_state,
        verbose=verbose,
    )
    start_pred = time.perf_counter()
    probabilities = dnn_model.predict(X_test_scaled, verbose=0)
    if is_binary:
        y_pred = (probabilities >= 0.5).astype(int).flatten()
    else:
        y_pred = probabilities.argmax(axis=1)
    prediction_time = time.perf_counter() - start_pred

    dnn_metrics = evaluate_predictions(y_test, y_pred, average)
    dnn_metrics.update(
        {
            "model": "Deep Neural Network",
            "training_time": dnn_train_time,
            "prediction_time": prediction_time,
            "best_validation_score": dnn_val,
            "best_params": json.dumps(dnn_params),
        }
    )
    results.append(dnn_metrics)
    print(
        f"  Deep Neural Network: accuracy={dnn_metrics['accuracy']:.4f}, f1={dnn_metrics['f1_score']:.4f}, val_f1={dnn_val:.4f}"
    )

    results_key = dataset_key if dataset_key != "five" else "five_class"
    results_path = save_results(results_key, results, results_dir=results_dir)
    print(f"Results saved to {results_path} (any existing file was overwritten)")
    return results_path


__all__ = ["train_and_evaluate"]
