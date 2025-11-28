"""Membership inference attack utilities."""

# Membership Inference Attacks (MIA) try to determine if a record was part of
# the victim model's training data by exploiting distributional differences in
# the outputs (higher confidence, lower entropy or lower loss) for seen versus
# unseen samples. A classifier is trained to separate "members" from
# "non-members", and AUC close to 0.5 means the model is robust, while larger
# AUC indicates possible leakage.

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

VictimModelName = Literal[
    "knn",
    "mlp",
    "decision_tree",
    "gaussian_nb",
    "logistic_regression",
    "random_forest",
]


@dataclass
class MIAResult:
    victim_model: VictimModelName
    auc: float
    accuracy: float
    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


def _get_victim_model(
    model_name: VictimModelName,
    random_state: int,
) -> Pipeline | DecisionTreeClassifier | RandomForestClassifier:
    if model_name == "knn":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        )
    if model_name == "mlp":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=random_state),
                ),
            ]
        )
    if model_name == "decision_tree":
        return DecisionTreeClassifier(random_state=random_state)
    if model_name == "gaussian_nb":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", GaussianNB()),
            ]
        )
    if model_name == "logistic_regression":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, multi_class="auto", random_state=random_state),
                ),
            ]
        )
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported victim model: {model_name}")


def train_victim_model(
    X: np.ndarray,
    y: np.ndarray,
    model_name: VictimModelName = "mlp",
    random_state: int = 42,
):
    model = _get_victim_model(model_name, random_state)
    model.fit(X, y)
    return model


def compute_membership_signals(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    probas = model.predict_proba(X)
    max_proba = probas.max(axis=1)
    entropy = -(probas * np.log(probas + 1e-12)).sum(axis=1)
    true_probs = probas[np.arange(len(y)), y]
    loss = -np.log(true_probs + 1e-12)
    return np.column_stack((max_proba, entropy, loss))


def train_membership_attacker(
    member_scores: np.ndarray,
    non_member_scores: np.ndarray,
    random_state: int = 42,
):
    X_attack = np.vstack((member_scores, non_member_scores))
    y_attack = np.concatenate(
        (
            np.ones(len(member_scores), dtype=int),
            np.zeros(len(non_member_scores), dtype=int),
        )
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_attack,
        y_attack,
        test_size=0.3,
        random_state=random_state,
        stratify=y_attack,
    )

    attacker = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=1000, random_state=random_state),
            ),
        ]
    )
    attacker.fit(X_train, y_train)
    y_pred = attacker.predict(X_test)
    y_scores = attacker.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_scores)
    accuracy = accuracy_score(y_test, y_pred)
    fpr, tpr, thresholds = roc_curve(y_test, y_scores)

    metrics = {
        "auc": auc,
        "accuracy": accuracy,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }

    return attacker, metrics


def evaluate_mia(
    X: np.ndarray,
    y: np.ndarray,
    victim_model_name: VictimModelName = "mlp",
    random_state: int = 42,
) -> MIAResult:
    X_victim, X_shadow, y_victim, y_shadow = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=random_state,
        stratify=y,
    )

    victim_model = train_victim_model(
        X_victim,
        y_victim,
        model_name=victim_model_name,
        random_state=random_state,
    )

    member_scores = compute_membership_signals(victim_model, X_victim, y_victim)
    non_member_scores = compute_membership_signals(victim_model, X_shadow, y_shadow)

    _, attacker_metrics = train_membership_attacker(
        member_scores,
        non_member_scores,
        random_state=random_state,
    )

    return MIAResult(
        victim_model=victim_model_name,
        auc=attacker_metrics["auc"],
        accuracy=attacker_metrics["accuracy"],
        fpr=attacker_metrics["fpr"],
        tpr=attacker_metrics["tpr"],
        thresholds=attacker_metrics["thresholds"],
    )
