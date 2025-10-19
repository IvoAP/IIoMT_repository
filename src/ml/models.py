from typing import Dict
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def suggest_model_params(model_name: str, trial: optuna.Trial, is_binary: bool) -> Dict[str, object]:
    if model_name == "SVM":
        return {
            "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
            "gamma": trial.suggest_float("gamma", 1e-4, 1e-1, log=True),
        }
    if model_name == "Random Forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 400, step=25),
            "max_depth": trial.suggest_int("max_depth", 6, 40, step=2),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }
    if model_name == "XGBoost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 150, 350, step=25),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if model_name == "MLP":
        return {
            "hidden_layer_sizes": trial.suggest_categorical(
                "hidden_layer_sizes",
                [(128,), (256,), (128, 64), (200, 100), (256, 128)],
            ),
            "alpha": trial.suggest_float("alpha", 1e-5, 1e-2, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        }
    raise ValueError(f"Unsupported model: {model_name}")


def instantiate_model(
    model_name: str,
    params: Dict[str, object],
    is_binary: bool,
    num_classes: int,
    *,
    random_state: int,
):
    if model_name == "SVM":
        return SVC(
            kernel="rbf",
            probability=True,
            C=params.get("C", 1.0),
            gamma=params.get("gamma", "scale"),
        )
    if model_name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth"),
            min_samples_split=params.get("min_samples_split", 2),
            min_samples_leaf=params.get("min_samples_leaf", 1),
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "XGBoost":
        base_params = {
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_estimators": params.get("n_estimators", 200),
            "max_depth": params.get("max_depth", 6),
            "learning_rate": params.get("learning_rate", 0.1),
            "subsample": params.get("subsample", 0.8),
            "colsample_bytree": params.get("colsample_bytree", 0.8),
            "reg_lambda": params.get("reg_lambda", 1.0),
        }
        if is_binary:
            base_params["objective"] = "binary:logistic"
        else:
            base_params["objective"] = "multi:softprob"
            base_params["num_class"] = num_classes
        return XGBClassifier(**base_params)
    if model_name == "MLP":
        return MLPClassifier(
            hidden_layer_sizes=params.get("hidden_layer_sizes", (100, 50)),
            alpha=params.get("alpha", 1e-4),
            learning_rate_init=params.get("learning_rate_init", 1e-3),
            max_iter=400,
            random_state=random_state,
        )
    raise ValueError(f"Unsupported model: {model_name}")
