"""Shared workflow utilities for the dual-notebook ensemble experiments.

This module keeps the training, calibration, random search, and ensemble
weight-search logic in one place so that the notebooks stay readable while
sharing one deterministic experimental protocol.
"""

from __future__ import annotations

import json
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Concatenate, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from torch.utils.data import DataLoader, TensorDataset

from cv_preprocessing import (
    calibrate_platt,
    compute_metrics,
    load_outcomes,
    load_raw_data,
    prepare_data_cv,
    print_cv_summary,
    random_oversample_indices,
    setup_cv_splits,
    youden_threshold,
)
from paths import RESULTS_DIR

warnings.filterwarnings("ignore")

SEARCH_SEED = 42
DEFAULT_WEIGHT_STEP = 0.05
DEFAULT_MIN_SPECIFICITY = 0.60
DISPLAY_METRICS = ("AUROC", "AUPRC", "F1", "Sensitivity", "Specificity")

DEFAULT_LSTM_SEARCH_SPACE = {
    "units": [[64], [128], [64, 32], [128, 64], [64, 48, 32]],
    "dropout": [0.3, 0.4, 0.5],
    "lr": [1e-2, 1e-3, 5e-4, 1e-4],
    "batch_size": [32],
    "max_epochs": [200],
    "patience": [20],
}

DEFAULT_GRUD_SEARCH_SPACE = {
    "hidden_size": [32, 64, 128],
    "dropout": [0.2, 0.3, 0.4, 0.5],
    "lr": [1e-2, 1e-3, 5e-4, 1e-4],
    "batch_size": [64],
    "weight_decay": [1e-4],
    "max_epochs": [100],
    "patience": [15],
}

DEFAULT_SVM_SEARCH_SPACE = {
    "kernel": ["linear"],
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
}

DEFAULT_LR_SEARCH_SPACE = {
    "C": [0.01, 0.1, 1.0, 10.0, 100.0],
    "penalty": ["l2"],
    "solver": ["liblinear"],
    "max_iter": [5000],
}


def resolve_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )


def configure_tensorflow_runtime(use_gpu: bool = True) -> dict:
    """Optionally disable TensorFlow GPU devices before the first TF workload."""
    gpu_devices = tf.config.list_physical_devices("GPU")
    if not use_gpu and gpu_devices:
        try:
            tf.config.set_visible_devices([], "GPU")
            gpu_devices = []
        except RuntimeError as exc:
            print(f"TensorFlow runtime already initialized; could not disable GPU: {exc}")
            gpu_devices = tf.config.list_physical_devices("GPU")

    active_gpu_devices = tf.config.list_logical_devices("GPU") if gpu_devices else []
    return {
        "requested_gpu": bool(use_gpu),
        "physical_gpu_count": len(gpu_devices),
        "logical_gpu_count": len(active_gpu_devices),
        "active_device": "GPU" if active_gpu_devices else "CPU",
    }


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    tf.keras.utils.set_random_seed(seed)


def print_section(title: str, char: str = "=") -> None:
    print()
    print(char * 70)
    print(title)
    print(char * 70)


def format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def summary_to_frame(
    name: str,
    summary: dict,
    metrics: tuple[str, ...] = DISPLAY_METRICS,
) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        rows.append(
            {
                "Model": name,
                "Metric": metric,
                "Mean": summary[metric]["mean"],
                "Std": summary[metric]["std"],
            }
        )
    return pd.DataFrame(rows)


def to_python(value):
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, list):
        return [to_python(v) for v in value]
    if isinstance(value, tuple):
        return [to_python(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(to_python(payload), handle, indent=2)


def make_search_split(all_ids, all_y, seed: int = SEARCH_SEED):
    train_ids, temp_ids, y_train, y_temp = train_test_split(
        all_ids,
        all_y,
        test_size=0.20,
        stratify=all_y,
        random_state=seed,
    )
    val_ids, test_ids, _, _ = train_test_split(
        temp_ids,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=seed,
    )
    return train_ids, val_ids, test_ids


def load_workflow_context():
    df_static, df_ts, ts_cols = load_raw_data()
    all_record_ids, y_all = load_outcomes(df_static)
    folds, _ = setup_cv_splits(all_record_ids, y_all, held_out_test_ids=None)
    return {
        "df_static": df_static,
        "df_ts": df_ts,
        "ts_cols": ts_cols,
        "all_record_ids": all_record_ids,
        "y_all": y_all,
        "folds": folds,
        "device": resolve_device(),
        "results_dir": RESULTS_DIR / "dual_ensemble_notebooks",
    }


def prepare_search_dataset(context: dict, seed: int = SEARCH_SEED) -> dict:
    train_ids, val_ids, test_ids = make_search_split(
        context["all_record_ids"], context["y_all"], seed=seed
    )
    data = prepare_data_cv(
        train_ids,
        val_ids,
        test_ids,
        context["df_ts"],
        context["df_static"],
        context["y_all"],
        context["all_record_ids"],
        context["ts_cols"],
    )
    idx_bal = random_oversample_indices(data["y_train"], seed=seed)
    return {
        "train_ids": train_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "data": data,
        "idx_bal": idx_bal,
    }


def sample_cfg(space: dict, rng: random.Random) -> dict:
    return {key: deepcopy(rng.choice(values)) for key, values in space.items()}


def build_lstm_model(ts_shape, static_dim, layer_units, dropout_rate, learning_rate):
    ts_input = Input(shape=ts_shape, name="ts_input")
    x = LSTM(layer_units[0], return_sequences=False)(ts_input)
    x = Dropout(dropout_rate)(x)

    static_input = Input(shape=(static_dim,), name="static_input")
    combined = Concatenate()([x, static_input])

    for units in layer_units[1:]:
        combined = Dense(units, activation="relu")(combined)
        combined = Dropout(dropout_rate)(combined)

    output = Dense(1, activation="sigmoid")(combined)
    model = Model(inputs=[ts_input, static_input], outputs=output)
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


class GRUDCell(nn.Module):
    def __init__(self, input_size, hidden_size, x_mean):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.register_buffer("x_mean", x_mean)
        self.w_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.b_gamma_x = nn.Parameter(torch.zeros(input_size))
        self.W_gamma_h = nn.Linear(input_size, hidden_size, bias=True)
        self.gru_cell = nn.GRUCell(input_size * 2, hidden_size)
        nn.init.zeros_(self.W_gamma_h.weight)
        nn.init.zeros_(self.W_gamma_h.bias)

    def forward(self, x, mask, delta, h):
        gamma_x = torch.exp(-F.relu(self.w_gamma_x * delta + self.b_gamma_x))
        x_tilde = mask * x + (1 - mask) * (
            gamma_x * x + (1 - gamma_x) * self.x_mean
        )
        gamma_h = torch.exp(-F.relu(self.W_gamma_h(delta)))
        return self.gru_cell(torch.cat([x_tilde, mask], dim=-1), gamma_h * h)


class GRUD(nn.Module):
    def __init__(self, ts_size, hidden_size, static_size, x_mean, dropout=0.3):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = GRUDCell(ts_size, hidden_size, x_mean)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size + static_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x_ts, mask, delta, x_static):
        hidden = torch.zeros(x_ts.size(0), self.hidden_size, device=x_ts.device)
        for step in range(x_ts.size(1)):
            hidden = self.cell(x_ts[:, step], mask[:, step], delta[:, step], hidden)
        return self.classifier(torch.cat([hidden, x_static], dim=-1)).squeeze(-1)


@torch.no_grad()
def grud_predict(model, x_ts, x_static, mask, delta, device, batch_size=64):
    model.eval()
    probs = []
    n_rows = len(x_ts)
    for start in range(0, n_rows, batch_size):
        stop = start + batch_size
        batch_x = torch.tensor(x_ts[start:stop], dtype=torch.float32).to(device)
        batch_s = torch.tensor(x_static[start:stop], dtype=torch.float32).to(device)
        batch_m = torch.tensor(mask[start:stop], dtype=torch.float32).to(device)
        batch_d = torch.tensor(delta[start:stop], dtype=torch.float32).to(device)
        logits = model(batch_x, batch_m, batch_d, batch_s)
        probs.append(logits.cpu().numpy())
    return np.concatenate(probs)


def fit_lstm_once(
    data: dict,
    cfg: dict,
    seed: int,
    idx_bal,
    keras_verbose: int = 0,
) -> dict:
    set_all_seeds(seed)
    tf.keras.backend.clear_session()
    model = build_lstm_model(
        ts_shape=(data["X_train"].shape[1], data["X_train"].shape[2]),
        static_dim=data["S_train"].shape[1],
        layer_units=cfg["units"],
        dropout_rate=cfg["dropout"],
        learning_rate=cfg["lr"],
    )
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=cfg["patience"],
            restore_best_weights=True,
        )
    ]
    start = time.perf_counter()
    history = model.fit(
        x=[data["X_train"][idx_bal], data["S_train"][idx_bal]],
        y=data["y_train"][idx_bal],
        validation_data=([data["X_val"], data["S_val"]], data["y_val"]),
        epochs=cfg["max_epochs"],
        batch_size=cfg["batch_size"],
        callbacks=callbacks,
        verbose=keras_verbose,
    )
    elapsed = time.perf_counter() - start
    val_score = model.predict([data["X_val"], data["S_val"]], verbose=0).ravel()
    test_score = model.predict([data["X_test"], data["S_test"]], verbose=0).ravel()
    epochs_ran = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    tf.keras.backend.clear_session()
    return {
        "val_score": val_score,
        "test_score": test_score,
        "elapsed_sec": elapsed,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
    }


def fit_svm_once(data: dict, cfg: dict, seed: int, idx_bal) -> dict:
    set_all_seeds(seed)
    model = SVC(
        kernel=cfg["kernel"],
        C=cfg["C"],
        probability=False,
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(data["flat_scaled_train"][idx_bal], data["y_train"][idx_bal])
    elapsed = time.perf_counter() - start
    return {
        "val_score": model.decision_function(data["flat_scaled_val"]),
        "test_score": model.decision_function(data["flat_scaled_test"]),
        "elapsed_sec": elapsed,
    }


def fit_lr_once(data: dict, cfg: dict, seed: int, idx_bal) -> dict:
    set_all_seeds(seed)
    model = LogisticRegression(
        C=cfg["C"],
        penalty=cfg["penalty"],
        solver=cfg["solver"],
        max_iter=cfg["max_iter"],
        random_state=seed,
    )
    start = time.perf_counter()
    model.fit(data["flat_scaled_train"][idx_bal], data["y_train"][idx_bal])
    elapsed = time.perf_counter() - start
    return {
        "val_score": model.decision_function(data["flat_scaled_val"]),
        "test_score": model.decision_function(data["flat_scaled_test"]),
        "elapsed_sec": elapsed,
    }


def fit_grud_once(
    data: dict,
    cfg: dict,
    seed: int,
    idx_bal,
    device: torch.device,
    ts_size: int,
    static_size: int,
    log_every: int = 5,
) -> dict:
    set_all_seeds(seed)
    x_mean = torch.tensor(
        data["X_train"].mean(axis=(0, 1)),
        dtype=torch.float32,
    ).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(data["X_train"][idx_bal], dtype=torch.float32),
            torch.tensor(data["S_train"][idx_bal], dtype=torch.float32),
            torch.tensor(data["mask_train"][idx_bal], dtype=torch.float32),
            torch.tensor(data["delta_train"][idx_bal], dtype=torch.float32),
            torch.tensor(data["y_train"][idx_bal], dtype=torch.float32),
        ),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    model = GRUD(
        ts_size=ts_size,
        hidden_size=cfg["hidden_size"],
        static_size=static_size,
        x_mean=x_mean,
        dropout=cfg["dropout"],
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        patience=5,
        factor=0.5,
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auprc = -1.0
    best_epoch = 0
    patience_count = 0
    best_state = None
    epochs_ran = 0
    start = time.perf_counter()

    for epoch in range(cfg["max_epochs"]):
        epochs_ran = epoch + 1
        model.train()
        for batch_x, batch_s, batch_m, batch_d, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_s = batch_s.to(device)
            batch_m = batch_m.to(device)
            batch_d = batch_d.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x, batch_m, batch_d, batch_s), batch_y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        val_score = grud_predict(
            model,
            data["X_val"],
            data["S_val"],
            data["mask_val"],
            data["delta_val"],
            device,
        )
        val_auprc = average_precision_score(data["y_val"], val_score)
        scheduler.step(val_auprc)

        if log_every and (((epoch + 1) == 1) or ((epoch + 1) % log_every == 0)):
            print(
                f"    epoch {epoch + 1:03d}/{cfg['max_epochs']} | "
                f"val AUPRC={val_auprc:.4f} | best={best_auprc:.4f}"
            )

        if val_auprc > best_auprc:
            best_auprc = val_auprc
            best_epoch = epoch + 1
            patience_count = 0
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= cfg["patience"]:
                print(f"    early stop at epoch {epoch + 1} (best epoch {best_epoch})")
                break

    if best_state is None:
        raise RuntimeError("GRU-D training finished without a saved best state.")

    model.load_state_dict(best_state)
    model.to(device)
    elapsed = time.perf_counter() - start
    return {
        "val_score": grud_predict(
            model,
            data["X_val"],
            data["S_val"],
            data["mask_val"],
            data["delta_val"],
            device,
        ),
        "test_score": grud_predict(
            model,
            data["X_test"],
            data["S_test"],
            data["mask_test"],
            data["delta_test"],
            device,
        ),
        "elapsed_sec": elapsed,
        "epochs_ran": epochs_ran,
        "best_epoch": best_epoch,
    }


def calibrate_fold_scores(y_val, val_score, test_score):
    val_prob_cal = calibrate_platt(y_val, val_score, val_score)
    test_prob_cal = calibrate_platt(y_val, val_score, test_score)
    return val_prob_cal, test_prob_cal


def _run_random_search(
    model_name: str,
    search_space: dict,
    fit_once: Callable,
    data: dict,
    idx_bal,
    n_trials: int,
    seed: int,
    fit_kwargs: dict | None = None,
) -> tuple[dict, pd.DataFrame]:
    print_section(f"{model_name} Random Search")
    rng = random.Random(seed)
    trials = []
    best_cfg = None
    best_score = -1.0
    fit_kwargs = fit_kwargs or {}

    for trial_idx in range(n_trials):
        cfg = sample_cfg(search_space, rng)
        preds = fit_once(data=data, cfg=cfg, seed=seed + trial_idx, idx_bal=idx_bal, **fit_kwargs)
        val_auprc = average_precision_score(data["y_val"], preds["val_score"])
        trial = {"trial": trial_idx + 1, **to_python(cfg), "val_auprc": float(val_auprc)}
        if "elapsed_sec" in preds:
            trial["elapsed_sec"] = float(preds["elapsed_sec"])
        if "epochs_ran" in preds:
            trial["epochs_ran"] = int(preds["epochs_ran"])
        if "best_epoch" in preds:
            trial["best_epoch"] = int(preds["best_epoch"])
        trials.append(trial)

        extra = ""
        if "epochs_ran" in preds:
            extra = (
                f" | epochs={preds['epochs_ran']}"
                f" | best_epoch={preds['best_epoch']}"
            )
        print(
            f"Trial {trial_idx + 1:02d} | cfg={to_python(cfg)}"
            f"{extra} | time={format_elapsed(preds['elapsed_sec'])}"
            f" | AUPRC={val_auprc:.4f}"
        )
        if val_auprc > best_score:
            best_score = val_auprc
            best_cfg = deepcopy(cfg)

    print(f"Best {model_name} config: {to_python(best_cfg)} | val AUPRC={best_score:.4f}")
    return to_python(best_cfg), pd.DataFrame(trials)


def run_lstm_random_search(
    data: dict,
    idx_bal,
    search_space: dict | None = None,
    n_trials: int = 15,
    seed: int = SEARCH_SEED,
    keras_verbose: int = 0,
):
    return _run_random_search(
        model_name="LSTM",
        search_space=search_space or DEFAULT_LSTM_SEARCH_SPACE,
        fit_once=fit_lstm_once,
        data=data,
        idx_bal=idx_bal,
        n_trials=n_trials,
        seed=seed,
        fit_kwargs={"keras_verbose": keras_verbose},
    )


def run_svm_random_search(
    data: dict,
    idx_bal,
    search_space: dict | None = None,
    n_trials: int = 10,
    seed: int = SEARCH_SEED,
):
    return _run_random_search(
        model_name="SVM",
        search_space=search_space or DEFAULT_SVM_SEARCH_SPACE,
        fit_once=fit_svm_once,
        data=data,
        idx_bal=idx_bal,
        n_trials=n_trials,
        seed=seed,
    )


def run_lr_random_search(
    data: dict,
    idx_bal,
    search_space: dict | None = None,
    n_trials: int = 10,
    seed: int = SEARCH_SEED,
):
    return _run_random_search(
        model_name="LR",
        search_space=search_space or DEFAULT_LR_SEARCH_SPACE,
        fit_once=fit_lr_once,
        data=data,
        idx_bal=idx_bal,
        n_trials=n_trials,
        seed=seed,
    )


def run_grud_random_search(
    data: dict,
    idx_bal,
    context: dict,
    search_space: dict | None = None,
    n_trials: int = 15,
    seed: int = SEARCH_SEED,
    log_every: int = 5,
):
    return _run_random_search(
        model_name="GRU-D",
        search_space=search_space or DEFAULT_GRUD_SEARCH_SPACE,
        fit_once=fit_grud_once,
        data=data,
        idx_bal=idx_bal,
        n_trials=n_trials,
        seed=seed,
        fit_kwargs={
            "device": context["device"],
            "ts_size": len(context["ts_cols"]),
            "static_size": 10,
            "log_every": log_every,
        },
    )


def _run_model_cv(
    model_name: str,
    best_cfg: dict,
    context: dict,
    fit_once: Callable,
    fit_kwargs: dict | None = None,
) -> dict:
    print_section(f"{model_name} 5-Fold CV")
    fold_metrics = []
    fold_outputs = []
    fit_kwargs = fit_kwargs or {}

    for fold_idx, train_ids, val_ids, test_ids in context["folds"]:
        print(f"Fold {fold_idx + 1}/5")
        data = prepare_data_cv(
            train_ids,
            val_ids,
            test_ids,
            context["df_ts"],
            context["df_static"],
            context["y_all"],
            context["all_record_ids"],
            context["ts_cols"],
        )
        idx_bal = random_oversample_indices(data["y_train"], seed=fold_idx)
        preds = fit_once(data=data, cfg=best_cfg, seed=fold_idx, idx_bal=idx_bal, **fit_kwargs)
        val_prob_cal, test_prob_cal = calibrate_fold_scores(
            data["y_val"], preds["val_score"], preds["test_score"]
        )
        threshold = youden_threshold(data["y_val"], val_prob_cal)
        metrics = compute_metrics(data["y_test"], test_prob_cal, threshold)
        fold_metrics.append(metrics)
        fold_outputs.append(
            {
                "fold_idx": fold_idx,
                "y_val": data["y_val"].copy(),
                "y_test": data["y_test"].copy(),
                "val_prob_cal": val_prob_cal.copy(),
                "test_prob_cal": test_prob_cal.copy(),
                "threshold": float(threshold),
            }
        )
        print(
            f"  AUROC={metrics['AUROC']:.4f} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"F1={metrics['F1']:.4f}"
        )

    summary = print_cv_summary(fold_metrics)
    return {
        "config": to_python(best_cfg),
        "summary": summary,
        "fold_metrics": fold_metrics,
        "fold_outputs": fold_outputs,
    }


def run_lstm_cv(best_cfg: dict, context: dict, keras_verbose: int = 0) -> dict:
    return _run_model_cv(
        model_name="LSTM",
        best_cfg=best_cfg,
        context=context,
        fit_once=fit_lstm_once,
        fit_kwargs={"keras_verbose": keras_verbose},
    )


def run_svm_cv(best_cfg: dict, context: dict) -> dict:
    return _run_model_cv(
        model_name="SVM",
        best_cfg=best_cfg,
        context=context,
        fit_once=fit_svm_once,
    )


def run_lr_cv(best_cfg: dict, context: dict) -> dict:
    return _run_model_cv(
        model_name="LR",
        best_cfg=best_cfg,
        context=context,
        fit_once=fit_lr_once,
    )


def run_grud_cv(
    best_cfg: dict,
    context: dict,
    log_every: int = 5,
) -> dict:
    return _run_model_cv(
        model_name="GRU-D",
        best_cfg=best_cfg,
        context=context,
        fit_once=fit_grud_once,
        fit_kwargs={
            "device": context["device"],
            "ts_size": len(context["ts_cols"]),
            "static_size": 10,
            "log_every": log_every,
        },
    )


def build_weight_grid(step: float = DEFAULT_WEIGHT_STEP) -> list[tuple[float, float, float]]:
    scaled_total = int(round(1.0 / step))
    weights = []
    for first in range(scaled_total + 1):
        for second in range(scaled_total - first + 1):
            third = scaled_total - first - second
            weights.append((first * step, second * step, third * step))
    return weights


def build_local_weight_grid(
    center_weights: tuple[float, float, float],
    step: float = 0.01,
    radius: float = 0.05,
) -> list[tuple[float, float, float]]:
    """Build a local simplex grid around one coarse optimum."""
    scale = int(round(1.0 / step))
    radius_units = int(round(radius / step))
    center_units = [int(round(weight / step)) for weight in center_weights]

    first_min = max(0, center_units[0] - radius_units)
    first_max = min(scale, center_units[0] + radius_units)
    second_min = max(0, center_units[1] - radius_units)
    second_max = min(scale, center_units[1] + radius_units)
    third_min = max(0, center_units[2] - radius_units)
    third_max = min(scale, center_units[2] + radius_units)

    candidates = []
    for first in range(first_min, first_max + 1):
        for second in range(second_min, second_max + 1):
            third = scale - first - second
            if third < third_min or third > third_max:
                continue
            if third < 0 or third > scale:
                continue
            candidates.append((first * step, second * step, third * step))

    # Preserve deterministic ordering and remove duplicates if rounding collides.
    return sorted(set(candidates))


def _combine_probs(fold_outputs: dict, model_order: tuple[str, str, str], weights: tuple[float, float, float], split: str):
    total = None
    for model_name, weight in zip(model_order, weights):
        probs = fold_outputs[model_name][f"{split}_prob_cal"]
        total = weight * probs if total is None else total + weight * probs
    return total


def _score_weight_candidates(
    component_results: dict,
    model_order: tuple[str, str, str],
    candidate_weights: list[tuple[float, float, float]],
    min_specificity: float = DEFAULT_MIN_SPECIFICITY,
) -> tuple[dict, pd.DataFrame]:
    fold_count = len(component_results[model_order[0]]["fold_outputs"])
    rows = []
    for weights in candidate_weights:
        val_metrics = []
        for fold_idx in range(fold_count):
            fold_outputs = {
                model_name: component_results[model_name]["fold_outputs"][fold_idx]
                for model_name in model_order
            }
            val_prob = _combine_probs(fold_outputs, model_order, weights, split="val")
            threshold = youden_threshold(fold_outputs[model_order[0]]["y_val"], val_prob)
            metrics = compute_metrics(fold_outputs[model_order[0]]["y_val"], val_prob, threshold)
            val_metrics.append(metrics)

        row = {
            "w_1": weights[0],
            "w_2": weights[1],
            "w_3": weights[2],
            "mean_sensitivity": float(np.mean([m["Sensitivity"] for m in val_metrics])),
            "mean_specificity": float(np.mean([m["Specificity"] for m in val_metrics])),
            "mean_auprc": float(np.mean([m["AUPRC"] for m in val_metrics])),
            "mean_auroc": float(np.mean([m["AUROC"] for m in val_metrics])),
            "mean_f1": float(np.mean([m["F1"] for m in val_metrics])),
        }
        rows.append(row)

    grid_df = pd.DataFrame(rows)
    eligible_df = grid_df[grid_df["mean_specificity"] >= min_specificity].copy()
    if eligible_df.empty:
        raise ValueError(
            "No ensemble weight combination satisfied the specificity floor "
            f"of {min_specificity:.2f}."
        )

    eligible_df = eligible_df.sort_values(
        by=["mean_sensitivity", "mean_auprc", "mean_specificity"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    best_row = eligible_df.iloc[0]
    best_weights = {
        model_order[0]: float(best_row["w_1"]),
        model_order[1]: float(best_row["w_2"]),
        model_order[2]: float(best_row["w_3"]),
    }
    print(
        f"Best weights: {best_weights} | "
        f"val Sensitivity={best_row['mean_sensitivity']:.4f} | "
        f"val Specificity={best_row['mean_specificity']:.4f} | "
        f"val AUPRC={best_row['mean_auprc']:.4f}"
    )
    return (
        {
            "weights": best_weights,
            "validation_summary": {
                "Sensitivity": float(best_row["mean_sensitivity"]),
                "Specificity": float(best_row["mean_specificity"]),
                "AUPRC": float(best_row["mean_auprc"]),
                "AUROC": float(best_row["mean_auroc"]),
                "F1": float(best_row["mean_f1"]),
            },
            "min_specificity": float(min_specificity),
        },
        grid_df.sort_values(
            by=["mean_sensitivity", "mean_auprc", "mean_specificity"],
            ascending=[False, False, False],
        ).reset_index(drop=True),
    )


def search_ensemble_weights(
    component_results: dict,
    model_order: tuple[str, str, str],
    step: float = DEFAULT_WEIGHT_STEP,
    min_specificity: float = DEFAULT_MIN_SPECIFICITY,
) -> tuple[dict, pd.DataFrame]:
    print_section(f"Weight Search: {' + '.join(model_order)}")
    summary, ranked_df = _score_weight_candidates(
        component_results=component_results,
        model_order=model_order,
        candidate_weights=build_weight_grid(step=step),
        min_specificity=min_specificity,
    )
    summary["step"] = float(step)
    print(
        f"Best weights: {summary['weights']} | "
        f"val Sensitivity={summary['validation_summary']['Sensitivity']:.4f} | "
        f"val Specificity={summary['validation_summary']['Specificity']:.4f} | "
        f"val AUPRC={summary['validation_summary']['AUPRC']:.4f}"
    )
    return summary, ranked_df


def search_ensemble_weights_with_local_refinement(
    component_results: dict,
    model_order: tuple[str, str, str],
    coarse_step: float = DEFAULT_WEIGHT_STEP,
    refine_step: float = 0.01,
    refine_radius: float = 0.05,
    min_specificity: float = DEFAULT_MIN_SPECIFICITY,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    print_section(f"Weight Search: {' + '.join(model_order)}")
    coarse_summary, coarse_ranked_df = _score_weight_candidates(
        component_results=component_results,
        model_order=model_order,
        candidate_weights=build_weight_grid(step=coarse_step),
        min_specificity=min_specificity,
    )
    coarse_summary["step"] = float(coarse_step)
    coarse_center = tuple(coarse_summary["weights"][name] for name in model_order)

    fine_summary, fine_ranked_df = _score_weight_candidates(
        component_results=component_results,
        model_order=model_order,
        candidate_weights=build_local_weight_grid(
            center_weights=coarse_center,
            step=refine_step,
            radius=refine_radius,
        ),
        min_specificity=min_specificity,
    )
    fine_summary.update(
        {
            "coarse_weights": coarse_summary["weights"],
            "coarse_step": float(coarse_step),
            "refine_step": float(refine_step),
            "refine_radius": float(refine_radius),
        }
    )
    print(
        f"Coarse best: {coarse_summary['weights']} | "
        f"val Sensitivity={coarse_summary['validation_summary']['Sensitivity']:.4f} | "
        f"val Specificity={coarse_summary['validation_summary']['Specificity']:.4f}"
    )
    print(
        f"Refined best: {fine_summary['weights']} | "
        f"val Sensitivity={fine_summary['validation_summary']['Sensitivity']:.4f} | "
        f"val Specificity={fine_summary['validation_summary']['Specificity']:.4f} | "
        f"val AUPRC={fine_summary['validation_summary']['AUPRC']:.4f}"
    )
    return fine_summary, coarse_ranked_df, fine_ranked_df


def evaluate_ensemble(
    component_results: dict,
    model_order: tuple[str, str, str],
    weights: dict,
    label: str,
) -> dict:
    print_section(label)
    fold_metrics = []
    fold_outputs = []

    for fold_idx in range(len(component_results[model_order[0]]["fold_outputs"])):
        fold_outputs_map = {
            model_name: component_results[model_name]["fold_outputs"][fold_idx]
            for model_name in model_order
        }
        ordered_weights = tuple(weights[model_name] for model_name in model_order)
        val_prob = _combine_probs(fold_outputs_map, model_order, ordered_weights, split="val")
        test_prob = _combine_probs(fold_outputs_map, model_order, ordered_weights, split="test")
        threshold = youden_threshold(fold_outputs_map[model_order[0]]["y_val"], val_prob)
        metrics = compute_metrics(fold_outputs_map[model_order[0]]["y_test"], test_prob, threshold)
        fold_metrics.append(metrics)
        fold_outputs.append(
            {
                "fold_idx": fold_idx,
                "y_val": fold_outputs_map[model_order[0]]["y_val"].copy(),
                "y_test": fold_outputs_map[model_order[0]]["y_test"].copy(),
                "val_prob": val_prob.copy(),
                "test_prob": test_prob.copy(),
                "threshold": float(threshold),
            }
        )
        print(
            f"Fold {fold_idx + 1}/5 | "
            f"AUROC={metrics['AUROC']:.4f} | "
            f"AUPRC={metrics['AUPRC']:.4f} | "
            f"F1={metrics['F1']:.4f}"
        )

    summary = print_cv_summary(fold_metrics)
    return {
        "label": label,
        "models": list(model_order),
        "weights": to_python(weights),
        "summary": summary,
        "fold_metrics": fold_metrics,
        "fold_outputs": fold_outputs,
    }


def comparison_table(single_model_results: dict, ensemble_results: dict) -> pd.DataFrame:
    frames = []
    for name, result in single_model_results.items():
        frames.append(summary_to_frame(name, result["summary"]))
    for name, result in ensemble_results.items():
        frames.append(summary_to_frame(name, result["summary"]))
    return pd.concat(frames, ignore_index=True)


def search_space_table(search_space: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [{"parameter": key, "values": json.dumps(to_python(values))} for key, values in search_space.items()]
    )
