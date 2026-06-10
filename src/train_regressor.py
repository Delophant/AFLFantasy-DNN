r"""Predict a player's next-match AFL Fantasy score, and benchmark honestly.

We climb a ladder of models, each compared on the SAME held-out test season:

  0. Naive baselines (no learning): global mean, last score, running averages.
  1. Linear regression (scikit-learn).
  2. Linear model in Keras  (Dense(1), no hidden layer)  <- same as #1, by design.
  3. Neural network in Keras (hidden layers + ReLU)       <- the "deep" model.

The question the table answers: does the neural net actually beat "just predict
their season average"? On tabular form data, that bar is surprisingly high.

Split is TEMPORAL (never random): train on past seasons, test on a future one.

Run:
    .\.venv\Scripts\python.exe src\train_regressor.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

from features import FEATURES, TARGET  # reuse the agreed feature list

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "regression_dataset.csv"
FIG_DIR = ROOT / "reports" / "figures"

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
TEST_SEASON = 2025


def metrics(y_true, y_pred) -> tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return mae, rmse


def build_keras_linear(n_features: int) -> keras.Model:
    # No hidden layer, no activation == linear regression.
    # Larger learning rate so the output bias can travel from ~0 to the target
    # mean (~67) within our epoch budget; otherwise gradient descent stalls short.
    model = keras.Sequential([keras.layers.Input((n_features,)), keras.layers.Dense(1)])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.05), loss="mse", metrics=["mae"])
    return model


def build_keras_mlp(n_features: int) -> keras.Model:
    # Hidden layers + ReLU == a (small) deep neural network.
    model = keras.Sequential(
        [
            keras.layers.Input((n_features,)),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss="mse", metrics=["mae"])
    return model


def fit_keras(model: keras.Model, Xtr, ytr, Xval, yval):
    stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=12, restore_best_weights=True
    )
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        epochs=200, batch_size=256, verbose=0, callbacks=[stop],
    )
    return history


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val = df[df["season"] == VAL_SEASON]
    test = df[df["season"] == TEST_SEASON]
    print(f"train rows: {len(train):,} | val rows: {len(val):,} | test rows: {len(test):,}")

    Xtr_raw, ytr = train[FEATURES].to_numpy(), train[TARGET].to_numpy()
    Xval_raw, yval = val[FEATURES].to_numpy(), val[TARGET].to_numpy()
    Xte_raw, yte = test[FEATURES].to_numpy(), test[TARGET].to_numpy()

    # Standardize features (fit on TRAIN only). Vital for the neural net; harmless
    # for linear models. Fitting on train only avoids leaking test statistics.
    scaler = StandardScaler().fit(Xtr_raw)
    Xtr, Xval, Xte = scaler.transform(Xtr_raw), scaler.transform(Xval_raw), scaler.transform(Xte_raw)

    results: dict[str, tuple[float, float]] = {}

    # ---- 0. Naive baselines (evaluated directly on the test season) ----
    results["Naive: global mean"] = metrics(yte, np.full_like(yte, ytr.mean(), dtype=float))
    results["Naive: last score"] = metrics(yte, test["prev_score"].to_numpy())
    results["Naive: season avg"] = metrics(yte, test["season_avg_fantasy"].to_numpy())
    results["Naive: career avg"] = metrics(yte, test["career_avg_fantasy"].to_numpy())

    # ---- 1. Linear regression (scikit-learn) ----
    lin = LinearRegression().fit(Xtr, ytr)
    results["Linear regression (sklearn)"] = metrics(yte, lin.predict(Xte))

    # ---- 2. Linear model in Keras (sanity: should match sklearn) ----
    klin = build_keras_linear(Xtr.shape[1])
    fit_keras(klin, Xtr, ytr, Xval, yval)
    results["Linear (Keras Dense(1))"] = metrics(yte, klin.predict(Xte, verbose=0).ravel())

    # ---- 3. Neural network (the deep model) ----
    mlp = build_keras_mlp(Xtr.shape[1])
    history = fit_keras(mlp, Xtr, ytr, Xval, yval)
    mlp_pred = mlp.predict(Xte, verbose=0).ravel()
    results["Neural net (32-16, ReLU)"] = metrics(yte, mlp_pred)

    # ---- Report ----
    print("\n" + "=" * 56)
    print(f"TEST SEASON {TEST_SEASON} | MAE = avg error in fantasy points")
    print("=" * 56)
    print(f"{'model':<32}{'MAE':>8}{'RMSE':>9}")
    print("-" * 56)
    for name, (mae, rmse) in sorted(results.items(), key=lambda kv: kv[1][0]):
        print(f"{name:<32}{mae:>8.2f}{rmse:>9.2f}")

    # ---- Figures ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history["mae"], label="train MAE")
    plt.plot(history.history["val_mae"], label="val MAE")
    plt.title("Neural net training curve")
    plt.xlabel("epoch")
    plt.ylabel("MAE (fantasy points)")
    plt.legend()
    plt.tight_layout()
    f1 = FIG_DIR / "regressor_training_curve.png"
    plt.savefig(f1, dpi=110)
    plt.close()

    plt.figure(figsize=(5.5, 5.5))
    plt.scatter(yte, mlp_pred, s=6, alpha=0.25, color="#3b7dd8")
    lim = [yte.min(), yte.max()]
    plt.plot(lim, lim, "r--", linewidth=1)
    plt.title(f"Neural net: predicted vs actual ({TEST_SEASON})")
    plt.xlabel("actual fantasy points")
    plt.ylabel("predicted")
    plt.tight_layout()
    f2 = FIG_DIR / "regressor_pred_vs_actual.png"
    plt.savefig(f2, dpi=110)
    plt.close()

    print(f"\nsaved figures:\n  {f1.relative_to(ROOT)}\n  {f2.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
