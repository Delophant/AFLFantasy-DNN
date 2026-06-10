r"""Form & anomaly detection with an LSTM sequence model (deliverable #2).

Step 1 showed that *aggregate* form features (averages) can't beat "predict their
average", because averaging discards the SHAPE of recent games. Here we feed an
LSTM the player's last N games as an ordered sequence of full stat-lines and ask
it to predict the next game's fantasy score. Two payoffs:

* Forecasting: does the trajectory's shape beat the season-average baseline?
* Form / anomaly signal:
    - predicted - recent_average  -> "about to hit form" (or decline) call
    - actual    - predicted       -> over/under-performance (surprise) anomaly

Split is temporal: train 2021-2023, validate 2024, test 2025. Sequences are built
within a single (player, season) so we never bridge the long off-season gap.

Run:
    .\.venv\Scripts\python.exe src\sequence_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow import keras

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
FIG_DIR = ROOT / "reports" / "figures"

SEQ_FEATURES = [
    "kicks", "handballs", "marks", "goals", "behinds", "tackles", "hitouts",
    "goal_assists", "inside50s", "clearances", "clangers", "rebound50s",
    "frees_for", "frees_against", "fantasy_points",
]
SEQ_LEN = 5                     # use the last 5 games to predict the next
TRAIN_SEASONS = {2021, 2022, 2023}
VAL_SEASON = 2024
TEST_SEASON = 2025


def build_sequences(df: pd.DataFrame):
    """Return arrays X (samples, SEQ_LEN, n_feats), y, plus aligned metadata."""
    X, y = [], []
    season, recent_avg, players, dates = [], [], [], []
    feat = df[SEQ_FEATURES].to_numpy(dtype="float32")

    for (player, _season), idx in df.groupby(["player", "season"], sort=False).indices.items():
        idx = np.sort(idx)            # chronological within the season
        fp = feat[idx][:, -1]         # fantasy_points is the last feature column
        for t in range(SEQ_LEN, len(idx)):
            window = idx[t - SEQ_LEN:t]
            X.append(feat[window])
            y.append(fp[t])
            recent_avg.append(fp[t - SEQ_LEN:t].mean())
            season.append(_season)
            players.append(player)
            dates.append(df["date"].to_numpy()[idx[t]])

    return (
        np.asarray(X, dtype="float32"),
        np.asarray(y, dtype="float32"),
        np.asarray(season),
        np.asarray(recent_avg, dtype="float32"),
        np.asarray(players),
        np.asarray(dates),
    )


def metrics(y_true, y_pred) -> tuple[float, float]:
    return (
        mean_absolute_error(y_true, y_pred),
        float(np.sqrt(mean_squared_error(y_true, y_pred))),
    )


def main() -> None:
    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["player", "season", "date"])
    X, y, season, recent_avg, players, dates = build_sequences(df)
    print(f"sequences: {len(X):,}  (shape {X.shape})")

    tr = np.isin(season, list(TRAIN_SEASONS))
    va = season == VAL_SEASON
    te = season == TEST_SEASON
    print(f"train {tr.sum():,} | val {va.sum():,} | test {te.sum():,}")

    # Standardize each feature using TRAIN statistics only (over all timesteps).
    flat = X[tr].reshape(-1, X.shape[-1])
    mu, sigma = flat.mean(0), flat.std(0) + 1e-8
    Xn = (X - mu) / sigma

    model = keras.Sequential([
        keras.layers.Input((SEQ_LEN, X.shape[-1])),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse", metrics=["mae"])
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True)
    history = model.fit(
        Xn[tr], y[tr], validation_data=(Xn[va], y[va]),
        epochs=120, batch_size=128, verbose=0, callbacks=[stop],
    )

    pred = model.predict(Xn[te], verbose=0).ravel()
    lstm_mae, lstm_rmse = metrics(y[te], pred)
    base_mae, base_rmse = metrics(y[te], recent_avg[te])           # predict last-5 average
    print("\n" + "=" * 52)
    print(f"TEST {TEST_SEASON}: predicting next-game fantasy score")
    print("=" * 52)
    print(f"{'model':<34}{'MAE':>7}{'RMSE':>8}")
    print("-" * 52)
    print(f"{'Naive: last-5 game average':<34}{base_mae:>7.2f}{base_rmse:>8.2f}")
    print(f"{'LSTM sequence model':<34}{lstm_mae:>7.2f}{lstm_rmse:>8.2f}")

    # ---- Form watch: biggest predicted swings vs recent average (test season) ----
    te_idx = np.where(te)[0]
    watch = pd.DataFrame({
        "player": players[te_idx],
        "date": dates[te_idx],
        "recent_avg": recent_avg[te_idx].round(1),
        "predicted": pred.round(1),
        "actual": y[te_idx],
    })
    watch["delta_vs_recent"] = (watch["predicted"] - watch["recent_avg"]).round(1)
    watch["surprise"] = (watch["actual"] - watch["predicted"]).round(1)

    print("\nTop 8 'about to HIT FORM' calls (model predicts a jump vs recent avg):")
    print(watch.sort_values("delta_vs_recent", ascending=False)
          .head(8)[["player", "date", "recent_avg", "predicted", "actual"]].to_string(index=False))
    print("\nTop 8 'about to COOL OFF' calls (model predicts a drop vs recent avg):")
    print(watch.sort_values("delta_vs_recent")
          .head(8)[["player", "date", "recent_avg", "predicted", "actual"]].to_string(index=False))
    print("\nTop 8 biggest ANOMALIES (actual blew past prediction):")
    print(watch.sort_values("surprise", ascending=False)
          .head(8)[["player", "date", "recent_avg", "predicted", "actual"]].to_string(index=False))

    # ---- Figures ----
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(history.history["mae"], label="train MAE")
    plt.plot(history.history["val_mae"], label="val MAE")
    plt.title("LSTM training curve")
    plt.xlabel("epoch"); plt.ylabel("MAE (fantasy points)"); plt.legend()
    plt.tight_layout()
    f1 = FIG_DIR / "sequence_training_curve.png"
    plt.savefig(f1, dpi=110); plt.close()

    # A single player's 2025 trajectory: actual vs LSTM expectation.
    star = watch["player"].value_counts().index[0]
    s = watch[watch["player"] == star].sort_values("date")
    if len(s) >= 4:
        plt.figure(figsize=(9, 4.5))
        plt.plot(s["date"], s["actual"], "o-", label="actual", color="#3b7dd8")
        plt.plot(s["date"], s["predicted"], "s--", label="LSTM predicted", color="#e07b39")
        plt.title(f"{star}: actual vs LSTM-predicted fantasy ({TEST_SEASON})")
        plt.xlabel("date"); plt.ylabel("fantasy points"); plt.legend()
        plt.tight_layout()
        f2 = FIG_DIR / "sequence_player_trajectory.png"
        plt.savefig(f2, dpi=110); plt.close()
        print(f"\nsaved figures:\n  {f1.relative_to(ROOT)}\n  {f2.relative_to(ROOT)}")
    else:
        print(f"\nsaved figure:\n  {f1.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
