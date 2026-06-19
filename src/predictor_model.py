r"""LightGBM fantasy score predictor — train, save, load, infer.

Residual target: predict (fantasy - roll5_fantasy), add roll5 back at inference.

Artifacts in models/:
  predictor_lgbm.txt       — LightGBM booster
  predictor_meta.json      — feature lists, blend weight, metrics

Run training:
    .\.venv\Scripts\python.exe src\train_predictor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (
    MIN_PRIOR_GAMES,
    PREDICT_CATEGORICAL,
    PREDICT_NUMERIC,
    RESIDUAL_TARGET,
    TARGET,
    build_player_row,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "prediction_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "predictor_lgbm.txt"
META_PATH = MODEL_DIR / "predictor_meta.json"

TRAIN_SEASONS = [2021, 2022, 2023]
VAL_SEASON = 2024
TEST_SEASON = 2025

BLEND_GRID = np.arange(0.0, 1.01, 0.05)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]) if len(y_true) > 1 else 0.0,
    }


def top_k_recall_per_match(df: pd.DataFrame, pred_col: str, k: int = 20) -> float:
    """Fraction of actual top-k scorers captured in predicted top-k, averaged per match."""
    if "date" not in df.columns:
        return 0.0
    hits, total = 0, 0
    for _, g in df.groupby("date"):
        if len(g) < k:
            continue
        actual_top = set(g.nlargest(k, TARGET)["player"])
        pred_top = set(g.nlargest(k, pred_col)["player"])
        hits += len(actual_top & pred_top)
        total += k
    return hits / total if total else 0.0


def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in PREDICT_CATEGORICAL:
        out[col] = out[col].astype("category")
    for col in PREDICT_NUMERIC:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    cols = PREDICT_NUMERIC + PREDICT_CATEGORICAL
    return _prepare_frame(df)[cols]


def tune_blend(y_true: np.ndarray, roll5: np.ndarray, gbm_pred: np.ndarray) -> float:
    best_w, best_mae = 0.0, float("inf")
    for w in BLEND_GRID:
        blended = w * roll5 + (1 - w) * gbm_pred
        mae = mean_absolute_error(y_true, blended)
        if mae < best_mae:
            best_mae = mae
            best_w = float(w)
    return best_w


def train(
    df: pd.DataFrame | None = None,
    save: bool = True,
) -> tuple[lgb.Booster, dict]:
    if df is None:
        df = pd.read_csv(DATA, parse_dates=["date"])

    train_df = df[df["season"].isin(TRAIN_SEASONS)]
    val_df = df[df["season"] == VAL_SEASON]
    test_df = df[df["season"] == TEST_SEASON]

    X_tr = feature_matrix(train_df)
    y_tr = train_df[RESIDUAL_TARGET].to_numpy()
    X_va = feature_matrix(val_df)
    y_va = val_df[RESIDUAL_TARGET].to_numpy()

    train_set = lgb.Dataset(X_tr, label=y_tr, categorical_feature=PREDICT_CATEGORICAL)
    val_set = lgb.Dataset(X_va, label=y_va, categorical_feature=PREDICT_CATEGORICAL, reference=train_set)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42,
    }

    booster = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(30, verbose=False)],
    )

    def predict_full(split_df: pd.DataFrame) -> np.ndarray:
        resid = booster.predict(feature_matrix(split_df))
        return split_df["roll5_fantasy"].to_numpy() + resid

    val_roll5 = val_df["roll5_fantasy"].to_numpy()
    val_gbm = predict_full(val_df)
    val_actual = val_df[TARGET].to_numpy()
    blend_w = tune_blend(val_actual, val_roll5, val_gbm)

    def blended(split_df: pd.DataFrame, gbm: np.ndarray) -> np.ndarray:
        r5 = split_df["roll5_fantasy"].to_numpy()
        return blend_w * r5 + (1 - blend_w) * gbm

    benchmark: dict[str, dict] = {}

    for name, split_df in [
        ("val", val_df),
        ("test", test_df),
    ]:
        actual = split_df[TARGET].to_numpy()
        roll5 = split_df["roll5_fantasy"].to_numpy()
        season_avg = split_df["season_avg_fantasy"].to_numpy()
        blend_base = 0.6 * roll5 + 0.4 * season_avg
        gbm_full = predict_full(split_df)
        gbm_blend = blended(split_df, gbm_full)

        benchmark[name] = {
            "roll5": metrics(actual, roll5),
            "season_avg": metrics(actual, season_avg),
            "blend_60_40": metrics(actual, blend_base),
            "gbm_residual": metrics(actual, gbm_full),
            "gbm_blend": metrics(actual, gbm_blend),
            "top20_recall_roll5": top_k_recall_per_match(split_df.assign(_p=roll5), "_p"),
            "top20_recall_gbm": top_k_recall_per_match(split_df.assign(_p=gbm_blend), "_p"),
        }

    meta = {
        "train_seasons": TRAIN_SEASONS,
        "val_season": VAL_SEASON,
        "test_season": TEST_SEASON,
        "blend_weight_roll5": blend_w,
        "numeric_features": PREDICT_NUMERIC,
        "categorical_features": PREDICT_CATEGORICAL,
        "min_prior_games": MIN_PRIOR_GAMES,
        "benchmark": benchmark,
    }

    if save:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        booster.save_model(str(MODEL_PATH))
        META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return booster, meta


def load_artifact() -> tuple[lgb.Booster, dict]:
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(
            f"Missing model artifacts. Run: .\\.venv\\Scripts\\python.exe src\\train_predictor.py"
        )
    booster = lgb.Booster(model_file=str(MODEL_PATH))
    meta = json.loads(META_PATH.read_text(encoding="utf-8"))
    return booster, meta


def predict_from_rows(rows: list[dict], booster: lgb.Booster | None = None, meta: dict | None = None) -> np.ndarray:
    if booster is None or meta is None:
        booster, meta = load_artifact()

    if not rows:
        return np.array([])

    df = pd.DataFrame(rows)
    resid = booster.predict(feature_matrix(df))
    roll5 = df["roll5_fantasy"].to_numpy()
    gbm_full = roll5 + resid
    w = meta.get("blend_weight_roll5", 0.0)
    return w * roll5 + (1 - w) * gbm_full


def predict_player_scores(
    history: pd.DataFrame,
    squad: list[tuple[str, str, str, str]],
    season: int,
    venue: str | None,
    match_date: pd.Timestamp | None = None,
    booster: lgb.Booster | None = None,
    meta: dict | None = None,
) -> list[dict]:
    """Predict scores for squad members. Each tuple: (player, team, opponent, home_away)."""
    rows = []
    for player, team, opponent, home_away in squad:
        row = build_player_row(
            history, player, team, opponent, home_away, venue, season, match_date
        )
        if row:
            rows.append(row)

    if not rows:
        return []

    preds = predict_from_rows(rows, booster, meta)
    out = []
    for row, pred in zip(rows, preds):
        recent = row["roll5_fantasy"]
        out.append({
            "player": row["player"],
            "team": row["team"],
            "opponent": row["opponent"],
            "home_away": row["home_away"],
            "games_2026": row["games_played"],
            "recent5_avg": round(recent, 1),
            "predicted_fantasy": round(float(pred), 1),
            "delta_vs_recent": round(float(pred) - recent, 1),
        })
    return sorted(out, key=lambda r: r["predicted_fantasy"], reverse=True)
