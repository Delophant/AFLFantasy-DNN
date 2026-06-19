r"""Backtest GBM predictor vs roll5 baseline on held-out rounds.

Run:
    .\.venv\Scripts\python.exe src\evaluate_predictor.py
    .\.venv\Scripts\python.exe src\evaluate_predictor.py --season 2026 --round 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import TARGET
from predictor_model import (
    TEST_SEASON,
    load_artifact,
    metrics,
    predict_from_rows,
    top_k_recall_per_match,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "prediction_dataset.csv"


def evaluate_split(df: pd.DataFrame, label: str) -> None:
    booster, meta = load_artifact()
    rows = df.to_dict("records")
    gbm_pred = predict_from_rows(rows, booster, meta)
    roll5 = df["roll5_fantasy"].to_numpy()
    actual = df[TARGET].to_numpy()

    eval_df = df.copy()
    eval_df["gbm_pred"] = gbm_pred
    eval_df["roll5_pred"] = roll5

    print(f"\n=== {label} ({len(df)} players) ===")
    print(f"{'model':<12}{'MAE':>8}{'RMSE':>8}{'corr':>8}{'max':>8}{'std':>8}{'>100':>6}")
    for name, pred in [("roll5", roll5), ("gbm", gbm_pred)]:
        m = metrics(actual, pred)
        print(
            f"{name:<12}{m['mae']:>8.2f}{m['rmse']:>8.2f}{m['corr']:>8.3f}"
            f"{pred.max():>8.1f}{pred.std():>8.1f}{(pred > 100).sum():>6}"
        )

    print(f"top-20 recall roll5: {top_k_recall_per_match(eval_df, 'roll5_pred'):.3f}")
    print(f"top-20 recall gbm:   {top_k_recall_per_match(eval_df, 'gbm_pred'):.3f}")

    print("\nTop 10 actual vs predictions:")
    show = eval_df.nlargest(10, TARGET)[
        ["player", "team", "opponent", TARGET, "roll5_fantasy", "gbm_pred"]
    ].rename(columns={TARGET: "actual", "roll5_fantasy": "roll5", "gbm_pred": "gbm"})
    print(show.round(1).to_string(index=False))


def filter_round(df: pd.DataFrame, season: int, round_num: int) -> pd.DataFrame:
    stats = pd.read_csv(
        ROOT / "data" / "processed" / "player_match_stats.csv",
        parse_dates=["date"],
    )
    round_dates = stats[
        (stats["season"] == season)
        & (stats["round"].astype(str).str.fullmatch(rf"Round\s+{round_num}", case=False, na=False))
    ]["date"].drop_duplicates()
    return df[df["date"].isin(round_dates)]


def main(season: int | None, round_num: int | None) -> None:
    df = pd.read_csv(DATA, parse_dates=["date"])
    booster, meta = load_artifact()
    print(f"loaded model (blend roll5 weight={meta['blend_weight_roll5']:.2f})")

    test_df = df[df["season"] == TEST_SEASON]
    evaluate_split(test_df, f"TEST SEASON {TEST_SEASON}")

    if season is not None:
        sdf = df[df["season"] == season]
        if round_num is not None:
            sdf = filter_round(sdf, season, round_num)
            label = f"SEASON {season} ROUND {round_num}"
        else:
            label = f"SEASON {season}"
        if not sdf.empty:
            evaluate_split(sdf, label)
        else:
            print(f"\n(no rows for {label})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GBM predictor vs baselines")
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--round", type=int, default=14, help="Played round to evaluate (default: 14)")
    args = parser.parse_args()
    main(args.season, args.round)
