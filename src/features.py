r"""Build leakage-free modelling tables for next-round fantasy prediction.

Every feature for a given match must be computable *before* that match is played.
Features use only prior games (shift/rolling on past rows).

Outputs:
  data/processed/regression_dataset.csv   — legacy feature set (train_regressor.py)
  data/processed/prediction_dataset.csv   — extended features (train_predictor.py)

Run:
    .\.venv\Scripts\python.exe src\features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "processed" / "player_match_stats.csv"
REGRESSION_OUT = ROOT / "data" / "processed" / "regression_dataset.csv"
PREDICTION_OUT = ROOT / "data" / "processed" / "prediction_dataset.csv"

# Legacy columns used by train_regressor.py
FEATURES = [
    "prev_score",
    "roll3_fantasy",
    "roll5_fantasy",
    "season_avg_fantasy",
    "career_avg_fantasy",
    "roll3_disposals",
    "roll5_disposals",
    "roll3_tackles",
    "games_played",
    "is_home",
]

# Extended numeric features for GBM predictor
PREDICT_NUMERIC = FEATURES + [
    "roll5_goals",
    "roll5_tackles",
    "opponent_avg_conceded",
    "player_vs_opponent_avg",
    "days_since_last_game",
]

PREDICT_CATEGORICAL = ["team", "opponent", "venue"]

# Archetype-enhanced variant (second GBM)
ARCHETYPE_NUMERIC = PREDICT_NUMERIC + [
    "latent_1",
    "latent_2",
    "form_trend",
    "roll5_fantasy_std",
]
ARCHETYPE_CATEGORICAL = PREDICT_CATEGORICAL + ["archetype_cluster"]

TARGET = "target"
RESIDUAL_TARGET = "target_residual"
MIN_PRIOR_GAMES = 3


def _prior_roll(by_player: pd.core.groupby.DataFrameGroupBy, col: str, window: int) -> pd.Series:
    return by_player[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=window).mean()
    )


def _opponent_conceded_prior(df: pd.DataFrame) -> pd.Series:
    """Expanding mean fantasy scored against each opponent before each match date."""
    rows = []
    for opponent, g in df.groupby("opponent", sort=False):
        g = g.sort_values("date")
        conceded = g.groupby("date")["fantasy_points"].mean().sort_index()
        expanding = conceded.expanding(min_periods=1).mean().shift(1)
        for date, val in expanding.items():
            rows.append({"opponent": opponent, "date": date, "opponent_avg_conceded": val})
    if not rows:
        return pd.Series(np.nan, index=df.index)
    lookup = pd.DataFrame(rows)
    merged = df[["opponent", "date"]].merge(lookup, on=["opponent", "date"], how="left")
    return merged["opponent_avg_conceded"]


def _player_vs_opponent_prior(df: pd.DataFrame) -> pd.Series:
    """Player's prior-game average vs this opponent (NaN if no prior meeting)."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    for (player, opponent), g in df.groupby(["player", "opponent"], sort=False):
        g = g.sort_values("date")
        prior_avg = g["fantasy_points"].shift(1).expanding(min_periods=1).mean()
        out.loc[g.index] = prior_avg.to_numpy()
    return out


def build_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """Attach all leakage-free features to the match-level player table."""
    df = df.sort_values(["player", "date"]).reset_index(drop=True)
    by_player = df.groupby("player", sort=False)

    df["prev_score"] = by_player["fantasy_points"].shift(1)
    df["roll3_fantasy"] = _prior_roll(by_player, "fantasy_points", 3)
    df["roll5_fantasy"] = _prior_roll(by_player, "fantasy_points", 5)
    df["roll3_disposals"] = _prior_roll(by_player, "disposals", 3)
    df["roll5_disposals"] = _prior_roll(by_player, "disposals", 5)
    df["roll3_tackles"] = _prior_roll(by_player, "tackles", 3)
    df["roll5_tackles"] = _prior_roll(by_player, "tackles", 5)
    df["roll5_goals"] = _prior_roll(by_player, "goals", 5)

    df["career_avg_fantasy"] = by_player["fantasy_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )
    df["season_avg_fantasy"] = df.groupby(["player", "season"])["fantasy_points"].transform(
        lambda s: s.shift(1).expanding(min_periods=1).mean()
    )

    df["games_played"] = by_player.cumcount()
    df["is_home"] = (df["home_away"] == "home").astype(int)
    df["venue"] = df["venue"].fillna("Unknown").astype(str)
    df["team"] = df["team"].astype(str)
    df["opponent"] = df["opponent"].astype(str)

    df["days_since_last_game"] = by_player["date"].diff().dt.days

    df = df.sort_values("date").reset_index(drop=True)
    df["opponent_avg_conceded"] = _opponent_conceded_prior(df)
    df["player_vs_opponent_avg"] = _player_vs_opponent_prior(df)

    df["target"] = df["fantasy_points"]
    df["target_residual"] = df["target"] - df["roll5_fantasy"]
    df["form_trend"] = df["roll3_fantasy"] - df["roll5_fantasy"]
    df["roll5_fantasy_std"] = by_player["fantasy_points"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=MIN_PRIOR_GAMES).std()
    )
    return df


def build(min_prior_games: int = MIN_PRIOR_GAMES) -> pd.DataFrame:
    df = pd.read_csv(SRC, parse_dates=["date"])
    df = build_raw_features(df)

    from archetype_features import attach_to_dataframe

    df = attach_to_dataframe(df)

    meta = ["season", "date", "player", "team", "opponent", "venue", "home_away", "match_id"]
    meta = [c for c in meta if c in df.columns]
    predict_cols = meta + ARCHETYPE_NUMERIC + ARCHETYPE_CATEGORICAL + [TARGET, RESIDUAL_TARGET]
    model_df = df[predict_cols].copy()

    required = [
        c
        for c in ARCHETYPE_NUMERIC
        if c not in ("player_vs_opponent_avg", "days_since_last_game", "roll5_fantasy_std")
    ]
    before = len(model_df)
    model_df = model_df.dropna(subset=required).reset_index(drop=True)
    model_df["archetype_cluster"] = model_df["archetype_cluster"].astype(int)
    print(
        f"rows: {before:,} -> {len(model_df):,} after dropping warm-up games "
        f"(need >={min_prior_games} prior games)"
    )
    return model_df


def build_player_row(
    history: pd.DataFrame,
    player: str,
    team: str,
    opponent: str,
    home_away: str,
    venue: str | None,
    season: int,
    match_date: pd.Timestamp | None = None,
) -> dict | None:
    """Build one pre-match feature row for an upcoming fixture (inference)."""
    p = history[(history["player"] == player) & (history["season"] == season)].sort_values("date")
    if len(p) < MIN_PRIOR_GAMES:
        return None

    as_of = match_date if match_date is not None else p["date"].max() + pd.Timedelta(days=1)

    vs_opp = p[(p["opponent"] == opponent) & (p["date"] < as_of)]
    player_vs_opp = float(vs_opp["fantasy_points"].mean()) if len(vs_opp) >= 1 else np.nan

    opp_rows = history[
        (history["opponent"] == opponent)
        & (history["season"] == season)
        & (history["date"] < as_of)
    ]
    if not opp_rows.empty:
        opp_conceded = float(opp_rows.groupby("date")["fantasy_points"].mean().mean())
    else:
        opp_conceded = np.nan

    days_since = np.nan
    if len(p) >= 2:
        days_since = (p["date"].iloc[-1] - p["date"].iloc[-2]).days

    row = {
        "player": player,
        "team": team,
        "opponent": opponent,
        "venue": str(venue or "Unknown"),
        "home_away": home_away,
        "season": season,
        "prev_score": float(p["fantasy_points"].iloc[-1]),
        "roll3_fantasy": float(p["fantasy_points"].tail(3).mean()),
        "roll5_fantasy": float(p["fantasy_points"].tail(5).mean()),
        "season_avg_fantasy": float(p["fantasy_points"].mean()),
        "career_avg_fantasy": float(
            history[(history["player"] == player) & (history["date"] < as_of)]["fantasy_points"].mean()
        ),
        "roll3_disposals": float(p["disposals"].tail(3).mean()),
        "roll5_disposals": float(p["disposals"].tail(5).mean()),
        "roll3_tackles": float(p["tackles"].tail(3).mean()),
        "roll5_tackles": float(p["tackles"].tail(5).mean()),
        "roll5_goals": float(p["goals"].tail(5).mean()),
        "games_played": len(p),
        "is_home": int(home_away == "home"),
        "opponent_avg_conceded": opp_conceded,
        "player_vs_opponent_avg": player_vs_opp,
        "days_since_last_game": days_since,
    }
    from archetype_features import enrich_player_row

    return enrich_player_row(row, history, player, season, match_date)


def main() -> None:
    model_df = build()
    model_df.to_csv(PREDICTION_OUT, index=False)

    legacy_cols = ["season", "date", "player", "team", "opponent"] + FEATURES + [TARGET]
    model_df[legacy_cols].to_csv(REGRESSION_OUT, index=False)

    print(f"\nprediction features ({len(ARCHETYPE_NUMERIC)} numeric + {len(ARCHETYPE_CATEGORICAL)} categorical)")
    print(f"numeric: {ARCHETYPE_NUMERIC}")
    print(f"categorical: {ARCHETYPE_CATEGORICAL}")
    print(f"\nseasons: {sorted(model_df['season'].unique())}")
    print(model_df.groupby("season").size().to_string())
    print(f"\nwrote -> {PREDICTION_OUT.relative_to(ROOT)}")
    print(f"wrote -> {REGRESSION_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
