r"""Leakage-free archetype features from saved autoencoder artifacts.

Rolling style profiles use only prior games (shift + rolling). At inference,
profiles are built from games strictly before match_date.

Run autoencoder.py first to create models/archetype_*.pkl|keras|json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from autoencoder import STYLE_FEATURES

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
SCALER_PATH = MODEL_DIR / "archetype_scaler.pkl"
ENCODER_PATH = MODEL_DIR / "archetype_encoder.keras"
KMEANS_PATH = MODEL_DIR / "archetype_kmeans.pkl"
PRIOR_CLUSTER_PATH = MODEL_DIR / "archetype_prior_clusters.json"

STYLE_WINDOW = 5
STYLE_MIN_PERIODS = 3
MISSING_CLUSTER = -1


def artifacts_ready() -> bool:
    return all(
        p.exists()
        for p in (SCALER_PATH, ENCODER_PATH, KMEANS_PATH, PRIOR_CLUSTER_PATH)
    )


def load_artifacts() -> tuple:
    if not artifacts_ready():
        raise FileNotFoundError(
            "Missing archetype artifacts. Run: .\\.venv\\Scripts\\python.exe src\\autoencoder.py"
        )
    scaler = joblib.load(SCALER_PATH)
    encoder = keras.models.load_model(str(ENCODER_PATH), compile=False)
    kmeans = joblib.load(KMEANS_PATH)
    prior_clusters = json.loads(PRIOR_CLUSTER_PATH.read_text(encoding="utf-8"))
    return scaler, encoder, kmeans, prior_clusters


def _prior_roll(by_player: pd.core.groupby.DataFrameGroupBy, col: str, window: int) -> pd.Series:
    return by_player[col].transform(
        lambda s: s.shift(1).rolling(window, min_periods=STYLE_MIN_PERIODS).mean()
    )


def rolling_style_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Per-row rolling mean of style stats from prior games only."""
    ordered = df.sort_values(["player", "date"]).reset_index(drop=True)
    by_player = ordered.groupby("player", sort=False)
    parts = [_prior_roll(by_player, feat, STYLE_WINDOW) for feat in STYLE_FEATURES]
    profiles = np.column_stack([p.to_numpy(dtype=float) for p in parts])
    valid = np.isfinite(profiles).all(axis=1)
    return ordered, profiles, valid


def encode_profiles(
    profiles: np.ndarray,
    valid_mask: np.ndarray,
    players: pd.Series,
    prior_clusters: dict[str, int],
    scaler,
    encoder,
    kmeans,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(profiles)
    latent_1 = np.zeros(n, dtype=float)
    latent_2 = np.zeros(n, dtype=float)
    clusters = np.full(n, MISSING_CLUSTER, dtype=int)

    if valid_mask.any():
        X = scaler.transform(profiles[valid_mask])
        latent = encoder.predict(X, verbose=0)
        latent_1[valid_mask] = latent[:, 0]
        latent_2[valid_mask] = latent[:, 1]
        clusters[valid_mask] = kmeans.predict(latent)

    for i, player in enumerate(players):
        if valid_mask[i]:
            continue
        prior = prior_clusters.get(player)
        if prior is not None:
            clusters[i] = int(prior)

    return latent_1, latent_2, clusters


def attach_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add latent_1, latent_2, archetype_cluster to a feature dataframe."""
    scaler, encoder, kmeans, prior_clusters = load_artifacts()
    ordered, profiles, valid = rolling_style_matrix(df)
    l1, l2, clusters = encode_profiles(
        profiles, valid, ordered["player"], prior_clusters, scaler, encoder, kmeans
    )
    ordered["latent_1"] = l1
    ordered["latent_2"] = l2
    ordered["archetype_cluster"] = clusters.astype(int)
    return ordered


def rolling_style_profile(
    history: pd.DataFrame,
    player: str,
    season: int,
    as_of: pd.Timestamp,
    window: int = STYLE_WINDOW,
) -> np.ndarray | None:
    """Mean style vector from prior games in this season before as_of."""
    p = history[
        (history["player"] == player)
        & (history["season"] == season)
        & (history["date"] < as_of)
    ].sort_values("date")
    if len(p) < STYLE_MIN_PERIODS:
        return None
    tail = p.tail(window)
    return tail[STYLE_FEATURES].mean().to_numpy(dtype=float)


def enrich_player_row(
    row: dict,
    history: pd.DataFrame,
    player: str,
    season: int,
    match_date: pd.Timestamp | None,
) -> dict:
    """Add archetype + form features to an inference row."""
    p = history[(history["player"] == player) & (history["season"] == season)].sort_values("date")
    as_of = match_date if match_date is not None else p["date"].max() + pd.Timedelta(days=1)

    roll3 = row.get("roll3_fantasy", np.nan)
    roll5 = row.get("roll5_fantasy", np.nan)
    row["form_trend"] = float(roll3 - roll5) if pd.notna(roll3) and pd.notna(roll5) else np.nan

    prior_scores = p[p["date"] < as_of]["fantasy_points"].tail(5)
    row["roll5_fantasy_std"] = float(prior_scores.std()) if len(prior_scores) >= STYLE_MIN_PERIODS else np.nan

    row["latent_1"] = 0.0
    row["latent_2"] = 0.0
    row["archetype_cluster"] = MISSING_CLUSTER

    if not artifacts_ready():
        return row

    scaler, encoder, kmeans, prior_clusters = load_artifacts()
    profile = rolling_style_profile(history, player, season, as_of)
    if profile is not None:
        X = scaler.transform(profile.reshape(1, -1))
        latent = encoder.predict(X, verbose=0)[0]
        row["latent_1"] = float(latent[0])
        row["latent_2"] = float(latent[1])
        row["archetype_cluster"] = int(kmeans.predict(latent.reshape(1, -1))[0])
    elif player in prior_clusters:
        row["archetype_cluster"] = int(prior_clusters[player])

    return row
