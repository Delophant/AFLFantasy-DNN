r"""Predict AFL Fantasy scores for upcoming matches using the trained LSTM.

Workflow
--------
1. Load refreshed player_match_stats.csv (must include 2026 through latest round).
2. Train the same LSTM architecture as sequence_model.py on 2021-2024 (val 2025).
3. Find FootyWire matches on the target date (default: tomorrow).
4. For each player with >=5 games already played in 2026, feed their last 5
   stat-lines and predict this week's fantasy score.

Limitations (be honest when reading output)
-------------------------------------------
* Only players with 5+ prior games *in 2026* get a prediction (no bridging off-season).
* Squads are inferred from each team's most recent 2026 lineup (not official teams).
* Model MAE ~18 pts on held-out 2025 — treat as a form guide, not gospel.

Run:
    .\.venv\Scripts\python.exe src\predict_upcoming.py
    .\.venv\Scripts\python.exe src\predict_upcoming.py 2026-06-12
"""

from __future__ import annotations

import re
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sequence_model import SEQ_FEATURES, SEQ_LEN, TRAIN_SEASONS, VAL_SEASON, build_sequences

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
keras.utils.set_random_seed(42)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
OUT = ROOT / "data" / "processed" / "predictions_upcoming.csv"
BASE = "https://www.footywire.com/afl/footy"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# FootyWire often lags the official fixture for unplayed games. Add rows here when needed.
# Team names must match FootyWire spelling in player_match_stats.csv.
MANUAL_FIXTURES: dict[date, list[dict]] = {
    date(2026, 6, 19): [
        {"home": "Gold Coast", "away": "Hawthorn", "round": "Round 15", "venue": "People's First Stadium"},
    ],
}


def _parse_match_date(title: str) -> pd.Timestamp | None:
    m = re.search(r"(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]+)\s+(\d{4})", title)
    if not m:
        return None
    day, month, year = m.groups()
    return pd.to_datetime(f"{day} {month} {year}", format="%d %B %Y", errors="coerce")


def _parse_match_teams(title: str) -> tuple[str, str] | None:
    # "Carlton defeats Richmond at MCG Round 5 ..." or "Carlton vs Richmond ..."
    m = re.search(
        r"AFL Match Statistics\s*:\s*(.+?)\s+(?:defeats|vs\.?|v)\s+(.+?)\s+at\s+",
        title,
        re.I,
    )
    if not m:
        return None
    return m.group(1).strip(), m.group(2).strip()


def fetch_fixtures(year: int) -> pd.DataFrame:
    """Return match_id, date, home, away, venue, round for every fixture that page lists."""
    html = requests.get(f"{BASE}/ft_match_list?year={year}", headers=HEADERS, timeout=30).text
    mids = list(dict.fromkeys(re.findall(r"ft_match_statistics\?mid=(\d+)", html)))
    rows = []
    cache = ROOT / "data" / "raw" / "html"
    for mid in mids:
        path = cache / f"{mid}.html"
        if path.exists():
            soup = BeautifulSoup(path.read_text(encoding="utf-8"), "lxml")
        else:
            try:
                r = requests.get(f"{BASE}/ft_match_statistics?mid={mid}", headers=HEADERS, timeout=30)
                r.raise_for_status()
                soup = BeautifulSoup(r.text, "lxml")
            except Exception:
                continue
        title = soup.title.text if soup.title else ""
        dt = _parse_match_date(title)
        teams = _parse_match_teams(title)
        if dt is None or teams is None:
            continue
        rnd = re.search(
            r"(Round\s+\d+|Qualifying Final|Elimination Final|Semi Final|"
            r"Preliminary Final|Grand Final)",
            title,
        )
        venue_m = re.search(r" at (.+?)\s+(Round\s+\d+|Qualifying|Elimination|Semi|Preliminary|Grand)", title)
        rows.append({
            "match_id": int(mid),
            "date": dt,
            "home": teams[0],
            "away": teams[1],
            "venue": venue_m.group(1).strip() if venue_m else None,
            "round": rnd.group(1) if rnd else None,
        })
    return pd.DataFrame(rows).drop_duplicates("match_id")


def train_lstm(df: pd.DataFrame):
    X, y, season, recent_avg, players, dates = build_sequences(df)
    tr = np.isin(season, list(TRAIN_SEASONS))
    va = season == VAL_SEASON
    flat = X[tr].reshape(-1, X.shape[-1])
    mu, sigma = flat.mean(0), flat.std(0) + 1e-8

    model = keras.Sequential([
        keras.layers.Input((SEQ_LEN, X.shape[-1])),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(1),
    ])
    model.compile(keras.optimizers.Adam(1e-3), "mse", metrics=["mae"])
    Xn_tr = (X[tr] - mu) / sigma
    Xn_va = (X[va] - mu) / sigma
    model.fit(
        Xn_tr, y[tr], validation_data=(Xn_va, y[va]),
        epochs=120, batch_size=128, verbose=0,
        callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=12, restore_best_weights=True)],
    )
    return model, mu, sigma


def recent_squad(df: pd.DataFrame, team: str, season: int) -> set[str]:
    """Players who appeared for this team in their latest game this season."""
    t = df[(df["team"] == team) & (df["season"] == season)].sort_values("date")
    if t.empty:
        return set()
    last_date = t["date"].max()
    return set(t[t["date"] == last_date]["player"])


def predict_for_match(
    df: pd.DataFrame, model, mu, sigma, home: str, away: str, season: int, match_date
) -> pd.DataFrame:
    feat = df[SEQ_FEATURES].to_numpy(dtype="float32")
    rows = []
    for team, opponent, ha in [(home, away, "home"), (away, home, "away")]:
        squad = recent_squad(df, team, season)
        for player in squad:
            p = df[(df["player"] == player) & (df["season"] == season)].sort_values("date")
            if len(p) < SEQ_LEN:
                continue
            idx = p.index.to_numpy()
            window = feat[idx[-SEQ_LEN:]]
            recent = p["fantasy_points"].iloc[-SEQ_LEN:].mean()
            x = ((window - mu) / sigma).reshape(1, SEQ_LEN, -1)
            pred = float(model.predict(x, verbose=0).ravel()[0])
            rows.append({
                "date": match_date,
                "team": team,
                "opponent": opponent,
                "home_away": ha,
                "player": player,
                "games_2026": len(p),
                "recent5_avg": round(recent, 1),
                "predicted_fantasy": round(pred, 1),
                "delta_vs_recent": round(pred - recent, 1),
            })
    return pd.DataFrame(rows).sort_values("predicted_fantasy", ascending=False)


def main(target: date | None = None) -> None:
    target = target or (date.today() + timedelta(days=1))
    target_ts = pd.Timestamp(target)

    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["player", "season", "date"])
    print(f"dataset: {len(df):,} rows | seasons {sorted(df['season'].unique())}")
    if 2026 not in df["season"].values:
        print("ERROR: no 2026 data — run scrape first.")
        sys.exit(1)

    print("training LSTM on 2021-2023 (val 2025)...")
    model, mu, sigma = train_lstm(df)

    fixtures = fetch_fixtures(2026)
    upcoming = fixtures[fixtures["date"].dt.date == target].copy()

    if upcoming.empty and target in MANUAL_FIXTURES:
        upcoming = pd.DataFrame(MANUAL_FIXTURES[target])
        upcoming["date"] = target_ts
        upcoming["match_id"] = None
        print(f"(using manual fixture list — FootyWire has no pages for {target} yet)")

    if upcoming.empty:
        print(f"\nNo FootyWire fixtures found for {target}.")
        if target not in MANUAL_FIXTURES:
            print(
                f"Add this date to MANUAL_FIXTURES in src/predict_upcoming.py, e.g.:\n"
                f"    date({target.year}, {target.month}, {target.day}): [\n"
                f'        {{"home": "Team A", "away": "Team B", "round": "Round N", "venue": "Stadium"}},\n'
                f"    ],\n"
                f"(use date(...) keys, not strings — team names must match the dataset)"
            )
        near = fixtures[(fixtures["date"] >= target_ts - pd.Timedelta(days=3)) &
                        (fixtures["date"] <= target_ts + pd.Timedelta(days=3))]
        if not near.empty:
            print("\nNearby FootyWire dates:")
            print(near[["date", "home", "away", "round", "venue"]].to_string(index=False))
        sys.exit(0)

    print(f"\nMatches on {target}:")
    print(upcoming[["home", "away", "round", "venue"]].to_string(index=False))

    all_preds = []
    for _, m in upcoming.iterrows():
        preds = predict_for_match(df, model, mu, sigma, m["home"], m["away"], 2026, m["date"])
        preds["match_id"] = m["match_id"]
        preds["round"] = m["round"]
        all_preds.append(preds)

    out = pd.concat(all_preds, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\nTop 15 predicted scorers for {target}:")
    cols = ["player", "team", "opponent", "recent5_avg", "predicted_fantasy", "delta_vs_recent"]
    print(out[cols].head(15).to_string(index=False))
    print(f"\nwrote {len(out)} predictions -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    d = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 else None
    main(d)
