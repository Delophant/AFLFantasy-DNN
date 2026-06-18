r"""Predict AFL Fantasy scores for the next unplayed round using the trained LSTM.



Workflow

--------

1. Load refreshed player_match_stats.csv (must include current season).

2. Fetch unplayed fixtures for the next round from Squiggle.

3. Halt if that round is already predicted (unless --force).

4. Train the same LSTM architecture as sequence_model.py on 2021-2023 (val 2024).

5. Write one CSV per match to web/data/predictions/ and rebuild manifest.json.



Run:

    .\.venv\Scripts\python.exe src\predict_upcoming.py

    .\.venv\Scripts\python.exe src\predict_upcoming.py --force

    .\.venv\Scripts\python.exe src\predict_upcoming.py --deploy

"""



from __future__ import annotations



import argparse

import json
import re
import subprocess

import sys

from datetime import datetime, timezone

from pathlib import Path



import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import Fixture, fixtures_dataframe, next_unplayed_round

sys.stdout.reconfigure(encoding="utf-8", errors="replace")



ROOT = Path(__file__).resolve().parent.parent

DATA = ROOT / "data" / "processed" / "player_match_stats.csv"

DEBUG_OUT = ROOT / "data" / "processed" / "predictions_upcoming.csv"

PREDICTIONS_DIR = ROOT / "web" / "data" / "predictions"

MANIFEST_PATH = PREDICTIONS_DIR / "manifest.json"

STATE_PATH = PREDICTIONS_DIR / ".prediction_state.json"

STALE_DAYS = 21

PUBLISH_SCRIPT = ROOT / "scripts" / "publish-predictions.ps1"





def train_lstm(df: pd.DataFrame):
    from tensorflow import keras

    from sequence_model import SEQ_LEN, TRAIN_SEASONS, VAL_SEASON, build_sequences

    keras.utils.set_random_seed(42)
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
    from sequence_model import SEQ_FEATURES, SEQ_LEN

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





def round_already_predicted(fixtures: list[Fixture]) -> bool:

    return all((PREDICTIONS_DIR / f.filename).exists() for f in fixtures)





def iso_now() -> str:

    return datetime.now(timezone.utc).astimezone().replace(microsecond=0).isoformat()





def load_state() -> dict:

    if STATE_PATH.exists():

        return json.loads(STATE_PATH.read_text(encoding="utf-8"))

    return {"games": {}}





def save_state(state: dict) -> None:

    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")





def rebuild_manifest(state: dict, batch_round: int, batch_name: str) -> None:

    cutoff = datetime.now(timezone.utc).astimezone().timestamp() - STALE_DAYS * 86400

    games_meta = []



    for path in sorted(PREDICTIONS_DIR.glob("round*.csv")):

        meta = state.get("games", {}).get(path.name)

        if meta:

            predicted_at = meta.get("predicted_at", "")

        else:

            predicted_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).astimezone().isoformat()



        try:

            ts = datetime.fromisoformat(predicted_at).timestamp()

        except ValueError:

            ts = path.stat().st_mtime

        if ts < cutoff:

            continue



        if meta:

            games_meta.append({

                "filename": path.name,

                "home": meta["home"],

                "away": meta["away"],

                "round": meta.get("round", ""),

                "venue": meta.get("venue"),

                "predicted_at": predicted_at,

            })

        else:

            # Fallback: parse round from filename

            m = path.stem  # round15_fremantle_vs_geelong

            parts = m.split("_vs_")

            if len(parts) == 2:

                home_slug = parts[0].split("_", 1)[-1].replace("_", " ").title()

                away_slug = parts[1].replace("_", " ").title()

            else:

                home_slug = away_slug = "Unknown"

            rnd_m = re.search(r"round(\d+)", path.name)

            games_meta.append({

                "filename": path.name,

                "home": home_slug,

                "away": away_slug,

                "round": f"Round {rnd_m.group(1)}" if rnd_m else "",

                "venue": None,

                "predicted_at": predicted_at,

            })



    games_meta.sort(key=lambda g: g.get("predicted_at", ""), reverse=True)



    manifest = {

        "last_updated": iso_now(),

        "prediction_batch": {"round": batch_round, "roundname": batch_name},

        "games": games_meta,

    }

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")





def publish_to_rpi() -> None:

    if not PUBLISH_SCRIPT.exists():

        raise FileNotFoundError(f"Missing publish script: {PUBLISH_SCRIPT}")

    subprocess.run(

        ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(PUBLISH_SCRIPT)],

        cwd=ROOT,

        check=True,

    )





def main(force: bool = False, deploy: bool = False) -> None:

    season = datetime.now().year



    df = pd.read_csv(DATA, parse_dates=["date"]).sort_values(["player", "season", "date"])

    print(f"dataset: {len(df):,} rows | seasons {sorted(df['season'].unique())}")

    if season not in df["season"].values:

        print(f"ERROR: no {season} data — run scrape first.")

        sys.exit(1)



    round_num, round_name, fixtures = next_unplayed_round(season)

    fixture_df = fixtures_dataframe(fixtures)

    print(f"\nNext round: {round_name} ({len(fixtures)} unplayed games)")

    print(fixture_df[["date", "home", "away", "venue"]].to_string(index=False))



    if round_already_predicted(fixtures) and not force:

        print(f"\nRound {round_num} already predicted — halting.")

        print(f"(use --force to regenerate)")

        sys.exit(0)



    if round_already_predicted(fixtures) and force:
        print("\n--force: regenerating predictions for this round")

    from sequence_model import TRAIN_SEASONS, VAL_SEASON

    print(f"\ntraining LSTM on {sorted(TRAIN_SEASONS)} (val {VAL_SEASON})...")

    model, mu, sigma = train_lstm(df)



    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    predicted_at = iso_now()

    state = load_state()

    state.setdefault("games", {})

    all_preds = []



    for fix in fixtures:

        preds = predict_for_match(df, model, mu, sigma, fix.home, fix.away, season, fix.date)

        preds["match_id"] = fix.match_id

        preds["round"] = fix.round_name

        out_path = PREDICTIONS_DIR / fix.filename

        preds.to_csv(out_path, index=False)

        state["games"][fix.filename] = {

            "home": fix.home,

            "away": fix.away,

            "round": fix.round_name,

            "venue": fix.venue,

            "predicted_at": predicted_at,

        }

        all_preds.append(preds)

        print(f"  wrote {out_path.relative_to(ROOT)} ({len(preds)} players)")



    state["last_round"] = round_num

    state["roundname"] = round_name

    state["predicted_at"] = predicted_at

    state["game_count"] = len(fixtures)

    save_state(state)

    rebuild_manifest(state, round_num, round_name)



    combined = pd.concat(all_preds, ignore_index=True)

    DEBUG_OUT.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(DEBUG_OUT, index=False)



    print(f"\nTop 15 predicted scorers for {round_name}:")

    cols = ["player", "team", "opponent", "recent5_avg", "predicted_fantasy", "delta_vs_recent"]

    print(combined[cols].head(15).to_string(index=False))

    print(f"\nmanifest -> {MANIFEST_PATH.relative_to(ROOT)} ({len(state['games'])} games tracked)")



    if deploy:

        print("\npublishing to RPi...")

        publish_to_rpi()





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Predict fantasy scores for the next AFL round")

    parser.add_argument("--force", action="store_true", help="Re-predict even if round CSVs exist")

    parser.add_argument("--deploy", action="store_true", help="Publish predictions to RPi after run")

    args = parser.parse_args()

    main(force=args.force, deploy=args.deploy)


