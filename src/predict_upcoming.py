r"""Predict AFL Fantasy scores for the next unplayed round using LightGBM.

Workflow
--------
1. Load refreshed player_match_stats.csv (must include current season).
2. Fetch unplayed fixtures for the next round from Squiggle.
3. Halt if that round is already predicted (unless --force).
4. Load trained GBM from models/ (or --retrain first).
5. Write one CSV per match to web/data/predictions/ and rebuild manifest.json.

Run:
    .\.venv\Scripts\python.exe src\predict_upcoming.py
    .\.venv\Scripts\python.exe src\predict_upcoming.py --force
    .\.venv\Scripts\python.exe src\predict_upcoming.py --retrain
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

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import Fixture, fixtures_dataframe, next_unplayed_round
from predictor_model import (
    load_artifact,
    load_artifact_archetype,
    predict_player_scores,
    train_both,
)

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed" / "player_match_stats.csv"
DEBUG_OUT = ROOT / "data" / "processed" / "predictions_upcoming.csv"
PREDICTIONS_DIR = ROOT / "web" / "data" / "predictions"
MANIFEST_PATH = PREDICTIONS_DIR / "manifest.json"
STATE_PATH = PREDICTIONS_DIR / ".prediction_state.json"
STALE_DAYS = 21
PUBLISH_SCRIPT = ROOT / "scripts" / "publish-predictions.ps1"


def recent_squad(df: pd.DataFrame, team: str, season: int) -> set[str]:
    """Players who appeared for this team in their latest game this season."""
    t = df[(df["team"] == team) & (df["season"] == season)].sort_values("date")
    if t.empty:
        return set()
    last_date = t["date"].max()
    return set(t[t["date"] == last_date]["player"])


def predict_for_match(
    history: pd.DataFrame,
    home: str,
    away: str,
    season: int,
    match_date,
    venue: str | None,
    booster,
    meta: dict,
    booster_v2=None,
    meta_v2=None,
) -> pd.DataFrame:
    squad: list[tuple[str, str, str, str]] = []
    for team, opponent, ha in [(home, away, "home"), (away, home, "away")]:
        for player in recent_squad(history, team, season):
            squad.append((player, team, opponent, ha))

    rows = predict_player_scores(
        history, squad, season, venue, match_date, booster, meta, booster_v2, meta_v2
    )
    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows)
    out.insert(0, "date", match_date)
    return out.sort_values("predicted_fantasy", ascending=False)


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


def _match_date_from_csv(path: Path) -> str | None:
    try:
        df = pd.read_csv(path, nrows=1, usecols=["date"])
        if not df.empty:
            return str(pd.to_datetime(df["date"].iloc[0]).isoformat())
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        pass
    return None


def _normalize_match_date(value) -> str | None:
    if value is None:
        return None
    try:
        return pd.to_datetime(value).isoformat()
    except (TypeError, ValueError):
        return str(value) if value else None


def rebuild_manifest(state: dict, batch_round: int, batch_name: str) -> None:
    cutoff = datetime.now(timezone.utc).astimezone().timestamp() - STALE_DAYS * 86400
    games_meta = []

    for path in sorted(PREDICTIONS_DIR.glob("round*.csv")):
        meta = state.get("games", {}).get(path.name)
        if meta:
            predicted_at = meta.get("predicted_at", "")
        else:
            predicted_at = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).astimezone().isoformat()

        try:
            ts = datetime.fromisoformat(predicted_at).timestamp()
        except ValueError:
            ts = path.stat().st_mtime
        if ts < cutoff:
            continue

        if meta:
            match_date = _normalize_match_date(meta.get("match_date")) or _match_date_from_csv(path)
            games_meta.append({
                "filename": path.name,
                "home": meta["home"],
                "away": meta["away"],
                "round": meta.get("round", ""),
                "venue": meta.get("venue"),
                "predicted_at": predicted_at,
                "match_date": match_date,
            })
        else:
            m = path.stem
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
                "match_date": _match_date_from_csv(path),
            })

    games_meta.sort(key=lambda g: g.get("match_date") or "")
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


def main(force: bool = False, deploy: bool = False, retrain: bool = False) -> None:
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
        print("(use --force to regenerate)")
        sys.exit(0)

    if round_already_predicted(fixtures) and force:
        print("\n--force: regenerating predictions for this round")

    if retrain:
        print("\nretraining base + archetype GBM predictors...")
        (booster, meta), (booster_v2, meta_v2) = train_both(save=True)
    else:
        print("\nloading GBM predictors...")
        booster, meta = load_artifact()
        try:
            booster_v2, meta_v2 = load_artifact_archetype()
            print("loaded archetype model for predicted_fantasy_v2")
        except FileNotFoundError:
            booster_v2 = meta_v2 = None
            print("archetype model not found — writing predicted_fantasy only")
    print(f"blend roll5 weight: {meta.get('blend_weight_roll5', 0):.2f}")

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    predicted_at = iso_now()
    state = load_state()
    state.setdefault("games", {})
    all_preds = []

    for fix in fixtures:
        preds = predict_for_match(
            df, fix.home, fix.away, season, fix.date, fix.venue, booster, meta, booster_v2, meta_v2
        )
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
            "match_date": pd.to_datetime(fix.date).isoformat(),
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
    p = combined["predicted_fantasy"]
    print(f"\nscore spread: max={p.max():.1f} mean={p.mean():.1f} std={p.std():.1f} >100={(p > 100).sum()}")
    print(f"manifest -> {MANIFEST_PATH.relative_to(ROOT)} ({len(state['games'])} games tracked)")

    if deploy:
        print("\npublishing to RPi...")
        publish_to_rpi()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict fantasy scores for the next AFL round")
    parser.add_argument("--force", action="store_true", help="Re-predict even if round CSVs exist")
    parser.add_argument("--retrain", action="store_true", help="Retrain GBM before predicting")
    parser.add_argument("--deploy", action="store_true", help="Publish predictions to RPi after run")
    args = parser.parse_args()
    main(force=args.force, deploy=args.deploy, retrain=args.retrain)
