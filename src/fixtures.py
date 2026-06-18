r"""Upcoming AFL fixtures via the Squiggle API.

Squiggle publishes full-season schedules including unplayed games (complete < 100).
Team names are normalised to match FootyWire spelling in player_match_stats.csv.

Run standalone:
    .\.venv\Scripts\python.exe src\fixtures.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import requests

SQUIGGLE_BASE = "https://api.squiggle.com.au/"
HEADERS = {"User-Agent": "AFLFantasy-DNN/1.0 (personal project)"}

# Squiggle name -> dataset / FootyWire team name
TEAM_MAP: dict[str, str] = {
    "Adelaide": "Adelaide",
    "Brisbane Lions": "Brisbane",
    "Carlton": "Carlton",
    "Collingwood": "Collingwood",
    "Essendon": "Essendon",
    "Fremantle": "Fremantle",
    "Geelong": "Geelong",
    "Gold Coast": "Gold Coast",
    "Greater Western Sydney": "GWS",
    "Hawthorn": "Hawthorn",
    "Melbourne": "Melbourne",
    "North Melbourne": "North Melbourne",
    "Port Adelaide": "Port Adelaide",
    "Richmond": "Richmond",
    "St Kilda": "St Kilda",
    "Sydney": "Sydney",
    "West Coast": "West Coast",
    "Western Bulldogs": "Western Bulldogs",
}


@dataclass(frozen=True)
class Fixture:
    match_id: int
    date: pd.Timestamp
    home: str
    away: str
    venue: str | None
    round_num: int
    round_name: str

    @property
    def filename(self) -> str:
        return match_filename(self.round_num, self.home, self.away)


def normalise_team(name: str) -> str:
    if name not in TEAM_MAP:
        known = ", ".join(sorted(TEAM_MAP))
        raise ValueError(f"Unknown Squiggle team {name!r}. Known: {known}")
    return TEAM_MAP[name]


def slug_team(name: str) -> str:
    return name.lower().replace(" ", "_")


def match_filename(round_num: int, home: str, away: str) -> str:
    return f"round{round_num}_{slug_team(home)}_vs_{slug_team(away)}.csv"


def fetch_games(year: int) -> list[dict]:
    url = f"{SQUIGGLE_BASE}?q=games;year={year};format=json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    return r.json().get("games", [])


def _parse_game(raw: dict) -> Fixture | None:
    if raw.get("complete", 0) >= 100:
        return None
    if not raw.get("hteam") or not raw.get("ateam"):
        return None
    return Fixture(
        match_id=int(raw["id"]),
        date=pd.to_datetime(raw["date"]),
        home=normalise_team(raw["hteam"]),
        away=normalise_team(raw["ateam"]),
        venue=raw.get("venue") or None,
        round_num=int(raw["round"]),
        round_name=raw.get("roundname") or f"Round {raw['round']}",
    )


def next_unplayed_round(year: int) -> tuple[int, str, list[Fixture]]:
    """Return (round_num, round_name, fixtures) for the earliest round with unplayed games."""
    games = [_parse_game(g) for g in fetch_games(year)]
    upcoming = [g for g in games if g is not None]
    if not upcoming:
        raise RuntimeError(f"No unplayed Squiggle fixtures found for {year}")

    round_num = min(g.round_num for g in upcoming)
    round_games = sorted(
        [g for g in upcoming if g.round_num == round_num],
        key=lambda g: g.date,
    )
    round_name = round_games[0].round_name
    return round_num, round_name, round_games


def fixtures_dataframe(fixtures: list[Fixture]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "match_id": f.match_id,
            "date": f.date,
            "home": f.home,
            "away": f.away,
            "venue": f.venue,
            "round": f.round_name,
            "round_num": f.round_num,
            "filename": f.filename,
        }
        for f in fixtures
    ])


if __name__ == "__main__":
    year = datetime.now().year
    rnd, name, games = next_unplayed_round(year)
    print(f"Next unplayed round: {name} (round {rnd}) — {len(games)} games")
    print(fixtures_dataframe(games).to_string(index=False))
