r"""Scrape per-player, per-match AFL stats (incl. AFL Fantasy points) from FootyWire.

Output is a *tidy long table*: one row per player per match, with the raw stat
categories plus the AFL Fantasy (AF) and SuperCoach (SC) scores and match metadata.

Design notes
------------
* We cache each match's raw HTML under data/raw/html/<mid>.html. Re-runs reparse
  from cache instantly and avoid hammering the site. Delete the cache to refetch.
* Team identity comes from the "<Team> Match Statistics" headings, which appear in
  the same document order as the player tables (home team first, away second).
* Be polite: a short sleep between live fetches.

Usage
-----
    .\.venv\Scripts\python.exe src\scrape_footywire.py 2021 2024
    # -> data/raw/footywire_<year>.csv per season
    # -> data/processed/player_match_stats.csv (all seasons combined)
"""

from __future__ import annotations

import re
import sys
import time
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.footywire.com/afl/footy"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
SLEEP_SECONDS = 1.0  # politeness delay between live fetches

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
HTML_CACHE = RAW_DIR / "html"
PROCESSED_DIR = ROOT / "data" / "processed"

# FootyWire's short stat codes -> readable column names.
STAT_COLUMNS = {
    "K": "kicks",
    "HB": "handballs",
    "D": "disposals",
    "M": "marks",
    "G": "goals",
    "B": "behinds",
    "T": "tackles",
    "HO": "hitouts",
    "GA": "goal_assists",
    "I50": "inside50s",
    "CL": "clearances",
    "CG": "clangers",
    "R50": "rebound50s",
    "FF": "frees_for",
    "FA": "frees_against",
    "AF": "fantasy_points",
    "SC": "supercoach_points",
}


def _fetch(url: str) -> str:
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    return resp.text


def get_match_ids(year: int) -> list[int]:
    """Return all match ids for a season, in fixture order."""
    html = _fetch(f"{BASE}/ft_match_list?year={year}")
    mids = re.findall(r"ft_match_statistics\?mid=(\d+)", html)
    # Preserve order, drop duplicates.
    seen: dict[int, None] = {}
    for m in mids:
        seen.setdefault(int(m), None)
    return list(seen)


def _match_html(mid: int) -> str:
    """Fetch a match page, using the on-disk cache when available."""
    HTML_CACHE.mkdir(parents=True, exist_ok=True)
    cached = HTML_CACHE / f"{mid}.html"
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    html = _fetch(f"{BASE}/ft_match_statistics?mid={mid}")
    cached.write_text(html, encoding="utf-8")
    time.sleep(SLEEP_SECONDS)
    return html


def _clean_name(raw: str) -> str:
    """Strip trailing role/sub glyphs and whitespace from a player name."""
    # Keep letters, spaces, and the punctuation found in real names.
    name = re.sub(r"[^A-Za-z .'\-]", " ", str(raw))
    return re.sub(r"\s+", " ", name).strip()


def _parse_metadata(soup: BeautifulSoup) -> dict:
    """Extract round, date and venue from the match page title."""
    title = soup.title.text if soup.title else ""
    title = title.replace("AFL Match Statistics :", "").strip()

    meta: dict = {"round": None, "date": None, "venue": None, "title": title}

    # Date, e.g. "Thursday, 14th March 2024"
    date_match = re.search(r"(\d{1,2})(?:st|nd|rd|th)\s+([A-Za-z]+)\s+(\d{4})", title)
    if date_match:
        day, month, year = date_match.groups()
        meta["date"] = pd.to_datetime(f"{day} {month} {year}", format="%d %B %Y", errors="coerce")

    # Round label, e.g. "Round 1" or a finals name.
    round_match = re.search(
        r"(Round\s+\d+|Qualifying Final|Elimination Final|Semi Final|"
        r"Preliminary Final|Grand Final)",
        title,
    )
    if round_match:
        meta["round"] = round_match.group(1)

    # Venue: the text between " at " and the round/date tail.
    venue_match = re.search(r" at (.+?)\s+(Round\s+\d+|Qualifying|Elimination|Semi|Preliminary|Grand)", title)
    if venue_match:
        meta["venue"] = venue_match.group(1).strip()
    return meta


def _team_names(soup: BeautifulSoup) -> list[str]:
    """Team names in document order (home first, away second)."""
    text = soup.get_text(" ")
    return [m.group(1).strip() for m in re.finditer(r"([A-Za-z ]+?) Match Statistics \(Sorted by Disposals\)", text)]


def _player_frames(soup: BeautifulSoup) -> list[pd.DataFrame]:
    """The two player-stat tables (18 cols, header row starts with 'Player')."""
    frames = pd.read_html(StringIO(str(soup)))
    out = []
    for df in frames:
        if df.shape[1] == 18 and str(df.iloc[0, 0]) == "Player":
            out.append(df)
    return out


def parse_match(mid: int, season: int) -> pd.DataFrame:
    """Parse one match into a tidy long DataFrame of player rows."""
    soup = BeautifulSoup(_match_html(mid), "lxml")
    meta = _parse_metadata(soup)
    teams = _team_names(soup)
    frames = _player_frames(soup)

    if len(frames) != 2 or len(teams) < 2:
        raise ValueError(f"mid={mid}: expected 2 teams/frames, got {len(teams)} teams / {len(frames)} frames")

    rows = []
    for idx, df in enumerate(frames):
        team = teams[idx]
        opponent = teams[1 - idx]
        home_away = "home" if idx == 0 else "away"

        body = df.iloc[1:].copy()  # drop the header row
        body.columns = ["Player"] + list(STAT_COLUMNS.keys())
        body = body.rename(columns={"Player": "player", **STAT_COLUMNS})
        body["player"] = body["player"].map(_clean_name)

        for col in STAT_COLUMNS.values():
            body[col] = pd.to_numeric(body[col], errors="coerce")

        body = body[body["player"].str.len() > 0].dropna(subset=["fantasy_points"])
        body.insert(0, "season", season)
        body.insert(1, "match_id", mid)
        body.insert(2, "round", meta["round"])
        body.insert(3, "date", meta["date"])
        body.insert(4, "venue", meta["venue"])
        body.insert(5, "team", team)
        body.insert(6, "opponent", opponent)
        body.insert(7, "home_away", home_away)
        rows.append(body)

    return pd.concat(rows, ignore_index=True)


def scrape_season(year: int) -> pd.DataFrame:
    mids = get_match_ids(year)
    print(f"[{year}] {len(mids)} matches found")
    frames = []
    for n, mid in enumerate(mids, start=1):
        try:
            frames.append(parse_match(mid, year))
        except Exception as exc:  # keep going; report the bad match
            print(f"   ! mid={mid} skipped: {exc}")
            continue
        if n % 25 == 0 or n == len(mids):
            print(f"   [{year}] parsed {n}/{len(mids)} matches")
    season_df = pd.concat(frames, ignore_index=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out = RAW_DIR / f"footywire_{year}.csv"
    season_df.to_csv(out, index=False)
    print(f"[{year}] wrote {len(season_df)} player-rows -> {out.relative_to(ROOT)}")
    return season_df


def main(start_year: int, end_year: int) -> None:
    all_frames = [scrape_season(y) for y in range(start_year, end_year + 1)]
    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.sort_values(["date", "match_id", "team"]).reset_index(drop=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "player_match_stats.csv"
    combined.to_csv(out, index=False)
    print(
        f"\nDONE: {len(combined):,} player-rows across "
        f"{combined['match_id'].nunique()} matches, "
        f"{combined['player'].nunique()} players -> {out.relative_to(ROOT)}"
    )


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 2021
    end = int(sys.argv[2]) if len(sys.argv) > 2 else 2024
    main(start, end)
