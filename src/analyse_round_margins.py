r"""Largest winning margin per AFL home-and-away round, then season summaries.

For each completed, non-finals match the margin is abs(home_score - away_score).
Each round's metric is the single largest of those match margins. Each season is
then summarised with the mean, median, and mode of its per-round series.

Sources scores from the Squiggle API (same feed as fixtures.py) rather than
the FootyWire player scraper, which does not store AFL match totals.

Run:
    python src/analyse_round_margins.py
    python src/analyse_round_margins.py 2021 2026
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, don't try to open a window
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fixtures import fetch_games, normalise_team

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

DEFAULT_START = 2021
DEFAULT_END = 2026


def banner(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _is_final(raw: dict) -> bool:
    value = raw.get("is_final", 0)
    try:
        return int(value) != 0
    except (TypeError, ValueError):
        return bool(value)


def _is_complete(raw: dict) -> bool:
    try:
        return int(raw.get("complete", 0) or 0) >= 100
    except (TypeError, ValueError):
        return False


def games_to_match_frame(games: list[dict], season: int) -> pd.DataFrame:
    """Keep finished home-and-away games and attach the winning margin."""
    rows = []
    for raw in games:
        if not _is_complete(raw) or _is_final(raw):
            continue
        hscore = raw.get("hscore")
        ascore = raw.get("ascore")
        if hscore is None or ascore is None:
            continue
        if not raw.get("hteam") or not raw.get("ateam"):
            continue
        hscore = int(hscore)
        ascore = int(ascore)
        rows.append(
            {
                "season": season,
                "round_num": int(raw["round"]),
                "round_name": raw.get("roundname") or f"Round {raw['round']}",
                "date": pd.to_datetime(raw.get("date")),
                "home": normalise_team(raw["hteam"]),
                "away": normalise_team(raw["ateam"]),
                "hscore": hscore,
                "ascore": ascore,
                "margin": abs(hscore - ascore),
                "match_id": int(raw["id"]),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "season", "round_num", "round_name", "date", "home", "away",
                "hscore", "ascore", "margin", "match_id",
            ]
        )
    return pd.DataFrame(rows)


def fetch_home_and_away(start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        games = fetch_games(year)
        frame = games_to_match_frame(games, year)
        print(f"[{year}] {len(games)} Squiggle games -> {len(frame)} completed H&A matches")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.empty:
        return combined
    return combined.sort_values(["season", "round_num", "date", "match_id"]).reset_index(drop=True)


def round_max_margins(matches: pd.DataFrame) -> pd.DataFrame:
    """One row per round: the game with the largest margin (earliest if tied)."""
    keys = ["season", "round_num", "round_name"]
    games = (
        matches.groupby(keys, as_index=False)
        .size()
        .rename(columns={"size": "games"})
    )
    ordered = matches.sort_values(
        ["season", "round_num", "margin", "date", "match_id"],
        ascending=[True, True, False, True, True],
    )
    blowouts = ordered.drop_duplicates(subset=keys, keep="first").copy()
    blowouts = blowouts.rename(columns={"margin": "max_margin"})
    blowouts = blowouts.merge(games, on=keys, how="left")
    cols = [
        "season", "round_num", "round_name", "games", "max_margin",
        "date", "home", "away", "hscore", "ascore", "match_id",
    ]
    return blowouts[cols].sort_values(["season", "round_num"]).reset_index(drop=True)


def _format_modes(values: pd.Series) -> str:
    modes = values.mode(dropna=True)
    if modes.empty:
        return ""
    return ", ".join(f"{m:g}" for m in modes.tolist())


def season_summaries(rounds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for season, group in rounds.groupby("season", sort=True):
        series = group["max_margin"]
        rows.append(
            {
                "season": int(season),
                "rounds": int(len(group)),
                "mean_max_margin": float(series.mean()),
                "median_max_margin": float(series.median()),
                "mode_max_margin": _format_modes(series),
            }
        )
    return pd.DataFrame(rows)


def save_figure(rounds: pd.DataFrame, seasons: pd.DataFrame) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [1.4, 1]})

    ax = axes[0]
    for season, group in rounds.groupby("season"):
        ax.plot(
            group["round_num"],
            group["max_margin"],
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            label=str(int(season)),
        )
    ax.set_title("Largest winning margin by round")
    ax.set_xlabel("Round")
    ax.set_ylabel("Max margin (points)")
    ax.legend(title="Season", ncols=3, fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    x = range(len(seasons))
    width = 0.35
    ax.bar(
        [i - width / 2 for i in x],
        seasons["mean_max_margin"],
        width,
        label="Mean of round highs",
        color="#3b7dd8",
    )
    ax.bar(
        [i + width / 2 for i in x],
        seasons["median_max_margin"],
        width,
        label="Median of round highs",
        color="#2bb673",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(seasons["season"].astype(int).astype(str))
    ax.set_title("Season summary of round-high margins")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "round_max_margins.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def _histogram_bins(values: pd.Series) -> list[float]:
    """Shared 10-point bins covering both the 2026 and all-years series."""
    lo = int(values.min() // 10 * 10)
    hi = int(values.max() // 10 * 10) + 10
    return list(range(max(0, lo), hi + 10, 10))


def save_histograms(rounds: pd.DataFrame) -> Path:
    """Histogram of per-round max margins: 2026 vs all seasons combined."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    all_years = rounds["max_margin"]
    y2026 = rounds.loc[rounds["season"] == 2026, "max_margin"]
    if y2026.empty:
        raise SystemExit("No 2026 rounds found to histogram")

    bins = _histogram_bins(all_years)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True)

    panels = [
        (axes[0], y2026, "2026 round-high margins", "#3b7dd8"),
        (axes[1], all_years, "2021–2026 round-high margins", "#2bb673"),
    ]
    for ax, series, title, color in panels:
        ax.hist(series, bins=bins, color=color, edgecolor="white")
        ax.axvline(series.mean(), color="#333333", linestyle="--", linewidth=1.2, label=f"mean {series.mean():.1f}")
        ax.axvline(series.median(), color="#c0392b", linestyle=":", linewidth=1.4, label=f"median {series.median():.1f}")
        ax.set_title(f"{title} (n={len(series)})")
        ax.set_ylabel("Rounds")
        ax.set_xticks(bins)
        ax.tick_params(axis="x", labelbottom=True, labelrotation=45)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    axes[0].set_xlabel("Largest winning margin (points)")
    axes[1].set_xlabel("Largest winning margin (points)")
    fig.tight_layout()
    out = FIG_DIR / "round_max_margins_histograms.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def main(start_year: int = DEFAULT_START, end_year: int = DEFAULT_END) -> None:
    banner("FETCH COMPLETED HOME-AND-AWAY GAMES")
    matches = fetch_home_and_away(start_year, end_year)
    if matches.empty:
        raise SystemExit(f"No completed non-finals matches found for {start_year}–{end_year}")

    print(f"\nmatches : {len(matches):,}")
    print(f"seasons : {sorted(matches['season'].unique().tolist())}")
    print(f"date    : {matches['date'].min().date()} -> {matches['date'].max().date()}")

    rounds = round_max_margins(matches)
    seasons = season_summaries(rounds)

    banner("LARGEST MARGIN BY ROUND")
    print(rounds.to_string(index=False))

    banner("SEASON SUMMARY (of per-round max margins)")
    display_seasons = seasons.copy()
    display_seasons["mean_max_margin"] = display_seasons["mean_max_margin"].round(1)
    display_seasons["median_max_margin"] = display_seasons["median_max_margin"].round(1)
    print(display_seasons.to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    match_path = PROCESSED_DIR / "match_margins.csv"
    round_path = PROCESSED_DIR / "round_max_margins.csv"
    season_path = PROCESSED_DIR / "season_margin_summary.csv"
    matches.to_csv(match_path, index=False)
    rounds.to_csv(round_path, index=False)
    seasons.to_csv(season_path, index=False)

    fig_path = save_figure(rounds, seasons)
    hist_path = save_histograms(rounds)

    old_round = PROCESSED_DIR / "round_median_margins.csv"
    old_fig = FIG_DIR / "round_median_margins.png"
    for stale in (old_round, old_fig):
        if stale.exists():
            stale.unlink()
            print(f"removed {stale.relative_to(ROOT)}")

    print("\nwrote:")
    for path in (match_path, round_path, season_path, fig_path, hist_path):
        print(f"  {path.relative_to(ROOT)}")


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_START
    end = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_END
    main(start, end)
