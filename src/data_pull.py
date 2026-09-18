"""Pull WNBA play-by-play, schedule, and box scores, cache to disk.

Data source: the sportsdataverse project's published parquet releases
(https://github.com/sportsdataverse/sportsdataverse-data), the same
pre-built ESPN-backed data the `wehoop` R package and `sportsdataverse`
Python package load under the hood.

We fetch these parquet files directly with pandas rather than going
through the `sportsdataverse` pip package: that package's __init__ eagerly
imports every sport's module (including college football's xgboost-based
win-probability models), which requires `typing.TypeAlias` (Python 3.10+)
and a system OpenMP runtime -- neither available in this project's Python
3.9 environment, and both irrelevant to WNBA data. Hitting the same public
release URLs directly avoids that dependency entirely.
"""
import argparse
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

BASE_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"

DATASETS = {
    "schedule": ("espn_wnba_schedules", "wnba_schedule_{season}.parquet"),
    "pbp": ("espn_wnba_pbp", "play_by_play_{season}.parquet"),
    "team_box": ("espn_wnba_team_boxscores", "team_box_{season}.parquet"),
    "player_box": ("espn_wnba_player_boxscores", "player_box_{season}.parquet"),
}


def _cache_path(name: str, season: int) -> Path:
    return RAW_DIR / f"{name}_{season}.parquet"


def _release_url(name: str, season: int) -> str:
    tag, filename_tpl = DATASETS[name]
    return f"{BASE_URL}/{tag}/{filename_tpl.format(season=season)}"


def pull_season(season: int, force: bool = False) -> dict[str, pd.DataFrame]:
    """Pull and cache schedule, pbp, player box, and team box for one season."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    frames = {}
    for name in DATASETS:
        path = _cache_path(name, season)
        if path.exists() and not force:
            print(f"[cache hit] {name} {season} -> {path}")
            frames[name] = pd.read_parquet(path)
            continue
        url = _release_url(name, season)
        print(f"[fetching] {name} {season} <- {url}")
        df = pd.read_parquet(url)
        df.to_parquet(path)
        print(f"[saved] {name} {season}: {len(df)} rows -> {path}")
        frames[name] = df
    return frames


def sanity_check(frames: dict[str, pd.DataFrame]) -> None:
    schedule, pbp, player_box, team_box = (
        frames["schedule"],
        frames["pbp"],
        frames["player_box"],
        frames["team_box"],
    )
    print(f"games in schedule:   {schedule['game_id'].nunique()}")
    print(f"games with pbp:      {pbp['game_id'].nunique()}")
    print(f"games with team box: {team_box['game_id'].nunique()}")
    print(f"games with player box: {player_box['game_id'].nunique()}")

    sample_game = team_box["game_id"].iloc[0]
    box_rows = team_box.loc[team_box["game_id"] == sample_game]
    cols = [c for c in ("team_id", "team_display_name", "team_score") if c in box_rows.columns]
    print(f"\nSample game {sample_game} team box rows:\n{box_rows[cols].to_string(index=False)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    frames = pull_season(args.season, force=args.force)
    sanity_check(frames)
