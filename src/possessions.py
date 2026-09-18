"""Reconstruct on-court lineups and possessions from WNBA play-by-play.

Algorithm, per game:
  1. Seed each team's 5-man on-court lineup from the box score's `starter` flag.
  2. Walk events in order. `Substitution` rows (athlete_id_1 enters,
     athlete_id_2 exits) update lineup state immediately, independent of
     possession logic (WNBA substitutions only happen on dead balls, so a
     lineup is constant for the life of a possession).
  3. A possession is a stretch of events belonging to one team's offensive
     trip. It ends (and flips the ball to the other team) on: a turnover, a
     defensive rebound, a made non-free-throw field goal (unless the very
     next event is a shooting foul setting up an and-1 free throw), or the
     last free throw of a trip being made. An offensive rebound continues
     the SAME possession. Missed final free throws don't close the
     possession by themselves -- the rebound event that follows does.
     Period/game end closes whatever possession is open.

Known simplifications (see README limitations):
  - Technical free throws are scored but never open/close a possession.
  - Flagrant-foul possession retention (the fouled team keeps the ball
    afterward even without a change-of-possession event) isn't modeled --
    flagrant free throws are just added to the currently open possession.
  - Jump balls don't get special handling; the offense of the first
    possession in a period is inferred from whichever team's action (shot/
    turnover) appears first, which is the team that won the tip anyway.
"""
import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

TURNOVER_EXCLUDE = {"No Turnover"}
PERIOD_END_TYPES = {"End Period", "End Game"}
FT_LAST_RE = re.compile(r"(\d+) of (\d+)$")


def _next_free_throw_team(events, i):
    """Look past a made shot for an immediately-following free throw's team.

    Used to tell a genuine and-1 (the free-throw shooter is on the team that
    just scored) apart from the deceptively identical case of a made basket
    followed by the *other* team drawing an unrelated shooting foul on their
    very next possession -- both look like "made shot, then Shooting Foul"
    from the event stream alone.
    """
    j = i + 1
    while j < len(events):
        t = events[j]["type_text"] or ""
        if t == "Substitution" or t in ("Full Timeout", "Official Timeout", "No Timeout"):
            j += 1
            continue
        if t == "Shooting Foul":
            j += 1
            continue
        if t.startswith("Free Throw"):
            return events[j]["team_id"]
        return None
    return None


def _starting_lineups(player_box: pd.DataFrame, game_id) -> dict:
    g = player_box[player_box["game_id"] == game_id]
    lineups = {}
    for team_id, grp in g.groupby("team_id"):
        starters = set(grp.loc[grp["starter"], "athlete_id"])
        lineups[team_id] = starters
    return lineups


def reconstruct_game(pbp_game: pd.DataFrame, player_box: pd.DataFrame, game_id) -> tuple[list[dict], bool]:
    """Returns (possessions, is_clean). is_clean is False if a substitution
    referenced a player who wasn't on/off court as expected -- a source data
    inconsistency (rare, but real in this feed) rather than a bug in this
    reconstruction. Games flagged unclean are excluded from the RAPM training
    set entirely rather than trained on with a corrupted lineup."""
    events = pbp_game.sort_values("sequence_number").to_dict("records")
    lineups = _starting_lineups(player_box, game_id)
    team_ids = list(lineups.keys())
    if len(team_ids) != 2:
        return [], False
    is_clean = True

    possessions = []
    current_offense = None
    points = 0
    start_seq = None

    def other(team):
        return team_ids[0] if team == team_ids[1] else team_ids[1]

    def open_possession(team, seq):
        nonlocal current_offense, points, start_seq
        current_offense = team
        points = 0
        start_seq = seq

    def close_possession(seq):
        nonlocal current_offense, points, start_seq
        if current_offense is None:
            return
        off, dfn = current_offense, other(current_offense)
        if len(lineups.get(off, ())) == 5 and len(lineups.get(dfn, ())) == 5:
            possessions.append(
                {
                    "game_id": game_id,
                    "period": events_period_lookup.get(start_seq),
                    "offense_team_id": off,
                    "defense_team_id": dfn,
                    "offense_lineup": tuple(sorted(lineups[off])),
                    "defense_lineup": tuple(sorted(lineups[dfn])),
                    "points": points,
                    "start_seq": start_seq,
                    "end_seq": seq,
                }
            )
        current_offense = None
        points = 0
        start_seq = None

    events_period_lookup = {ev["sequence_number"]: ev["period_number"] for ev in events}

    for i, ev in enumerate(events):
        type_text = ev["type_text"] or ""
        team = ev["team_id"]
        seq = ev["sequence_number"]

        if type_text == "Substitution":
            if team in lineups:
                entering, exiting = ev["athlete_id_1"], ev["athlete_id_2"]
                if exiting not in lineups[team] or entering in lineups[team]:
                    is_clean = False
                lineups[team].discard(exiting)
                lineups[team].add(entering)
            continue

        if type_text in PERIOD_END_TYPES:
            close_possession(seq)
            continue

        if "Turnover" in type_text and type_text not in TURNOVER_EXCLUDE:
            if current_offense is None:
                open_possession(team, seq)
            close_possession(seq)
            continue

        if type_text == "Offensive Rebound":
            if current_offense is None:
                open_possession(team, seq)
            continue

        if type_text == "Defensive Rebound":
            if current_offense is None:
                open_possession(other(team), seq)
            close_possession(seq)
            continue

        if ev["shooting_play"] and not type_text.startswith("Free Throw"):
            if current_offense is None:
                open_possession(team, seq)
            if ev["scoring_play"]:
                points += ev["score_value"] or 0
                ft_team = _next_free_throw_team(events, i)
                and_one = ft_team is not None and ft_team == team
                if not and_one:
                    close_possession(seq)
            continue

        if type_text.startswith("Free Throw"):
            if current_offense is None:
                open_possession(team, seq)
            if ev["scoring_play"]:
                points += ev["score_value"] or 0
            if type_text == "Free Throw - Technical":
                continue
            m = FT_LAST_RE.search(type_text)
            is_last = m is not None and m.group(1) == m.group(2)
            if is_last and ev["scoring_play"]:
                close_possession(seq)
            continue

        # everything else (non-turnover fouls, timeouts, jump balls,
        # challenges, ejections) doesn't affect possession/lineup state
        continue

    close_possession(events[-1]["sequence_number"] if events else None)
    return possessions, is_clean


def build_possessions(season: int, force: bool = False) -> pd.DataFrame:
    out_path = PROCESSED_DIR / f"possessions_{season}.parquet"
    if out_path.exists() and not force:
        return pd.read_parquet(out_path)

    pbp = pd.read_parquet(RAW_DIR / f"pbp_{season}.parquet")
    player_box = pd.read_parquet(RAW_DIR / f"player_box_{season}.parquet")
    team_box = pd.read_parquet(RAW_DIR / f"team_box_{season}.parquet")

    # Exclude exhibitions (e.g. the All-Star Game, played by ad-hoc draft
    # teams like "TEAM CLARK") -- a real franchise plays dozens of games a
    # season, so a low games-played count reliably flags a one-off team
    # without hardcoding IDs or names that could change season to season.
    games_played = team_box.groupby("team_id")["game_id"].nunique()
    real_teams = set(games_played[games_played >= 10].index)
    exhibition_games = set(
        team_box.loc[~team_box["team_id"].isin(real_teams), "game_id"]
    )
    if exhibition_games:
        print(f"[excluded] {len(exhibition_games)} exhibition game(s) (e.g. All-Star Game)")
        pbp = pbp[~pbp["game_id"].isin(exhibition_games)]
        player_box = player_box[~player_box["game_id"].isin(exhibition_games)]

    all_possessions = []
    excluded_games = []
    for game_id, pbp_game in pbp.groupby("game_id"):
        game_possessions, is_clean = reconstruct_game(pbp_game, player_box, game_id)
        if is_clean:
            all_possessions.extend(game_possessions)
        else:
            excluded_games.append(game_id)

    df = pd.DataFrame(all_possessions)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"[saved] {len(df)} possessions across {df['game_id'].nunique()} games -> {out_path}")
    print(
        f"[excluded] {len(excluded_games)} / {pbp['game_id'].nunique()} games dropped "
        f"for substitution-tracking inconsistencies in the source data"
    )
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_possessions(args.season, force=args.force)
