from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
import math
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
import requests
import streamlit as st

# Optional Gemini + YouTube support
try:
    from google import genai
    from google.genai import types
except Exception:
    genai = None
    types = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
except Exception:
    YouTubeTranscriptApi = None


# ============================================================
# APP CONFIG
# ============================================================

st.set_page_config(
    page_title="FPL Assistant Manager 2.0",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

API = "https://fantasy.premierleague.com/api"
HEADERS = {
    "User-Agent": "Mozilla/5.0 FPL Assistant Manager 2.0",
    "Accept": "application/json,text/plain,*/*",
}

# Historical archive. Public community archive of FPL snapshots.
VAAS_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
HISTORICAL_SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26"]

MAX_PER_CLUB = 3
TRANSFER_HIT = 4
HORIZON = 8
PROJECTION_WEEKS = 5
ROLLING_WINDOWS = (3, 5, 8)

FORMATIONS = [
    (3, 4, 3), (3, 5, 2), (4, 4, 2), (4, 3, 3),
    (4, 5, 1), (5, 4, 1), (5, 3, 2), (5, 2, 3),
]

POSITION_NAMES = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

GEMINI_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-3.5-flash"]

# User can change these in the sidebar without editing code.
DEFAULT_TEAM_ID = 3240706
DEFAULT_LEAGUES = {"Lads League": 70818, "IMW": 637276}
DEFAULT_ELITES = {
    "Ben Crellin": 53517,
    "FPL Harry": 3054,
    "Andy LTFPL": 41,
    "Tom Dollimore": 179777,
    "Pras United": 3315,
    "Sam Bonfield": 2977,
    "BigMan Bakar": 5133,
}
CREATOR_CHANNELS = {
    "FPL Harry": "https://www.youtube.com/@FPLHarry",
    "Let's Talk FPL": "https://www.youtube.com/@LetsTalkFPL",
    "FPL Focal": "https://www.youtube.com/@FPLFocal",
    "FPL Mate": "https://www.youtube.com/@FPLMate",
    "Planet FPL": "https://www.youtube.com/@PlanetFPL",
}


# ============================================================
# GENERIC HELPERS
# ============================================================

def num(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def mean(values: list[float], default: float = 0.0) -> float:
    vals = [x for x in values if x is not None and math.isfinite(x)]
    return sum(vals) / len(vals) if vals else default


def percentile(values: list[float], q: float, default: float = 0.0) -> float:
    vals = sorted(x for x in values if x is not None and math.isfinite(x))
    if not vals:
        return default
    idx = (len(vals) - 1) * q
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return vals[lo]
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


def status_factor(chance: float, status: str = "a") -> float:
    if status in {"u", "s", "i"}:
        return 0.0 if chance <= 0 else 0.35
    if chance >= 90:
        return 1.0
    if chance >= 75:
        return 0.85
    if chance >= 50:
        return 0.60
    if chance > 0:
        return 0.30
    return 0.0


def get_secret(name: str, default: str | None = None) -> str | None:
    try:
        value = st.secrets.get(name)
        return str(value) if value is not None else default
    except Exception:
        return default


# ============================================================
# HTTP / FPL API
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def api_get(url: str) -> dict:
    r = requests.get(url, headers=HEADERS, timeout=25)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=900, show_spinner=False)
def get_entry_info(entry_id: int) -> dict:
    return api_get(f"{API}/entry/{entry_id}/")


@st.cache_data(ttl=600, show_spinner=False)
def get_entry_history(entry_id: int) -> dict:
    return api_get(f"{API}/entry/{entry_id}/history/")


@st.cache_data(ttl=300, show_spinner=False)
def get_entry_picks(entry_id: int, gw: int) -> dict:
    return api_get(f"{API}/entry/{entry_id}/event/{gw}/picks/")


@st.cache_data(ttl=600, show_spinner=False)
def get_entry_transfers(entry_id: int) -> list[dict]:
    data = api_get(f"{API}/entry/{entry_id}/transfers/")
    return data if isinstance(data, list) else data.get("transfers", [])


@st.cache_data(ttl=600, show_spinner=False)
def get_league(league_id: int) -> dict:
    return api_get(f"{API}/leagues-classic/{league_id}/standings/")


@st.cache_data(ttl=300, show_spinner=False)
def get_live_gw(gw: int) -> dict[int, int]:
    data = api_get(f"{API}/event/{gw}/live/")
    return {
        safe_int(e.get("id")): safe_int(e.get("stats", {}).get("total_points"))
        for e in data.get("elements", [])
    }


# ============================================================
# OPTIONAL UNDERSTAT CURRENT xG/xA
# ============================================================

@st.cache_data(ttl=21600, show_spinner=False)
def fetch_understat_players(season_year: int) -> pd.DataFrame:
    """Best-effort Understat player data. The app always works without it."""
    url = f"https://understat.com/getLeagueData/EPL/{season_year}"
    headers = {
        "User-Agent": HEADERS["User-Agent"],
        "Referer": "https://understat.com/league/EPL",
        "X-Requested-With": "XMLHttpRequest",
    }
    try:
        r = requests.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        raw = r.json()
        players = raw.get("players", raw if isinstance(raw, list) else [])
        rows = []
        for p in players:
            rows.append({
                "under_name": str(p.get("player_name", "")),
                "under_team": str(p.get("team_title", "")),
                "under_xg": num(p.get("xG")),
                "under_xa": num(p.get("xA")),
                "under_npxg": num(p.get("npxG")),
                "under_minutes": safe_int(p.get("time")),
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def normalise_name(s: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return s


def attach_understat(players: list[dict], season_year: int) -> list[dict]:
    df = fetch_understat_players(season_year)
    if df.empty:
        return players
    lookup = {}
    for _, row in df.iterrows():
        lookup.setdefault(normalise_name(row["under_name"]), row.to_dict())
    for p in players:
        keys = [normalise_name(p.get("full_name", "")), normalise_name(p.get("name", ""))]
        hit = next((lookup[k] for k in keys if k in lookup), None)
        if hit:
            p["under_xg"] = num(hit.get("under_xg"))
            p["under_xa"] = num(hit.get("under_xa"))
            p["npxg"] = num(hit.get("under_npxg"))
            p["xg_source"] = "Understat"
        else:
            p.setdefault("under_xg", p.get("xg", 0.0))
            p.setdefault("under_xa", p.get("xa", 0.0))
            # Official FPL expected_goals includes penalties; therefore this is explicitly a proxy.
            p.setdefault("npxg", max(0.0, p.get("xg", 0.0) - 0.0))
            p["xg_source"] = "FPL proxy"
    return players


# ============================================================
# HISTORICAL DATA LAYER
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_csv(url: str) -> pd.DataFrame:
    try:
        return pd.read_csv(url)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner="Loading historical FPL archive...")
def load_historical_database() -> dict:
    seasons: dict[str, dict] = {}
    for season in HISTORICAL_SEASONS:
        players_raw = fetch_csv(f"{VAAS_BASE}/{season}/players_raw.csv")
        fixtures = fetch_csv(f"{VAAS_BASE}/{season}/fixtures.csv")
        merged_gw = fetch_csv(f"{VAAS_BASE}/{season}/gws/merged_gw.csv")
        if merged_gw.empty:
            merged_gw = fetch_csv(f"{VAAS_BASE}/{season}/gws/merged_gws.csv")
        teams = fetch_csv(f"{VAAS_BASE}/{season}/teams.csv")
        seasons[season] = {
            "players": players_raw,
            "fixtures": fixtures,
            "gws": merged_gw,
            "teams": teams,
        }
    return seasons


def historical_summary(hist: dict) -> dict:
    rows = []
    for season, d in hist.items():
        p = d["players"]
        f = d["fixtures"]
        g = d["gws"]
        rows.append({
            "Season": season,
            "Players": len(p),
            "Fixtures": len(f),
            "GW rows": len(g),
            "Status": "Loaded" if not p.empty and not f.empty else "Partial/unavailable",
        })
    return rows


def historical_player_prior(hist: dict, current_name: str) -> dict:
    """Find recent historical FPL rates by fuzzy-ish normalised name."""
    key = normalise_name(current_name)
    season_rows = []
    for season in reversed(HISTORICAL_SEASONS):
        df = hist.get(season, {}).get("players", pd.DataFrame())
        if df.empty:
            continue
        name_col = "web_name" if "web_name" in df.columns else "second_name"
        hits = df[df[name_col].astype(str).map(normalise_name) == key]
        if hits.empty and "first_name" in df.columns and "second_name" in df.columns:
            full = (hits["first_name"].astype(str) + hits["second_name"].astype(str)).map(normalise_name)
            hits = df[full == key]
        if not hits.empty:
            r = hits.iloc[0]
            mins = num(r.get("minutes"))
            xgi = num(r.get("expected_goal_involvements_per_90"))
            ppg = num(r.get("points_per_game"))
            season_rows.append({"season": season, "minutes": mins, "xgi90": xgi, "ppg": ppg})
    if not season_rows:
        return {"hist_xgi90": 0.0, "hist_ppg": 0.0, "hist_minutes": 0.0, "seasons": 0}
    weights = [0.50, 0.30, 0.15, 0.05][:len(season_rows)]
    total_w = sum(weights)
    return {
        "hist_xgi90": sum(r["xgi90"] * w for r, w in zip(season_rows, weights)) / total_w,
        "hist_ppg": sum(r["ppg"] * w for r, w in zip(season_rows, weights)) / total_w,
        "hist_minutes": sum(r["minutes"] * w for r, w in zip(season_rows, weights)) / total_w,
        "seasons": len(season_rows),
    }


def historical_dgw_bgw_windows(hist: dict) -> pd.DataFrame:
    rows = []
    for season, d in hist.items():
        fx = d["fixtures"]
        if fx.empty or "event" not in fx.columns:
            continue
        fx = fx[fx["event"].notna()].copy()
        if fx.empty:
            continue
        counts = defaultdict(lambda: defaultdict(int))
        for _, r in fx.iterrows():
            gw = safe_int(r.get("event"), -1)
            if gw < 0:
                continue
            counts[gw][safe_int(r.get("team_h"))] += 1
            counts[gw][safe_int(r.get("team_a"))] += 1
        for gw, team_counts in counts.items():
            vals = list(team_counts.values())
            doubles = sum(v >= 2 for v in vals)
            blanks = max(0, 20 - len(vals)) if vals else 0
            if doubles or blanks:
                rows.append({"Season": season, "GW": gw, "Double teams": doubles, "Blank teams": blanks})
    return pd.DataFrame(rows)


# ============================================================
# CURRENT FPL DATA
# ============================================================

@st.cache_data(ttl=900, show_spinner="Loading official FPL data...")
def load_current_data() -> dict:
    bootstrap = api_get(f"{API}/bootstrap-static/")
    fixtures_raw = api_get(f"{API}/fixtures/")
    events = bootstrap.get("events", [])
    raw_teams = bootstrap.get("teams", [])
    raw_players = bootstrap.get("elements", [])
    teams = {t["id"]: t for t in raw_teams}
    team_names = {t["id"]: t.get("short_name", "?") for t in raw_teams}

    current_event = next((e for e in events if e.get("is_current")), None)
    next_event = next((e for e in events if e.get("is_next")), None)
    if current_event:
        current_gw = safe_int(current_event.get("id"), 1)
    elif next_event:
        current_gw = max(1, safe_int(next_event.get("id"), 1) - 1)
    else:
        current_gw = 1
    next_gw = safe_int(next_event.get("id"), current_gw + 1) if next_event else current_gw + 1

    fixture_map = defaultdict(list)
    for f in fixtures_raw:
        gw = f.get("event")
        if gw is None:
            continue
        gw = safe_int(gw)
        if gw < next_gw or gw > next_gw + HORIZON - 1:
            continue
        h, a = f.get("team_h"), f.get("team_a")
        if h:
            fixture_map[h].append({
                "gw": gw, "home": True, "opponent": a,
                "difficulty": safe_int(f.get("team_h_difficulty"), 3),
                "finished": bool(f.get("finished")),
            })
        if a:
            fixture_map[a].append({
                "gw": gw, "home": False, "opponent": h,
                "difficulty": safe_int(f.get("team_a_difficulty"), 3),
                "finished": bool(f.get("finished")),
            })

    players = []
    for raw in raw_players:
        team_id = safe_int(raw.get("team"))
        chance = raw.get("chance_of_playing_next_round")
        chance = 100 if chance is None else num(chance)
        p = {
            "id": safe_int(raw.get("id")),
            "name": raw.get("web_name", "?"),
            "full_name": f"{raw.get('first_name','')} {raw.get('second_name','')}".strip(),
            "position": POSITION_NAMES.get(safe_int(raw.get("element_type")), "?"),
            "team_id": team_id,
            "team": team_names.get(team_id, "?"),
            "price": num(raw.get("now_cost")) / 10,
            "points": safe_int(raw.get("total_points")),
            "ppg": num(raw.get("points_per_game")),
            "form": num(raw.get("form")),
            "minutes": safe_int(raw.get("minutes")),
            "goals": safe_int(raw.get("goals_scored")),
            "assists": safe_int(raw.get("assists")),
            "clean_sheets": safe_int(raw.get("clean_sheets")),
            "bonus": safe_int(raw.get("bonus")),
            "bps": safe_int(raw.get("bps")),
            "ep_next": num(raw.get("ep_next")),
            "ep_this": num(raw.get("ep_this")),
            "ownership": num(raw.get("selected_by_percent")),
            "chance": chance,
            "status": raw.get("status", "a"),
            "news": raw.get("news", ""),
            "xg": num(raw.get("expected_goals")),
            "xa": num(raw.get("expected_assists")),
            "xgi": num(raw.get("expected_goal_involvements")),
            "xg90": num(raw.get("expected_goals_per_90")),
            "xa90": num(raw.get("expected_assists_per_90")),
            "xgi90": num(raw.get("expected_goal_involvements_per_90")),
            "xgc90": num(raw.get("expected_goals_conceded_per_90")),
            "ict": num(raw.get("ict_index")),
            "transfers_in": safe_int(raw.get("transfers_in_event")),
            "transfers_out": safe_int(raw.get("transfers_out_event")),
            "net_transfers": safe_int(raw.get("transfers_in_event")) - safe_int(raw.get("transfers_out_event")),
            "price_change": safe_int(raw.get("cost_change_event")),
            "penalties_order": safe_int(raw.get("penalties_order"), 0),
            "corners_order": safe_int(raw.get("corners_and_indirect_freekicks_order"), 0),
            "defcon90": num(raw.get("defensive_contribution_per_90")),
        }
        games = sorted(fixture_map.get(team_id, []), key=lambda x: (x["gw"], not x["home"]))
        p["fdr"] = mean([num(f["difficulty"], 3) for f in games], 3.0)
        p["next_gw_fixtures"] = sum(1 for f in games if f["gw"] == next_gw)
        p["fixtures"] = " | ".join(
            f"GW{f['gw']} {team_names.get(f['opponent'],'?')} {'H' if f['home'] else 'A'} [{f['difficulty']}]"
            for f in games[:HORIZON]
        ) or "No fixtures loaded"
        players.append(p)

    # Current Understat data is best-effort and never blocks the app.
    season_year = 2026
    players = attach_understat(players, season_year)

    return {
        "bootstrap": bootstrap,
        "fixtures_raw": fixtures_raw,
        "teams": teams,
        "team_names": team_names,
        "events": events,
        "current_gw": current_gw,
        "next_gw": next_gw,
        "fixture_map": dict(fixture_map),
        "players": players,
        "player_by_id": {p["id"]: p for p in players},
    }


# ============================================================
# PLAYER UNDERLYING MODEL
# ============================================================

def player_recent_history(player_id: int) -> list[dict]:
    try:
        data = api_get(f"{API}/element-summary/{player_id}/")
        return data.get("history", [])
    except Exception:
        return []


@st.cache_data(ttl=1800, show_spinner=False)
def recent_history_bulk(player_ids: tuple[int, ...]) -> dict[int, list[dict]]:
    out = {}
    for pid in player_ids:
        out[pid] = player_recent_history(pid)
    return out


def rolling_metric(history: list[dict], key: str, n: int) -> float:
    rows = history[-n:]
    vals = [num(r.get(key)) for r in rows]
    return mean(vals, 0.0)


def build_player_model(players: list[dict], history_map: dict[int, list[dict]], hist_db: dict) -> list[dict]:
    out = []
    for p0 in players:
        p = p0.copy()
        h = history_map.get(p["id"], [])
        recent = h[-8:]
        mins8 = sum(safe_int(r.get("minutes")) for r in recent)
        # FPL official xG/xA are cumulative; rolling expected rates are available from history snapshots where present.
        xg_recent = sum(num(r.get("expected_goals")) for r in recent)
        xa_recent = sum(num(r.get("expected_assists")) for r in recent)
        # If history contains cumulative expected values, derive gameweek deltas.
        cum_xg = [num(r.get("expected_goals")) for r in recent]
        cum_xa = [num(r.get("expected_assists")) for r in recent]
        dxg, dxa = [], []
        prevx, preva = None, None
        for x, a in zip(cum_xg, cum_xa):
            if prevx is not None:
                dxg.append(max(0.0, x - prevx))
                dxa.append(max(0.0, a - preva))
            prevx, preva = x, a
        p["rolling3_points"] = mean([safe_int(r.get("total_points")) for r in recent[-3:]], 0.0)
        p["rolling5_points"] = mean([safe_int(r.get("total_points")) for r in recent[-5:]], 0.0)
        p["rolling8_points"] = mean([safe_int(r.get("total_points")) for r in recent[-8:]], 0.0)
        p["rolling3_minutes"] = mean([safe_int(r.get("minutes")) for r in recent[-3:]], 0.0)
        p["rolling5_minutes"] = mean([safe_int(r.get("minutes")) for r in recent[-5:]], 0.0)
        p["rolling_xg"] = sum(dxg[-5:])
        p["rolling_xa"] = sum(dxa[-5:])
        p["rolling_npxg"] = p["rolling_xg"]
        p["recent_minutes"] = mins8
        p["minutes_probability"] = clamp((p["rolling5_minutes"] / 90) * 0.65 + status_factor(p["chance"], p["status"]) * 0.35, 0.05, 1.0)
        p["actual_xg_gap"] = p["goals"] - p.get("under_xg", p["xg"])
        p["actual_xa_gap"] = p["assists"] - p.get("under_xa", p["xa"])
        hist_prior = historical_player_prior(hist_db, p["name"])
        p.update(hist_prior)

        # Regression-to-mean: current underlying rates dominate, historical rate provides a stabiliser.
        current_rate = p["xgi90"]
        hist_rate = p["hist_xgi90"]
        hist_weight = 0.15 if p["minutes"] >= 1000 else 0.25
        p["regressed_xgi90"] = current_rate * (1 - hist_weight) + hist_rate * hist_weight if hist_rate else current_rate

        p["underlying_score"] = underlying_score(p)
        p["model_xp"] = projected_gameweek_points(p)
        p["value_score"] = p["model_xp"] / max(p["price"], 0.1)
        p["decision_score"] = decision_score(p)
        out.append(p)
    return out


def underlying_score(p: dict) -> float:
    xgi = p.get("regressed_xgi90", p.get("xgi90", 0.0))
    xa90 = p.get("xa90", 0.0)
    xg90 = p.get("xg90", 0.0)
    form = p.get("form", 0.0)
    minutes = p.get("minutes_probability", 0.5)
    npxg = p.get("npxg", p.get("xg", 0.0))
    penalty = 1.0 if p.get("penalties_order", 0) == 1 else 0.0
    finishing = clamp(p.get("actual_xg_gap", 0.0), -3, 3)
    # Positive finishing gap is deliberately shrunk, not rewarded fully.
    return (
        xgi * 18
        + xa90 * 7
        + xg90 * 6
        + form * 0.55
        + minutes * 4
        + penalty * 1.5
        + finishing * 0.25
        + math.log1p(max(npxg, 0)) * 0.8
    )


def projected_gameweek_points(p: dict, fdr: float | None = None, fixtures: int = 1) -> float:
    if fixtures <= 0:
        return 0.0
    base = (
        p.get("ep_next", 0.0) * 0.35
        + p.get("ppg", 0.0) * 0.10
        + p.get("regressed_xgi90", p.get("xgi90", 0.0)) * 3.0
        + p.get("xgc90", 0.0) * 0.0
        + p.get("minutes_probability", 0.5) * 2.0
    )
    if fdr is None:
        fdr = p.get("fdr", 3.0)
    fixture_mult = clamp(1 + (3.0 - fdr) * 0.10, 0.72, 1.30)
    dgw_mult = 1 + 0.78 * max(0, fixtures - 1)
    blank = 0 if fixtures else 1
    return max(0.0, base * fixture_mult * dgw_mult * (1 - blank))


def decision_score(p: dict) -> float:
    return (
        p.get("model_xp", 0.0) * 1.5
        + p.get("underlying_score", 0.0) * 0.55
        + p.get("value_score", 0.0) * 1.5
        + max(0, 3 - p.get("fdr", 3)) * 1.5
        + p.get("minutes_probability", 0.5) * 3
        + (2.5 if p.get("next_gw_fixtures", 1) >= 2 else 0)
    )


def player_status(p: dict) -> str:
    if p.get("status") != "a":
        return "🔴 Unavailable"
    if p.get("chance", 100) < 50:
        return "🔴 Major doubt"
    if p.get("chance", 100) < 75:
        return "🟠 Rotation risk"
    if p.get("next_gw_fixtures", 1) == 0:
        return "⚠️ Blank GW"
    if p.get("next_gw_fixtures", 1) >= 2:
        return "⚡ Double GW"
    if p.get("form", 0) >= 5:
        return "🟢 In form"
    return "🟡 Normal"


# ============================================================
# FIXTURE / FUTURE PROJECTION ENGINE
# ============================================================

def team_fixture_map_for_gw(fixture_map: dict, team_id: int, gw: int) -> list[dict]:
    return [f for f in fixture_map.get(team_id, []) if f["gw"] == gw]


def player_future_projection(p: dict, fixture_map: dict, start_gw: int, weeks: int = HORIZON) -> dict[int, float]:
    result = {}
    for gw in range(start_gw, start_gw + weeks):
        fx = team_fixture_map_for_gw(fixture_map, p["team_id"], gw)
        if not fx:
            result[gw] = 0.0
        else:
            result[gw] = sum(projected_gameweek_points(p, num(f["difficulty"], 3), len(fx)) / max(1, len(fx)) for f in fx)
    return result


def squad_week_projection(squad: list[dict], fixture_map: dict, gw: int, include_bench: bool = False) -> float:
    if not squad:
        return 0.0
    xi = optimise_xi(squad, fixture_map, gw)
    if not xi:
        return 0.0
    selected = xi["lineup"] + (xi["bench"] if include_bench else [])
    return sum(player_fixture_xp(p, fixture_map, gw) for p in selected)


def player_fixture_xp(p: dict, fixture_map: dict, gw: int) -> float:
    fx = team_fixture_map_for_gw(fixture_map, p["team_id"], gw)
    if not fx:
        return 0.0
    # Do not double count the same player across fixtures: each fixture gets its own opportunity.
    base = 0.0
    for f in fx:
        base += projected_gameweek_points(p, num(f["difficulty"], 3), 1)
    # Two fixtures give more opportunities but a small rotation discount.
    if len(fx) >= 2:
        base *= 0.94 + 0.03 * min(2, p.get("minutes_probability", 0.7))
    return base


def optimise_xi(squad: list[dict], fixture_map: dict, gw: int) -> dict | None:
    if len(squad) < 11:
        return None
    by = defaultdict(list)
    for p in squad:
        by[p["position"]].append(p)
    for pos in by:
        by[pos].sort(key=lambda p: player_fixture_xp(p, fixture_map, gw), reverse=True)
    gks, defs, mids, fwds = by["GK"], by["DEF"], by["MID"], by["FWD"]
    if not all([gks, defs, mids, fwds]):
        return None
    best, best_val = None, -1e9
    for d, m, f in FORMATIONS:
        if len(defs) < d or len(mids) < m or len(fwds) < f:
            continue
        # Start with best positional candidates, then validate club limit.
        lineup = [gks[0]] + defs[:d] + mids[:m] + fwds[:f]
        if len({p["id"] for p in lineup}) != 11:
            continue
        if any(sum(p["team_id"] == t for p in lineup) > MAX_PER_CLUB for t in {p["team_id"] for p in lineup}):
            # Try a simple replacement pass.
            lineup = repair_club_limit(lineup, squad, fixture_map, gw)
        val = sum(player_fixture_xp(p, fixture_map, gw) for p in lineup)
        if val > best_val and len(lineup) == 11:
            ids = {p["id"] for p in lineup}
            best = {"formation": f"{d}-{m}-{f}", "lineup": lineup, "bench": [p for p in squad if p["id"] not in ids], "score": val}
            best_val = val
    return best


def repair_club_limit(lineup: list[dict], squad: list[dict], fixture_map: dict, gw: int) -> list[dict]:
    out = list(lineup)
    for _ in range(10):
        counts = defaultdict(int)
        for p in out:
            counts[p["team_id"]] += 1
        bad = next((t for t, c in counts.items() if c > 3), None)
        if bad is None:
            return out
        candidates = [p for p in out if p["team_id"] == bad]
        replace = min(candidates, key=lambda p: player_fixture_xp(p, fixture_map, gw))
        pos = replace["position"]
        alternatives = [p for p in squad if p["position"] == pos and p["id"] not in {x["id"] for x in out} and counts[p["team_id"]] < 3]
        if not alternatives:
            return out
        incoming = max(alternatives, key=lambda p: player_fixture_xp(p, fixture_map, gw))
        out[out.index(replace)] = incoming
    return out


# ============================================================
# SQUAD / TRANSFERS
# ============================================================

def load_manager_team(entry_id: int, current_gw: int, player_by_id: dict[int, dict]) -> tuple[dict, list[dict]]:
    data = get_entry_picks(entry_id, current_gw)
    squad = []
    for pick in data.get("picks", []):
        p = player_by_id.get(safe_int(pick.get("element")))
        if not p:
            continue
        q = p.copy()
        q.update({
            "is_captain": bool(pick.get("is_captain")),
            "is_vice": bool(pick.get("is_vice_captain")),
            "multiplier": safe_int(pick.get("multiplier"), 1),
            "position_slot": safe_int(pick.get("position"), 0),
            "purchase_price": num(pick.get("purchase_price")) / 10,
        })
        squad.append(q)
    return data, squad


def squad_counts(squad: list[dict]) -> dict[int, int]:
    c = defaultdict(int)
    for p in squad:
        c[p["team_id"]] += 1
    return dict(c)


def legal_replacement(outgoing: dict, incoming: dict, squad: list[dict], bank: float) -> bool:
    if incoming["position"] != outgoing["position"]:
        return False
    if incoming["price"] > bank + outgoing["price"] + 1e-9:
        return False
    ids = {p["id"] for p in squad}
    if incoming["id"] in ids:
        return False
    counts = squad_counts(squad)
    counts[outgoing["team_id"]] -= 1
    counts[incoming["team_id"]] += 1
    return counts[incoming["team_id"]] <= MAX_PER_CLUB


def transfer_suggestions(squad: list[dict], bank: float, free_transfers: int, elite_counts: dict[int, int]) -> list[dict]:
    owned = {p["id"] for p in squad}
    suggestions = []
    for out in squad:
        candidates = [
            p for p in PLAYERS
            if p["id"] not in owned
            and p["position"] == out["position"]
            and p["status"] == "a"
            and p["chance"] >= 50
            and legal_replacement(out, p, squad, bank)
        ]
        candidates.sort(key=lambda p: p["decision_score"], reverse=True)
        for inc in candidates[:30]:
            gain = inc["multi_projection"] - out["multi_projection"]
            elite = elite_counts.get(inc["id"], 0) - elite_counts.get(out["id"], 0)
            hit = 0 if free_transfers > 0 else TRANSFER_HIT
            net = gain - hit + elite * 0.30
            suggestions.append({"out": out, "in": inc, "gain": gain, "hit": hit, "net": net, "elite": elite})
    return sorted(suggestions, key=lambda x: x["net"], reverse=True)[:15]


def transfer_decision(squad: list[dict], bank: float, free_transfers: int, elite_counts: dict[int, int]) -> dict:
    opts = transfer_suggestions(squad, bank, free_transfers, elite_counts)
    if not opts:
        return {"decision": "ROLL", "reason": "No legal transfer meaningfully improves the squad.", "options": []}
    best = opts[0]
    if free_transfers > 0 and best["gain"] >= 3.0:
        return {"decision": "TRANSFER", "reason": f"{best['in']['name']} for {best['out']['name']} adds about {best['gain']:.1f} projected points over the planning window.", "options": opts}
    if free_transfers == 0 and best["net"] >= 1.5:
        return {"decision": "TAKE HIT", "reason": f"The projected gain is {best['gain']:.1f}; after -4 the model still sees a positive net edge of {best['net']:.1f}.", "options": opts}
    return {"decision": "ROLL", "reason": "The best transfer does not clear the model's value threshold.", "options": opts}


def captain_candidates(squad: list[dict], fixture_map: dict, gw: int, elite_caps: dict[str, int]) -> list[dict]:
    candidates = [p for p in squad if p["status"] == "a" and p["chance"] >= 75 and team_fixture_map_for_gw(fixture_map, p["team_id"], gw)]
    for p in candidates:
        xp = player_fixture_xp(p, fixture_map, gw)
        p["captain_score"] = xp + p.get("underlying_score", 0) * 0.15 + elite_caps.get(p["name"], 0) * 1.2
    return sorted(candidates, key=lambda p: p["captain_score"], reverse=True)[:5]


# ============================================================
# ELITE MANAGERS
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def load_elite(name: str, entry_id: int, gw: int, player_by_id: dict[int, dict]) -> dict:
    try:
        info = get_entry_info(entry_id)
        picks = get_entry_picks(entry_id, gw)
        squad = []
        captain = None
        for pick in picks.get("picks", []):
            p = player_by_id.get(safe_int(pick.get("element")))
            if not p:
                continue
            squad.append(p)
            if pick.get("is_captain"):
                captain = p["name"]
        return {"name": name, "entry_id": entry_id, "team": info.get("name", ""), "squad": squad, "captain": captain, "status": "OK"}
    except Exception as e:
        return {"name": name, "entry_id": entry_id, "team": "", "squad": [], "captain": None, "status": f"Failed: {e}"}


def elite_counts(rows: list[dict]) -> tuple[dict[int, int], dict[str, int]]:
    pc, cc = defaultdict(int), defaultdict(int)
    for r in rows:
        for p in r.get("squad", []):
            pc[p["id"]] += 1
        if r.get("captain"):
            cc[r["captain"]] += 1
    return dict(pc), dict(cc)


# ============================================================
# CHIP ENGINE
# ============================================================

@dataclass
class ChipOption:
    chip: str
    gw: int
    score: float
    gain: float
    rationale: str
    secondary: float = 0.0


def remaining_chips(entry_id: int, current_gw: int) -> dict[str, bool]:
    try:
        h = get_entry_history(entry_id)
        chips = h.get("chips", [])
        used = {str(c.get("name", "")).lower() for c in chips if safe_int(c.get("event")) <= current_gw}
    except Exception:
        used = set()
    half = 1 if current_gw <= 19 else 2
    # API chip names are normally wildcard/freehit/benchboost/triplecaptain; use half based on actual used event.
    out = {}
    for chip in ("wildcard", "freehit", "benchboost", "triplecaptain"):
        used_half = any(str(c.get("name", "")).lower() == chip and (safe_int(c.get("event")) <= 19) == (half == 1) for c in get_entry_history(entry_id).get("chips", [])) if entry_id else False
        out[chip] = not used_half
    return out


def fixture_windows(fixture_map: dict, teams: dict[int, dict], start_gw: int, end_gw: int) -> pd.DataFrame:
    rows = []
    for gw in range(start_gw, end_gw + 1):
        counts = defaultdict(int)
        for team_id, fs in fixture_map.items():
            counts[team_id] = sum(1 for f in fs if f["gw"] == gw)
        doubles = sum(v >= 2 for v in counts.values())
        blanks = max(0, len(teams) - sum(v > 0 for v in counts.values()))
        rows.append({"GW": gw, "Double teams": doubles, "Blank teams": blanks})
    return pd.DataFrame(rows)


def squad_expected_gw(squad: list[dict], fixture_map: dict, gw: int, bench_boost: bool = False) -> float:
    if not squad:
        return 0.0
    xi = optimise_xi(squad, fixture_map, gw)
    if not xi:
        return 0.0
    lineup = xi["lineup"]
    total = sum(player_fixture_xp(p, fixture_map, gw) for p in lineup)
    if bench_boost:
        total += sum(player_fixture_xp(p, fixture_map, gw) for p in xi["bench"])
    return total


def optimise_wildcard_squad(players: list[dict], budget: float, target_gws: list[int], fixture_map: dict) -> list[dict]:
    """Fast heuristic full-squad optimiser. It favours projected points, minutes and fixture continuity."""
    pool = [p for p in players if p["status"] == "a" and p["chance"] >= 50 and p["price"] <= budget]
    # position quotas 2/5/5/3
    targets = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    selected = []
    remaining_budget = budget
    counts = defaultdict(int)
    club_counts = defaultdict(int)
    for pos, n in targets.items():
        pos_pool = [p for p in pool if p["position"] == pos]
        scored = []
        for p in pos_pool:
            future = sum(player_fixture_xp(p, fixture_map, gw) for gw in target_gws)
            value = future + p["minutes_probability"] * 3 + p["underlying_score"] * 0.18 + p.get("ownership", 0) * 0.01
            scored.append((value, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        for _, p in scored:
            if len([x for x in selected if x["position"] == pos]) >= n:
                break
            if club_counts[p["team_id"]] >= 3:
                continue
            if p["price"] > remaining_budget - sum(x["price"] for x in selected) + 1e-9:
                continue
            selected.append(p)
            club_counts[p["team_id"]] += 1
    # If budget/club constraints prevented 15, fill cheapest valid players.
    for pos, n in targets.items():
        while len([x for x in selected if x["position"] == pos]) < n:
            candidates = [p for p in pool if p["position"] == pos and p["id"] not in {x["id"] for x in selected} and club_counts[p["team_id"]] < 3]
            candidates.sort(key=lambda p: (p["price"], -p["minutes_probability"]))
            if not candidates:
                break
            p = candidates[0]
            if sum(x["price"] for x in selected) + p["price"] > budget:
                # replace the weakest selected same position with a cheaper player
                same = [x for x in selected if x["position"] == pos and x["price"] > p["price"]]
                if not same:
                    break
                weak = min(same, key=lambda x: x["decision_score"])
                selected.remove(weak)
                club_counts[weak["team_id"]] -= 1
            selected.append(p)
            club_counts[p["team_id"]] += 1
    return selected[:15]


def chip_score_table(squad: list[dict], fixture_map: dict, teams: dict[int, dict], start_gw: int, end_gw: int, budget: float) -> pd.DataFrame:
    rows = []
    base_by_gw = {gw: squad_expected_gw(squad, fixture_map, gw, False) for gw in range(start_gw, end_gw + 1)}
    wildcard_cache: dict[int, list[dict]] = {}
    for gw in range(start_gw, end_gw + 1):
        doubles = sum(len(team_fixture_map_for_gw(fixture_map, tid, gw)) >= 2 for tid in teams)
        blanks = sum(len(team_fixture_map_for_gw(fixture_map, tid, gw)) == 0 for tid in teams)
        # Bench boost: current squad's marginal bench value.
        bb = squad_expected_gw(squad, fixture_map, gw, True) - base_by_gw[gw]
        # Free hit: optimal one-week squad versus current squad.
        fh_pool = [p for p in PLAYERS if p["status"] == "a" and p["chance"] >= 50]
        # Build a temporary squad cheaply: best 15 affordable around current total budget.
        fh_squad = optimise_wildcard_squad(fh_pool, budget, [gw], fixture_map)
        fh = squad_expected_gw(fh_squad, fixture_map, gw, False) - base_by_gw[gw]
        # Triple captain: best captain incremental multiplier over normal captain.
        caps = [p for p in squad if team_fixture_map_for_gw(fixture_map, p["team_id"], gw)]
        tc = 0.0
        if caps:
            vals = sorted(((player_fixture_xp(p, fixture_map, gw), p) for p in caps), reverse=True, key=lambda x: x[0])
            tc = vals[0][0]
        # Wildcard: compare 4-GW squad output before and after a hypothetical wildcard.
        target = list(range(gw, min(end_gw, gw + 3) + 1))
        if gw not in wildcard_cache:
            wildcard_cache[gw] = optimise_wildcard_squad(PLAYERS, budget, target, fixture_map)
        wc_squad = wildcard_cache[gw]
        wc = sum(squad_expected_gw(wc_squad, fixture_map, w, False) for w in target) - sum(squad_expected_gw(squad, fixture_map, w, False) for w in target)
        # Strategy score weights immediate chip value plus structural value.
        rows.append({
            "GW": gw,
            "BB EV": round(bb, 1),
            "FH EV": round(fh, 1),
            "TC EV": round(tc, 1),
            "WC 4GW EV": round(wc, 1),
            "DGW teams": doubles,
            "Blank teams": blanks,
        })
    return pd.DataFrame(rows)


def optimise_chip_sequence(squad: list[dict], fixture_map: dict, teams: dict[int, dict], start_gw: int, end_gw: int, budget: float, chips: dict[str, bool]) -> list[dict]:
    """Evaluate a manageable set of chip sequences. This is intentionally transparent rather than a black-box solver."""
    score_df = chip_score_table(squad, fixture_map, teams, start_gw, end_gw, budget)
    candidates = []
    for chip, col in [("Bench Boost", "BB EV"), ("Free Hit", "FH EV"), ("Triple Captain", "TC EV"), ("Wildcard", "WC 4GW EV")]:
        key = chip.lower().replace(" ", "")
        if not chips.get(key, True):
            continue
        if col not in score_df:
            continue
        top = score_df.nlargest(4, col)
        for _, r in top.iterrows():
            val = num(r[col])
            if val <= 0:
                continue
            reason = []
            if chip == "Bench Boost" and r["DGW teams"] >= 4:
                reason.append(f"{int(r['DGW teams'])} teams double")
            if chip == "Free Hit" and r["Blank teams"] >= 4:
                reason.append(f"{int(r['Blank teams'])} teams blank")
            if chip == "Triple Captain":
                reason.append("highest captain multiplier opportunity")
            if chip == "Wildcard":
                reason.append("strong multi-GW squad reset value")
            candidates.append({"chip": chip, "gw": int(r["GW"]), "score": val, "rationale": "; ".join(reason) or "best current modelled opportunity"})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    # Add interaction bonus for WC immediately before BB and preserve one-chip-per-GW rule.
    for a in candidates:
        for b in candidates:
            if a is b or a["chip"] != "Wildcard" or b["chip"] != "Bench Boost":
                continue
            if b["gw"] == a["gw"] + 1:
                a["score"] += min(8.0, b["score"] * 0.30)
                a["rationale"] += f"; pairs with BB in GW{b['gw']}"
    selected = []
    used_gws = set()
    used_chips = set()
    for c in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if c["gw"] in used_gws or c["chip"] in used_chips:
            continue
        selected.append(c)
        used_gws.add(c["gw"])
        used_chips.add(c["chip"])
        if len(selected) >= 4:
            break
    return sorted(selected, key=lambda x: x["gw"])


# ============================================================
# MINI-LEAGUE / RIVALS
# ============================================================

def league_analysis(league_id: int, entry_id: int) -> dict:
    data = get_league(league_id)
    results = data.get("standings", {}).get("results", [])
    me = next((r for r in results if safe_int(r.get("entry")) == entry_id), None)
    if not me:
        return {"results": results, "me": None, "above": None, "below": None}
    rank = safe_int(me.get("rank"))
    above = next((r for r in results if safe_int(r.get("rank")) == rank - 1), None)
    below = next((r for r in results if safe_int(r.get("rank")) == rank + 1), None)
    return {"results": results, "me": me, "above": above, "below": below}


# ============================================================
# YOUTUBE / GEMINI
# ============================================================

def extract_video_id(url: str) -> str | None:
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url.strip()):
        return url.strip()
    parsed = urlparse(url.strip())
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        return parsed.path.strip("/")[:11]
    if parsed.hostname and "youtube.com" in parsed.hostname:
        q = parse_qs(parsed.query).get("v")
        if q:
            return q[0][:11]
        m = re.search(r"/(?:shorts|live)/([A-Za-z0-9_-]{11})", parsed.path)
        return m.group(1) if m else None
    return None


def fetch_transcript(url: str) -> tuple[str | None, str | None]:
    if YouTubeTranscriptApi is None:
        return None, "youtube-transcript-api is not installed."
    vid = extract_video_id(url)
    if not vid:
        return None, "Could not identify the YouTube video ID."
    try:
        api = YouTubeTranscriptApi()
        transcript = api.fetch(vid)
        text = " ".join(getattr(x, "text", str(x)) for x in transcript)
        return text, None
    except Exception as e:
        return None, f"Transcript unavailable: {e}"


def gemini_generate(prompt: str, system: str) -> tuple[str, str]:
    key = get_secret("GEMINI_API_KEY")
    if not key or genai is None:
        raise RuntimeError("GEMINI_API_KEY is missing or Gemini package is unavailable.")
    client = genai.Client(api_key=key)
    last = None
    for model in GEMINI_MODELS:
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(system_instruction=system, temperature=0.2),
            )
            return response.text or "No response returned.", model
        except Exception as e:
            last = e
    raise RuntimeError(last)


# ============================================================
# LOAD / BUILD
# ============================================================

try:
    DATA = load_current_data()
except Exception as exc:
    st.error("⚠️ The official FPL API could not be loaded.")
    st.code(str(exc))
    st.stop()

CURRENT_GW = DATA["current_gw"]
NEXT_GW = DATA["next_gw"]
FIXTURE_MAP = DATA["fixture_map"]
TEAMS = DATA["teams"]
TEAM_NAMES = DATA["team_names"]
BASE_PLAYERS = DATA["players"]
PLAYER_BY_ID = DATA["player_by_id"]

with st.spinner("Building the player model..."):
    try:
        # Avoid hundreds of element-summary API calls. Use the strongest current candidates for rolling form.
        seed_ids = {p["id"] for p in BASE_PLAYERS if p["minutes"] >= 300}
        ranked_seed = sorted(BASE_PLAYERS, key=lambda p: (p["xgi90"] * 2 + p["form"] + p["ep_next"]), reverse=True)[:120]
        seed_ids.update(p["id"] for p in ranked_seed)
        HISTORY = recent_history_bulk(tuple(sorted(seed_ids)))
    except Exception:
        HISTORY = {p["id"]: [] for p in BASE_PLAYERS}
    HIST_DB = load_historical_database()
    PLAYERS = build_player_model(BASE_PLAYERS, HISTORY, HIST_DB)
    PLAYER_BY_ID = {p["id"]: p for p in PLAYERS}

# Add multi-GW projections after the model exists.
for p in PLAYERS:
    future = player_future_projection(p, FIXTURE_MAP, NEXT_GW, PROJECTION_WEEKS)
    p["multi_projection"] = sum(future.values())
    p["future_projection"] = future


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

st.sidebar.title("⚙️ Manager Settings")
team_id = safe_int(st.sidebar.number_input("Your FPL Team ID", value=DEFAULT_TEAM_ID, min_value=1, step=1))
league_options = dict(DEFAULT_LEAGUES)
league_id = st.sidebar.selectbox("Mini-League", list(league_options.keys()))

st.sidebar.divider()
st.sidebar.caption(f"Current GW: {CURRENT_GW}  |  Planning GW: {NEXT_GW}")
st.sidebar.caption("Historical archive: 2022/23–2025/26")
if st.sidebar.button("🔄 Refresh live data"):
    st.cache_data.clear()
    st.rerun()

# Load user's team.
try:
    TEAM_DATA, MY_SQUAD = load_manager_team(team_id, CURRENT_GW, PLAYER_BY_ID)
except Exception as exc:
    st.error(f"Could not load Team ID {team_id}: {exc}")
    st.stop()

entry_history = get_entry_history(team_id)
current_history_rows = entry_history.get("current", [])
latest_history = current_history_rows[-1] if current_history_rows else {}
BANK = num(latest_history.get("bank")) / 10

# The public FPL API does not expose the exact current banked free-transfer count for
# every manager. We therefore make it an explicit setting instead of pretending that
# event_transfers is the same thing.
FREE_TRANSFERS = safe_int(st.sidebar.number_input("Free transfers available", min_value=1, max_value=5, value=1, step=1))
CHIPS = remaining_chips(team_id, CURRENT_GW)

# Elite data.
elite_rows = []
for name, eid in DEFAULT_ELITES.items():
    row = load_elite(name, eid, CURRENT_GW, PLAYER_BY_ID)
    if row.get("status") == "OK":
        elite_rows.append(row)
ELITE_PLAYER_COUNTS, ELITE_CAP_COUNTS = elite_counts(elite_rows)

TRANSFER = transfer_decision(MY_SQUAD, BANK, FREE_TRANSFERS, ELITE_PLAYER_COUNTS)
CAPTAINS = captain_candidates(MY_SQUAD, FIXTURE_MAP, NEXT_GW, ELITE_CAP_COUNTS)


# ============================================================
# HEADER
# ============================================================

st.title("⚽ FPL Assistant Manager 2.0")
st.caption("Decision engine • underlying player potential • historical evidence • squad optimisation • chip strategy")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Current GW", CURRENT_GW)
c2.metric("Planning GW", NEXT_GW)
c3.metric("Bank", f"£{BANK:.1f}m")
c4.metric("Free Transfers", FREE_TRANSFERS)
c5.metric("Players", len(MY_SQUAD))

st.info("This is a decision-support model, not a guarantee. Current-season data is weighted above historical data, and historical chip patterns are treated as evidence rather than rules.")

TABS = st.tabs([
    "🏠 Dashboard", "👥 Squad", "🔁 Transfers", "📈 Players", "🎯 Captain", "🧠 Chip Optimiser",
    "📅 Fixtures", "📚 History", "🏆 Elite Managers", "⚔️ Mini-Leagues", "📺 Creator AI", "💬 AI Assistant", "ℹ️ User Guide"
])


# ============================================================
# DASHBOARD
# ============================================================

with TABS[0]:
    st.header("🧠 Your decision this week")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transfer", TRANSFER["decision"])
    col2.metric("Captain", CAPTAINS[0]["name"] if CAPTAINS else "—")
    col3.metric("Best chip signal", "See optimiser")
    st.write(TRANSFER["reason"])

    if CAPTAINS:
        st.subheader("🎯 Captain shortlist")
        st.dataframe(pd.DataFrame([
            {"Player": p["name"], "Club": p["team"], "GW xP": round(player_fixture_xp(p, FIXTURE_MAP, NEXT_GW), 1), "Underlying": round(p["underlying_score"], 1), "Fixtures": p["next_gw_fixtures"]}
            for p in CAPTAINS
        ]), use_container_width=True, hide_index=True)

    st.subheader("🚨 Squad warnings")
    warnings = []
    for p in MY_SQUAD:
        if p["status"] != "a" or p["chance"] < 75:
            warnings.append(f"{p['name']} — availability {p['chance']:.0f}% — {p['news'] or 'check team news'}")
        if p["next_gw_fixtures"] == 0:
            warnings.append(f"{p['name']} — blank in GW{NEXT_GW}")
    if warnings:
        for w in warnings[:12]:
            st.warning(w)
    else:
        st.success("No major availability or blank-GW problems detected in the current squad.")

    st.subheader("🔥 Highest underlying potential")
    top = sorted(PLAYERS, key=lambda p: p["underlying_score"], reverse=True)[:15]
    st.dataframe(pd.DataFrame([
        {"Player": p["name"], "Club": p["team"], "Pos": p["position"], "£m": p["price"], "xGI/90": round(p["xgi90"], 2), "xA/90": round(p["xa90"], 2), "npxG source": p.get("xg_source", "FPL"), "Model": round(p["decision_score"], 1)}
        for p in top
    ]), use_container_width=True, hide_index=True)


# ============================================================
# SQUAD
# ============================================================

with TABS[1]:
    st.header("👥 Your Squad")
    rows = []
    for p in MY_SQUAD:
        rows.append({
            "Player": p["name"], "Club": p["team"], "Pos": p["position"], "Price": f"£{p['price']:.1f}m",
            "xGI/90": round(p["xgi90"], 2), "npxG": round(p.get("npxg", p.get("xg", 0)), 2), "xA": round(p.get("under_xa", p.get("xa", 0)), 2),
            "GW xP": round(player_fixture_xp(p, FIXTURE_MAP, NEXT_GW), 1), "5GW xP": round(p["multi_projection"], 1), "Status": player_status(p),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    xi = optimise_xi(MY_SQUAD, FIXTURE_MAP, NEXT_GW)
    if xi:
        st.subheader(f"Best XI for GW{NEXT_GW} — {xi['formation']}")
        st.dataframe(pd.DataFrame([{"Player": p["name"], "Club": p["team"], "Pos": p["position"], "GW xP": round(player_fixture_xp(p, FIXTURE_MAP, NEXT_GW), 1)} for p in xi["lineup"]]), use_container_width=True, hide_index=True)
        st.subheader("Bench")
        st.dataframe(pd.DataFrame([{"Player": p["name"], "Club": p["team"], "Pos": p["position"], "GW xP": round(player_fixture_xp(p, FIXTURE_MAP, NEXT_GW), 1)} for p in xi["bench"]]), use_container_width=True, hide_index=True)


# ============================================================
# TRANSFERS
# ============================================================

with TABS[2]:
    st.header("🔁 Transfer Engine")
    st.metric("Recommendation", TRANSFER["decision"])
    st.write(TRANSFER["reason"])
    if TRANSFER["options"]:
        st.dataframe(pd.DataFrame([
            {"OUT": o["out"]["name"], "IN": o["in"]["name"], "Price": f"£{o['in']['price']:.1f}m", "5GW gain": round(o["gain"], 1), "Hit": o["hit"], "Net": round(o["net"], 1), "Elite": o["elite"]}
            for o in TRANSFER["options"]
        ]), use_container_width=True, hide_index=True)
    st.caption("Transfers are evaluated against your actual squad, budget and three-player-per-club rule. A hit is only recommended when the projected net gain clears a threshold.")


# ============================================================
# PLAYERS
# ============================================================

with TABS[3]:
    st.header("📈 Player Intelligence")
    st.caption("The model prioritises underlying potential and minutes, then blends current FPL output, fixtures and historical priors.")
    pos = st.multiselect("Positions", ["GK", "DEF", "MID", "FWD"], default=["GK", "DEF", "MID", "FWD"])
    max_price = st.slider("Maximum price", 4.0, 16.0, 16.0, 0.1)
    min_minutes = st.slider("Minimum season minutes", 0, 2500, 0, 100)
    df_players = pd.DataFrame([
        {"Player": p["name"], "Club": p["team"], "Pos": p["position"], "£m": p["price"], "xG": p["xg"], "npxG": p.get("npxg", p["xg"]), "xA": p.get("under_xa", p["xa"]), "xG/90": p["xg90"], "xA/90": p["xa90"], "xGI/90": p["xgi90"], "3GW form": p["rolling3_points"], "5GW xP": p["multi_projection"], "Minutes %": round(p["minutes_probability"] * 100), "Model": p["decision_score"], "Status": player_status(p)}
        for p in PLAYERS if p["position"] in pos and p["price"] <= max_price and p["minutes"] >= min_minutes
    ]).sort_values("Model", ascending=False)
    st.dataframe(df_players.head(100), use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download player model CSV", df_players.to_csv(index=False), "fpl_player_model.csv", "text/csv")


# ============================================================
# CAPTAIN
# ============================================================

with TABS[4]:
    st.header(f"🎯 Captaincy — GW{NEXT_GW}")
    if CAPTAINS:
        st.dataframe(pd.DataFrame([
            {"Rank": i + 1, "Player": p["name"], "Club": p["team"], "GW xP": round(player_fixture_xp(p, FIXTURE_MAP, NEXT_GW), 1), "Fixtures": p["next_gw_fixtures"], "xGI/90": round(p["xgi90"], 2), "Minutes %": round(p["minutes_probability"] * 100), "Elite captain votes": ELITE_CAP_COUNTS.get(p["name"], 0)}
            for i, p in enumerate(CAPTAINS)
        ]), use_container_width=True, hide_index=True)
        st.success(f"Model captain: **{CAPTAINS[0]['name']}**. This is based on expected output, underlying potential, minutes and fixture context — not simply last week's points.")
    else:
        st.warning("No captain candidate meets the availability threshold.")


# ============================================================
# CHIP OPTIMISER
# ============================================================

with TABS[5]:
    st.header("🧠 Chip Strategy Optimiser")
    st.write("This is the major new engine. It evaluates chip value against your actual squad, future fixtures and alternative Gameweeks. It also looks for chip interactions such as **Wildcard → Bench Boost**.")
    chip_cols = st.columns(4)
    for i, chip in enumerate(["wildcard", "benchboost", "freehit", "triplecaptain"]):
        chip_cols[i].metric(chip.replace("benchboost", "Bench Boost").replace("freehit", "Free Hit").replace("triplecaptain", "Triple Captain").title(), "AVAILABLE" if CHIPS.get(chip, True) else "USED")

    end = min(NEXT_GW + 7, 38)
    score_df = chip_score_table(MY_SQUAD, FIXTURE_MAP, TEAMS, NEXT_GW, end, sum(p["price"] for p in MY_SQUAD) + BANK)
    st.subheader("Chip opportunity calendar")
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    sequence = optimise_chip_sequence(MY_SQUAD, FIXTURE_MAP, TEAMS, NEXT_GW, end, sum(p["price"] for p in MY_SQUAD) + BANK, CHIPS)
    st.subheader("🏆 Current best chip plan")
    if sequence:
        st.dataframe(pd.DataFrame([
            {"GW": x["gw"], "Chip": x["chip"], "Model EV": round(x["score"], 1), "Why": x["rationale"]}
            for x in sequence
        ]), use_container_width=True, hide_index=True)
        for x in sequence:
            st.info(f"**GW{x['gw']} — {x['chip']}**: {x['rationale']}. Model value: **{x['score']:.1f}**.")
    else:
        st.success("No chip currently has enough projected value to force a decision. Holding is a valid strategy.")

    st.subheader("Why this is different from a basic DGW rule")
    st.markdown("""
    - **Bench Boost** looks at the marginal value of your actual bench, not just the number of double-gameweek players.
    - **Free Hit** compares a one-week optimised squad against the team you can already field.
    - **Triple Captain** measures the captaincy multiplier opportunity.
    - **Wildcard** looks beyond one Gameweek and asks whether a new squad improves the next several Gameweeks.
    - **Sequence bonus:** a Wildcard immediately before a strong Bench Boost opportunity can receive an interaction premium.
    - **Only one chip per Gameweek** is enforced.
    """)


# ============================================================
# FIXTURES
# ============================================================

with TABS[6]:
    st.header("📅 Fixture Intelligence")
    rows = []
    for gw in range(NEXT_GW, min(38, NEXT_GW + HORIZON - 1) + 1):
        counts = defaultdict(int)
        for tid in TEAMS:
            counts[tid] = len(team_fixture_map_for_gw(FIXTURE_MAP, tid, gw))
        rows.append({"GW": gw, "Double teams": sum(v >= 2 for v in counts.values()), "Blank teams": sum(v == 0 for v in counts.values())})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Team fixture runs")
    team_rows = []
    for tid, team in TEAMS.items():
        fs = sorted(FIXTURE_MAP.get(tid, []), key=lambda x: x["gw"])
        team_rows.append({"Club": team.get("short_name", "?"), "Next": " | ".join(f"GW{f['gw']} {'H' if f['home'] else 'A'} {TEAM_NAMES.get(f['opponent'],'?')}" for f in fs[:6]), "Avg FDR": round(mean([num(f['difficulty'], 3) for f in fs[:5]], 3), 2)})
    st.dataframe(pd.DataFrame(team_rows).sort_values("Avg FDR"), use_container_width=True, hide_index=True)


# ============================================================
# HISTORY
# ============================================================

with TABS[7]:
    st.header("📚 Historical FPL Evidence")
    st.write("The app keeps the historical archive as a cached reference layer. It is used to stabilise player priors and to understand how blank/double-gameweek structures behaved historically. It does **not** override current-season information.")
    st.dataframe(pd.DataFrame(historical_summary(HIST_DB)), use_container_width=True, hide_index=True)

    hist_windows = historical_dgw_bgw_windows(HIST_DB)
    if not hist_windows.empty:
        st.subheader("Historical blank/double windows")
        st.dataframe(hist_windows.sort_values(["Season", "GW"]), use_container_width=True, hide_index=True)

    st.subheader("Historical player priors")
    names = st.multiselect("Check players", [p["name"] for p in sorted(PLAYERS, key=lambda x: x["name"])][:250])
    if names:
        st.dataframe(pd.DataFrame([{**{"Player": n}, **historical_player_prior(HIST_DB, n)} for n in names]), use_container_width=True, hide_index=True)


# ============================================================
# ELITE MANAGERS
# ============================================================

with TABS[8]:
    st.header("🏆 Elite Manager Consensus")
    if elite_rows:
        rows = []
        for r in elite_rows:
            rows.append({"Manager": r["name"], "Team": r["team"], "Captain": r["captain"] or "—", "Players loaded": len(r["squad"])})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        pc = sorted(ELITE_PLAYER_COUNTS.items(), key=lambda x: x[1], reverse=True)
        st.subheader("Most-owned across tracked managers")
        st.dataframe(pd.DataFrame([{ "Player": PLAYER_BY_ID.get(pid, {}).get("name", str(pid)), "Managers": n } for pid, n in pc[:25]]), use_container_width=True, hide_index=True)
    else:
        st.warning("Elite manager data could not be loaded.")


# ============================================================
# MINI-LEAGUES
# ============================================================

with TABS[9]:
    st.header("⚔️ Mini-League & Rivals")
    try:
        la = league_analysis(league_options[league_id], team_id)
        me = la["me"]
        if me:
            a, b, c, d = st.columns(4)
            a.metric("Rank", f"#{safe_int(me.get('rank'))}")
            b.metric("Total", safe_int(me.get("total")))
            c.metric("GW", safe_int(me.get("event_total")))
            d.metric("Managers", len(la["results"]))
            if la["above"]:
                st.info(f"Manager above: **{la['above'].get('player_name','—')}** — {safe_int(la['above'].get('total')) - safe_int(me.get('total'))} points ahead.")
            if la["below"]:
                st.success(f"Manager below: **{la['below'].get('player_name','—')}** — {safe_int(me.get('total')) - safe_int(la['below'].get('total'))} points behind you.")
        st.dataframe(pd.DataFrame([{ "Rank": r.get("rank"), "Manager": r.get("player_name"), "Team": r.get("entry_name"), "Total": r.get("total"), "GW": r.get("event_total") } for r in la["results"][:30]]), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(str(e))


# ============================================================
# CREATOR AI
# ============================================================

with TABS[10]:
    st.header("📺 Creator Intelligence")
    st.caption("Paste a YouTube FPL video and compare its recommendations with your squad, the model and tracked elite managers.")
    st.markdown(" | ".join(f"[{n}]({u})" for n, u in CREATOR_CHANNELS.items()))
    video_url = st.text_input("YouTube video URL or ID")
    if st.button("🧠 Analyse video", type="primary"):
        transcript, error = fetch_transcript(video_url)
        if error:
            st.error(error)
        else:
            top = sorted(PLAYERS, key=lambda p: p["decision_score"], reverse=True)[:30]
            pdata = "\n".join(f"- {p['name']} | {p['team']} | {p['position']} | £{p['price']:.1f}m | xGI/90 {p['xgi90']:.2f} | xA/90 {p['xa90']:.2f} | 5GW xP {p['multi_projection']:.1f}" for p in top)
            squad_text = "\n".join(f"- {p['name']} | {p['team']} | {p['position']} | 5GW xP {p['multi_projection']:.1f}" for p in MY_SQUAD)
            elite_text = "\n".join(f"- {r['name']}: {', '.join(p['name'] for p in r['squad'])}; captain={r['captain']}" for r in elite_rows)
            prompt = f"""
CURRENT GW: {CURRENT_GW}\nPLANNING GW: {NEXT_GW}\n
USER SQUAD:\n{squad_text}\n
MODEL PLAYERS:\n{pdata}\n
ELITE MANAGERS:\n{elite_text}\n
YOUTUBE TRANSCRIPT:\n{transcript[:30000]}

Extract the creator's recommendations. Separate creator-only, model-only and agreement. Flag transfers, captaincy and chip recommendations. Never invent data or quotes. Prioritise the user's actual squad.
"""
            try:
                result, model = gemini_generate(prompt, "You are an objective elite FPL analyst. Use only supplied data. Be practical and honest about uncertainty.")
                st.success(f"Analysis completed with {model}.")
                st.markdown(result)
            except Exception as e:
                st.error(str(e))


# ============================================================
# AI ASSISTANT
# ============================================================

with TABS[11]:
    st.header("💬 FPL AI Assistant")
    if not get_secret("GEMINI_API_KEY"):
        st.warning("Add GEMINI_API_KEY to Streamlit Secrets to enable this tab.")
    else:
        pin = str(get_secret("AI_ASSISTANT_PIN", "2325"))
        entered = st.text_input("Manager PIN", type="password")
        if entered != pin:
            st.info("🔒 Enter the Manager PIN to unlock the assistant.")
        else:
            if "messages_v2" not in st.session_state:
                st.session_state["messages_v2"] = []
            for m in st.session_state["messages_v2"]:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            prompt = st.chat_input("Ask about transfers, chips, captaincy, fixtures or your squad...")
            if prompt:
                st.session_state["messages_v2"].append({"role": "user", "content": prompt})
                squad_text = "\n".join(f"{p['name']} | {p['team']} | {p['position']} | 5GW xP {p['multi_projection']:.1f}" for p in MY_SQUAD)
                chip_text = json.dumps(CHIPS)
                assistant_prompt = f"""
Current GW: {CURRENT_GW}\nPlanning GW: {NEXT_GW}\nBank: £{BANK:.1f}m\nFree transfers: {FREE_TRANSFERS}\nRemaining chips: {chip_text}\n
SQUAD:\n{squad_text}\n
TRANSFER MODEL:\n{TRANSFER['reason']}\n
CAPTAIN:\n{CAPTAINS[0]['name'] if CAPTAINS else 'No clear captain'}\n
CHIP PLAN:\n{sequence}\n
USER QUESTION:\n{prompt}\n
Give practical, decisive FPL advice. Prioritise the actual squad. Use supplied data. Explain uncertainty. Do not invent statistics.
"""
                with st.chat_message("assistant"):
                    try:
                        answer, model = gemini_generate(assistant_prompt, "You are an elite FPL strategist. Be data-led, practical and honest.")
                        st.markdown(answer)
                        st.caption(f"Model: {model}")
                        st.session_state["messages_v2"].append({"role": "assistant", "content": answer})
                    except Exception as e:
                        st.error(str(e))


# ============================================================
# USER GUIDE
# ============================================================

with TABS[12]:
    st.header("ℹ️ Simple User Guide")
    st.markdown("""
### 🏠 Dashboard
Your quick weekly decision: transfer, captain, warnings and the players with the strongest underlying potential.

### 👥 Squad
Shows your actual 15 players, expected output, underlying metrics and the best XI for the next Gameweek.

### 🔁 Transfers
Compares players you own with realistic replacements while respecting budget and the three-player-per-club rule. Hits are only recommended when the projected gain is large enough.

### 📈 Players
The scouting database. **xG/xA/xGI are not treated as goals/assists:** the model is designed to look at underlying performance, minutes and fixtures.

### 🎯 Captain
Ranks captain candidates using expected Gameweek output, underlying potential, minutes and elite-manager consensus.

### 🧠 Chip Optimiser
The key new feature. It asks **when your chips are worth the most**, not simply whether the next Gameweek is a DGW. It compares your actual squad with alternative chip timings and looks for sequences such as **Wildcard → Bench Boost**.

### 📅 Fixtures
Shows upcoming blank/double Gameweeks and fixture runs.

### 📚 History
Historical seasons are used as a supporting evidence layer. They help stabilise player priors and identify historical BGW/DGW patterns. Current-season data remains more important.

### 🏆 Elite Managers
Tracks the selected high-performing managers and compares their squads/captains with yours.

### ⚔️ Mini-Leagues
Shows your rank and immediate rivals.

### 📺 Creator AI
Paste a YouTube FPL video. Gemini compares the creator's recommendations with your squad and the model.

### 💬 AI Assistant
Ask normal questions about your team. It receives the model's current information rather than being asked to guess.

### Important
No model can predict FPL perfectly. The goal is to make **better decisions repeatedly**, especially by comparing alternatives rather than reacting to one Gameweek.
""")

st.divider()
st.caption("FPL Assistant Manager 2.0 • Official FPL API + historical archive + underlying metrics + squad optimisation + chip strategy + elite consensus")
