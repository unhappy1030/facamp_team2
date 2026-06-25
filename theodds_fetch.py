"""
the-odds-api 2026 월드컵 배당률 수집 스크립트.

soccer_fifa_world_cup 키에서 다가오는 경기의 1X2(h2h) 배당률을 받아
북메이커별/평균 내재확률(vig 제거)까지 만들어 parquet 으로 저장한다.

사용 전:
    export ODDS_API_KEY="발급키"     # https://the-odds-api.com/#get-access (무료)
    conda activate fac

실행 예:
    python theodds_fetch.py --list                 # 현재 시즌 중인 종목 키 확인
    python theodds_fetch.py                         # 다가오는 월드컵 경기 배당률
    python theodds_fetch.py --scores --days 3       # 최근 끝난 경기 결과(라벨용)

출력:
    raw/odds_<sport>_<ts>.json     원본 응답
    out/odds.parquet               (경기 x 북메이커) 배당 long 테이블
    out/odds_avg.parquet           경기별 북메이커 평균 + vig 제거 내재확률

비용 주의:
    /odds 요청 1건 = (markets 수 x regions 수) 크레딧 소모.
    historical(과거 pre-match 스냅샷)은 유료 플랜 전용이라 여기 미포함.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# --- 설정 -------------------------------------------------------------------

BASE_URL = "https://api.the-odds-api.com/v4"
API_KEY = os.environ.get("ODDS_API_KEY")

SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR / "raw"
OUT_DIR = SCRIPT_DIR / "out"

SPORT = "soccer_fifa_world_cup"
DEFAULT_REGIONS = "uk,eu"   # us,uk,eu,au 중 선택(콤마구분). 많을수록 크레딧↑
MARKETS = "h2h"             # 1X2
ODDS_FORMAT = "decimal"


# --- HTTP -------------------------------------------------------------------

def _require_key() -> str:
    if not API_KEY:
        raise SystemExit(
            "환경변수 ODDS_API_KEY 가 없습니다.\n"
            '  export ODDS_API_KEY="..." 후 다시 실행하세요.'
        )
    return API_KEY


def _get(path: str, params: dict) -> tuple[object, dict]:
    """GET 요청. (json, 남은쿼터헤더) 반환."""
    url = f"{BASE_URL}/{path.lstrip('/')}"
    params = dict(params, apiKey=_require_key())
    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        raise SystemExit(f"요청 실패 {resp.status_code} {url}\n{resp.text[:500]}")
    quota = {
        "remaining": resp.headers.get("x-requests-remaining"),
        "used": resp.headers.get("x-requests-used"),
        "last_cost": resp.headers.get("x-requests-last"),
    }
    return resp.json(), quota


# --- 명령들 -----------------------------------------------------------------

def list_sports() -> None:
    """현재 접근 가능한 종목 키 목록(월드컵 키가 활성인지 확인용)."""
    data, quota = _get("sports", {"all": "true"})
    print(f"쿼터 remaining={quota['remaining']} used={quota['used']}\n")
    for s in data:
        if "soccer" in s.get("key", ""):
            active = "active" if s.get("active") else "  off "
            print(f"  [{active}] {s['key']:40} {s.get('title','')}")
    print("\n(soccer_fifa_world_cup 가 active 여야 다가오는 경기 배당이 나옵니다)")


def _save_raw(name: str, data: object) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{name}_{ts}.json"
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  raw 저장: {path}")


def fetch_odds(regions: str) -> None:
    """다가오는 월드컵 경기의 h2h 배당률 수집."""
    data, quota = _get(
        f"sports/{SPORT}/odds",
        {"regions": regions, "markets": MARKETS, "oddsFormat": ODDS_FORMAT},
    )
    print(f"이벤트 {len(data)}건 / 쿼터 remaining={quota['remaining']} "
          f"used={quota['used']} 이번비용={quota['last_cost']}")
    _save_raw(f"odds_{SPORT}", data)

    long_rows = flatten_odds(data)
    if not long_rows:
        print("배당 데이터가 비어있습니다(아직 오픈된 경기 없음일 수 있음).")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    long_df = pd.DataFrame(long_rows)
    long_df.to_parquet(OUT_DIR / "odds.parquet", index=False)
    print(f"odds(long) 저장: {OUT_DIR/'odds.parquet'}  shape={long_df.shape}")

    avg_df = aggregate_and_devig(long_df)
    avg_df.to_parquet(OUT_DIR / "odds_avg.parquet", index=False)
    print(f"odds_avg 저장: {OUT_DIR/'odds_avg.parquet'}  shape={avg_df.shape}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n미리보기:\n", avg_df.head(10).to_string(index=False))


def fetch_scores(days: int) -> None:
    """최근 끝난 경기 결과(백테스트 라벨용)."""
    data, quota = _get(f"sports/{SPORT}/scores", {"daysFrom": str(days)})
    print(f"경기 {len(data)}건 / 쿼터 remaining={quota['remaining']}")
    _save_raw(f"scores_{SPORT}", data)

    rows = []
    for ev in data:
        scores = {s["name"]: s.get("score") for s in (ev.get("scores") or [])}
        rows.append({
            "id": ev.get("id"),
            "commence_time": ev.get("commence_time"),
            "completed": ev.get("completed"),
            "home_team": ev.get("home_team"),
            "away_team": ev.get("away_team"),
            "home_score": scores.get(ev.get("home_team")),
            "away_score": scores.get(ev.get("away_team")),
        })
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_DIR / "scores.parquet", index=False)
    print(f"scores 저장: {OUT_DIR/'scores.parquet'}  shape={df.shape}")


# --- 평탄화 / 확률 변환 ------------------------------------------------------

def flatten_odds(events: list[dict]) -> list[dict]:
    """이벤트 리스트 -> (경기 x 북메이커) long row.

    soccer h2h outcomes = [home_team, away_team, 'Draw'] (순서 불정).
    home/away/draw 로 매핑한다.
    """
    rows: list[dict] = []
    for ev in events:
        home, away = ev.get("home_team"), ev.get("away_team")
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key") != "h2h":
                    continue
                price = {o["name"]: o["price"] for o in mk.get("outcomes", [])}
                rows.append({
                    "event_id": ev.get("id"),
                    "commence_time": ev.get("commence_time"),
                    "home_team": home,
                    "away_team": away,
                    "bookmaker": bk.get("key"),
                    "last_update": mk.get("last_update") or bk.get("last_update"),
                    "odds_home": price.get(home),
                    "odds_away": price.get(away),
                    "odds_draw": price.get("Draw"),
                })
    return rows


def aggregate_and_devig(long_df: pd.DataFrame) -> pd.DataFrame:
    """경기별 북메이커 평균 배당 -> vig 제거 내재확률."""
    grp = (long_df
           .groupby(["event_id", "commence_time", "home_team", "away_team"],
                    as_index=False)[["odds_home", "odds_draw", "odds_away"]]
           .mean(numeric_only=True))
    inv = 1.0 / grp[["odds_home", "odds_draw", "odds_away"]]
    overround = inv.sum(axis=1)
    grp["p_home"] = inv["odds_home"] / overround
    grp["p_draw"] = inv["odds_draw"] / overround
    grp["p_away"] = inv["odds_away"] / overround
    grp["overround"] = overround
    return grp


# --- CLI --------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="the-odds-api 월드컵 배당률 수집")
    p.add_argument("--list", action="store_true", help="접근 가능한 종목 키 목록만 출력")
    p.add_argument("--scores", action="store_true", help="배당 대신 최근 경기 결과 수집")
    p.add_argument("--days", type=int, default=3, help="--scores 시 며칠 전까지(최대 3, 일부 플랜)")
    p.add_argument("--regions", default=DEFAULT_REGIONS, help="배당 지역 us,uk,eu,au (콤마)")
    args = p.parse_args()

    if args.list:
        list_sports()
    elif args.scores:
        fetch_scores(args.days)
    else:
        fetch_odds(args.regions)


if __name__ == "__main__":
    main()
