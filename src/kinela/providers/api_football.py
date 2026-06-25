from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from kinela.http import CachedJsonClient


class ApiFootball:
    def __init__(self, data_root: Path, api_key: str | None = None) -> None:
        key = api_key or os.getenv("APISPORTS_KEY")
        if not key:
            raise RuntimeError("APISPORTS_KEY is not configured")
        self.client = CachedJsonClient(
            "https://v3.football.api-sports.io",
            data_root / "raw" / "api_football",
            headers={"x-apisports-key": key},
            min_interval_seconds=6.5,
        )

    def fixtures(
        self,
        *,
        team_id: int,
        last: int = 15,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            "fixtures",
            {"team": team_id, "last": last, "status": "FT-AET-PEN"},
            cache_name=f"fixtures/team-{team_id}-last-{last}.json",
            refresh=refresh,
        )

    def fixtures_by_season(
        self,
        *,
        team_id: int,
        season: int,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            "fixtures",
            {"team": team_id, "season": season},
            cache_name=f"fixtures/team-{team_id}-season-{season}.json",
            refresh=refresh,
        )

    def fixtures_by_date(
        self,
        *,
        date: str,
        league_id: int | None = None,
        season: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        parts = [f"date-{date}"]
        if league_id is not None:
            parts.append(f"league-{league_id}")
        if season is not None:
            parts.append(f"season-{season}")
        return self.client.get(
            "fixtures",
            {"date": date, "league": league_id, "season": season},
            cache_name=f"fixtures/{'-'.join(parts)}.json",
            refresh=refresh,
        )

    def fixture_by_id(self, fixture_id: int, *, refresh: bool = False) -> dict[str, Any]:
        return self.client.get(
            "fixtures",
            {"id": fixture_id},
            cache_name=f"fixtures/fixture-{fixture_id}.json",
            refresh=refresh,
        )

    def teams_by_league_season(
        self,
        *,
        league_id: int,
        season: int,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            "teams",
            {"league": league_id, "season": season},
            cache_name=f"teams/league-{league_id}-season-{season}.json",
            refresh=refresh,
        )

    def standings_by_league_season(
        self,
        *,
        league_id: int,
        season: int,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            "standings",
            {"league": league_id, "season": season},
            cache_name=f"standings/league-{league_id}-season-{season}.json",
            refresh=refresh,
        )

    def fixture_details(self, fixture_id: int, *, refresh: bool = False) -> dict[str, Any]:
        # A request by fixture id includes events, lineups, fixture statistics,
        # and player statistics when the competition provides them.
        return self.client.get(
            "fixtures",
            {"id": fixture_id},
            cache_name=f"fixtures/details-{fixture_id}.json",
            refresh=refresh or not self.fixture_details_cached(fixture_id),
        )

    def fixture_details_batch(
        self,
        fixture_ids: list[int],
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        # API-Football supports up to 20 fixture IDs in one request and returns
        # the same detail blocks as a single fixture lookup.
        ids = [int(fixture_id) for fixture_id in fixture_ids[:20]]
        ids_text = "-".join(str(fixture_id) for fixture_id in ids)
        digest = hashlib.sha256(ids_text.encode("utf-8")).hexdigest()[:12]
        payload = self.client.get(
            "fixtures",
            {"ids": ids_text},
            cache_name=f"fixtures/details-batch-{digest}.json",
            refresh=refresh,
        )
        for item in payload.get("response") or []:
            fixture_id = (item.get("fixture") or {}).get("id")
            if fixture_id is None:
                continue
            detail_path = self.client.cache_dir / f"fixtures/details-{int(fixture_id)}.json"
            if detail_path.exists() and not refresh:
                continue
            detail_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = detail_path.with_suffix(".tmp")
            temporary.write_text(
                json.dumps({"response": [item]}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(detail_path)
        return payload

    def fixture_details_cached(self, fixture_id: int) -> bool:
        path = self.client.cache_dir / f"fixtures/details-{fixture_id}.json"
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return False
        return bool(payload.get("response"))

    def players_by_team_season(
        self,
        *,
        team_id: int,
        season: int,
        refresh: bool = False,
    ) -> list[dict[str, Any]]:
        pages: list[dict[str, Any]] = []
        page = 1
        while True:
            payload = self.client.get(
                "players",
                {"team": team_id, "season": season, "page": page},
                cache_name=f"players/team-{team_id}-season-{season}-page-{page}.json",
                refresh=refresh,
            )
            pages.append(payload)
            if page >= int(payload.get("paging", {}).get("total", 1)):
                return pages
            page += 1

    def player_by_id_season(
        self,
        *,
        player_id: int,
        season: int,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            "players",
            {"id": player_id, "season": season},
            cache_name=f"players/player-{player_id}-season-{season}.json",
            refresh=refresh,
        )
