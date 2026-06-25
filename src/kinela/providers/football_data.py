from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from kinela.http import CachedJsonClient


class FootballData:
    def __init__(self, data_root: Path, token: str | None = None) -> None:
        auth_token = token or os.getenv("FOOTBALL_DATA_TOKEN")
        if not auth_token:
            raise RuntimeError("FOOTBALL_DATA_TOKEN is not configured")
        self.client = CachedJsonClient(
            "https://api.football-data.org/v4",
            data_root / "raw" / "football_data",
            headers={"X-Auth-Token": auth_token},
            min_interval_seconds=6.5,
        )

    def team_matches(
        self,
        *,
        team_id: int,
        limit: int = 15,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            f"teams/{team_id}/matches",
            {"status": "FINISHED", "limit": limit},
            cache_name=f"teams/{team_id}/matches-last-{limit}.json",
            refresh=refresh,
        )

    def competition_teams(
        self,
        competition_code: str,
        *,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            f"competitions/{competition_code}/teams",
            cache_name=f"competitions/{competition_code}/teams.json",
            refresh=refresh,
        )

    def competitions(self, *, refresh: bool = False) -> dict[str, Any]:
        return self.client.get(
            "competitions",
            cache_name="competitions.json",
            refresh=refresh,
        )

    def competition_matches(
        self,
        competition_code: str,
        *,
        season: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        params = {"season": season} if season else {}
        suffix = f"-season-{season}" if season else ""
        return self.client.get(
            f"competitions/{competition_code}/matches",
            params,
            cache_name=f"competitions/{competition_code}/matches{suffix}.json",
            refresh=refresh,
        )

    def competition_scorers(
        self,
        competition_code: str,
        *,
        limit: int = 100,
        season: int | None = None,
        refresh: bool = False,
    ) -> dict[str, Any]:
        params: dict[str, int] = {"limit": limit}
        suffix = ""
        if season is not None:
            params["season"] = season
            suffix = f"-season-{season}"
        return self.client.get(
            f"competitions/{competition_code}/scorers",
            params,
            cache_name=(
                f"competitions/{competition_code}/"
                f"scorers{suffix}-limit-{limit}.json"
            ),
            refresh=refresh,
        )

    def team(self, team_id: int, *, refresh: bool = False) -> dict[str, Any]:
        return self.client.get(
            f"teams/{team_id}",
            cache_name=f"teams/{team_id}.json",
            refresh=refresh,
        )

    def person_matches(
        self,
        *,
        person_id: int,
        limit: int = 100,
        refresh: bool = False,
    ) -> dict[str, Any]:
        return self.client.get(
            f"persons/{person_id}/matches",
            {"status": "FINISHED", "limit": limit},
            cache_name=f"persons/{person_id}/matches-last-{limit}.json",
            refresh=refresh,
        )
