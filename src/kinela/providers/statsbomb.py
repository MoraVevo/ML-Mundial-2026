from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from kinela.http import CachedJsonClient

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
WORLD_CUP_2022_COMPETITION_ID = 43
WORLD_CUP_2022_SEASON_ID = 106

MEN_EXTRA_TIME_TOURNAMENTS = (
    (1267, 107, "African Cup of Nations", "2023"),
    (223, 282, "Copa America", "2024"),
    (43, 106, "FIFA World Cup", "2022"),
    (43, 3, "FIFA World Cup", "2018"),
    (55, 282, "UEFA Euro", "2024"),
    (55, 43, "UEFA Euro", "2020"),
)


def _is_knockout_match(match: dict[str, Any]) -> bool:
    stage = str((match.get("competition_stage") or {}).get("name") or "").casefold()
    return bool(stage) and "group" not in stage


class StatsBombOpenData:
    def __init__(self, data_root: Path) -> None:
        self.raw_dir = data_root / "raw" / "statsbomb"
        self.client = CachedJsonClient(BASE_URL, self.raw_dir)

    def collect_world_cup_2022(
        self,
        *,
        include_360: bool = True,
        refresh: bool = False,
        workers: int = 8,
    ) -> dict[str, Any]:
        competition_id = WORLD_CUP_2022_COMPETITION_ID
        season_id = WORLD_CUP_2022_SEASON_ID
        self.client.get(
            "competitions.json",
            cache_name="competitions.json",
            refresh=refresh,
        )
        matches = self.client.get(
            f"matches/{competition_id}/{season_id}.json",
            cache_name=f"matches/{competition_id}/{season_id}.json",
            refresh=refresh,
        )

        match_ids = [int(match["match_id"]) for match in matches]

        def download_match(match_id: int) -> tuple[int, bool]:
            self.client.get(
                f"events/{match_id}.json",
                cache_name=f"events/{match_id}.json",
                refresh=refresh,
            )
            self.client.get(
                f"lineups/{match_id}.json",
                cache_name=f"lineups/{match_id}.json",
                refresh=refresh,
            )
            has_360 = False
            if include_360:
                try:
                    self.client.get(
                        f"three-sixty/{match_id}.json",
                        cache_name=f"three-sixty/{match_id}.json",
                        refresh=refresh,
                    )
                    has_360 = True
                except RuntimeError as exc:
                    if "HTTP 404" not in str(exc):
                        raise
            return match_id, has_360

        with ThreadPoolExecutor(max_workers=workers) as executor:
            downloaded = list(executor.map(download_match, match_ids))

        manifest = {
            "provider": "statsbomb-open-data",
            "competition_id": competition_id,
            "season_id": season_id,
            "competition": "FIFA World Cup",
            "season": "2022",
            "matches": len(match_ids),
            "event_files": len(match_ids),
            "lineup_files": len(match_ids),
            "three_sixty_files": sum(has_360 for _, has_360 in downloaded),
            "source": "https://github.com/statsbomb/open-data",
            "attribution": "Data provided by StatsBomb",
        }
        manifest_path = self.raw_dir.parent.parent / "manifests" / "statsbomb-world-cup-2022.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest

    def collect_men_extra_time_tournaments(
        self,
        *,
        refresh: bool = False,
        workers: int = 8,
    ) -> dict[str, Any]:
        """Cache knockout events from recent open men's national tournaments."""

        self.client.get(
            "competitions.json",
            cache_name="competitions.json",
            refresh=refresh,
        )
        tournament_rows: list[dict[str, Any]] = []
        download_queue: list[int] = []
        for competition_id, season_id, competition, season in MEN_EXTRA_TIME_TOURNAMENTS:
            matches = self.client.get(
                f"matches/{competition_id}/{season_id}.json",
                cache_name=f"matches/{competition_id}/{season_id}.json",
                refresh=refresh,
            )
            knockout = [match for match in matches if _is_knockout_match(match)]
            match_ids = [int(match["match_id"]) for match in knockout]
            download_queue.extend(match_ids)
            tournament_rows.append(
                {
                    "competition_id": competition_id,
                    "season_id": season_id,
                    "competition": competition,
                    "season": season,
                    "all_matches": len(matches),
                    "knockout_event_files": len(match_ids),
                }
            )

        def download_events(match_id: int) -> int:
            self.client.get(
                f"events/{match_id}.json",
                cache_name=f"events/{match_id}.json",
                refresh=refresh,
            )
            return match_id

        unique_match_ids = sorted(set(download_queue))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            downloaded = list(executor.map(download_events, unique_match_ids))

        manifest = {
            "provider": "statsbomb-open-data",
            "scope": "recent-men-national-team-knockouts-for-extra-time",
            "tournaments": tournament_rows,
            "event_files": len(downloaded),
            "source": "https://github.com/statsbomb/open-data",
            "attribution": "Data provided by StatsBomb",
        }
        manifest_path = (
            self.raw_dir.parent.parent / "manifests" / "statsbomb-men-extra-time.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
