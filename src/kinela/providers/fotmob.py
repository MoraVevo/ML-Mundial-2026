from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any


FOTMOB_BASE_URL = "https://www.fotmob.com"
WORLD_CUP_LEAGUE_ID = 77
WORLD_CUP_SEASONS = ("2022", "2026")


class FotMobWorldCup:
    def __init__(self, data_root: Path) -> None:
        self.raw_dir = data_root / "raw" / "fotmob" / "world_cup"
        self.headers = {"User-Agent": "Mozilla/5.0"}

    def _fetch_text(self, url: str) -> str:
        request = urllib.request.Request(url, headers=self.headers)
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read().decode("utf-8")

    @staticmethod
    def _next_data(html: str) -> dict[str, Any]:
        match = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
        )
        if not match:
            raise RuntimeError("FotMob page did not include __NEXT_DATA__")
        return json.loads(match.group(1))

    def collect_world_cups(
        self,
        *,
        seasons: tuple[str, ...] = WORLD_CUP_SEASONS,
        refresh: bool = False,
        min_interval_seconds: float = 0.25,
    ) -> dict[str, Any]:
        summary: list[dict[str, Any]] = []
        for season in seasons:
            season_dir = self.raw_dir / season
            season_dir.mkdir(parents=True, exist_ok=True)
            fixture_path = season_dir / "fixtures.json"
            if fixture_path.exists() and not refresh:
                fixtures_payload = json.loads(fixture_path.read_text(encoding="utf-8"))
            else:
                url = f"{FOTMOB_BASE_URL}/leagues/{WORLD_CUP_LEAGUE_ID}/matches/world-cup?season={season}"
                html = self._fetch_text(url)
                fixtures_payload = self._next_data(html)
                fixture_path.write_text(
                    json.dumps(fixtures_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                time.sleep(min_interval_seconds)

            fixtures = (
                fixtures_payload.get("props", {})
                .get("pageProps", {})
                .get("fixtures", {})
                .get("allMatches", [])
            )
            finished = [item for item in fixtures if item.get("status", {}).get("finished")]
            downloaded = 0
            for match in finished:
                match_id = str(match.get("id") or "")
                page_url = str(match.get("pageUrl") or "").split("#")[0]
                if not match_id or not page_url:
                    continue
                match_path = season_dir / "matches" / f"{match_id}.json"
                if match_path.exists() and not refresh:
                    downloaded += 1
                    continue
                match_path.parent.mkdir(parents=True, exist_ok=True)
                html = self._fetch_text(f"{FOTMOB_BASE_URL}{page_url}#{match_id}")
                payload = self._next_data(html)
                match_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                downloaded += 1
                time.sleep(min_interval_seconds)
            summary.append(
                {
                    "season": season,
                    "fixtures": len(fixtures),
                    "finished": len(finished),
                    "match_pages": downloaded,
                }
            )
        return {"provider": "fotmob", "competition": "FIFA World Cup", "summary": summary}

