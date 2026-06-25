from pathlib import Path

from kinela.providers.api_football import ApiFootball


class _FakeClient:
    def __init__(self) -> None:
        self.calls = []

    def get(self, endpoint, params, *, cache_name, refresh):
        self.calls.append(
            {
                "endpoint": endpoint,
                "params": params,
                "cache_name": cache_name,
                "refresh": refresh,
            }
        )
        return {"response": []}


def test_player_by_id_season_has_stable_cache_path(tmp_path: Path) -> None:
    provider = ApiFootball(tmp_path, api_key="test-key")
    provider.client = _FakeClient()

    provider.player_by_id_season(player_id=1100, season=2024)

    assert provider.client.calls == [
        {
            "endpoint": "players",
            "params": {"id": 1100, "season": 2024},
            "cache_name": "players/player-1100-season-2024.json",
            "refresh": False,
        }
    ]
