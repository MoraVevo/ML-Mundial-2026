from pathlib import Path

from kinela.providers.football_data import FootballData


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
        return {"scorers": []}


def test_competition_scorers_cache_is_season_specific(tmp_path: Path) -> None:
    provider = FootballData(tmp_path, token="test-token")
    provider.client = _FakeClient()

    provider.competition_scorers("PL", season=2022, limit=100)

    assert provider.client.calls == [
        {
            "endpoint": "competitions/PL/scorers",
            "params": {"limit": 100, "season": 2022},
            "cache_name": "competitions/PL/scorers-season-2022-limit-100.json",
            "refresh": False,
        }
    ]
