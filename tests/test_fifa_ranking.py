from datetime import date
from pathlib import Path

from kinela.fifa_ranking import (
    fifa_ranking_for_match_date,
    load_fifa_ranking_history,
    normalise_team_name,
    update_live_fifa_points,
)


def test_historical_ranking_becomes_available_on_publication_date(
    tmp_path: Path,
) -> None:
    output = tmp_path / "processed" / "fifa"
    output.mkdir(parents=True)
    (output / "mens_ranking_history.csv").write_text(
        "team_key,rank,total_points,confederation,schedule_id,"
        "match_window_end_date,official_date,visibility_date,published_at\n"
        "guatemala,100,1200.0,CONCACAF,old,2024-01-01,2024-01-04,2024-01-04,"
        "2024-01-04T09:00:00+00:00\n"
        "guatemala,90,1250.0,CONCACAF,new,2024-02-01,2024-02-05,2024-02-05,"
        "2024-02-05T09:00:00+00:00\n",
        encoding="utf-8",
    )

    history = load_fifa_ranking_history(tmp_path)

    before_publication = fifa_ranking_for_match_date(
        history,
        "Guatemala",
        date(2024, 2, 3),
    )
    after_publication = fifa_ranking_for_match_date(
        history,
        "Guatemala",
        date(2024, 2, 6),
    )
    assert before_publication is not None
    assert before_publication["schedule_id"] == "old"
    assert after_publication is not None
    assert after_publication["schedule_id"] == "new"


def test_live_fifa_points_reward_an_upset_without_future_data() -> None:
    favorite, underdog = update_live_fifa_points(1700.0, 1400.0, 0, 1)

    assert favorite < 1700.0
    assert underdog > 1400.0
    assert favorite + underdog == 3100.0


def test_live_fifa_points_downweight_friendly_result() -> None:
    full_a, full_b = update_live_fifa_points(1700.0, 1400.0, 0, 1)
    friendly_a, friendly_b = update_live_fifa_points(
        1700.0,
        1400.0,
        0,
        1,
        importance_weight=0.6,
    )

    assert abs(friendly_a - 1700.0) < abs(full_a - 1700.0)
    assert abs(friendly_b - 1400.0) < abs(full_b - 1400.0)
    assert friendly_a + friendly_b == 3100.0


def test_provider_team_aliases_match_official_fifa_names() -> None:
    assert normalise_team_name("Rep. Of Ireland") == "republic of ireland"
    assert normalise_team_name("Bosnia & Herzegovina") == "bosnia and herzegovina"
    assert normalise_team_name("FYR Macedonia") == "north macedonia"
    assert normalise_team_name("Kyrgyzstan") == "kyrgyz republic"
