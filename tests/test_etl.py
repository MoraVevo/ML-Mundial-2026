from kinela.etl import _competition_stage, _competition_type, _fd_result


def test_competition_type_friendlies() -> None:
    assert _competition_type(10, "Friendlies", "Friendlies 1") == "friendly"


def test_competition_type_qualifier() -> None:
    assert (
        _competition_type(32, "World Cup - Qualification Europe", "Group Stage - 4")
        == "qualifier"
    )


def test_competition_stage_group() -> None:
    assert _competition_stage("Group E - 2") == "group_stage"


def test_competition_stage_qualifying_round() -> None:
    assert _competition_stage("Qualifying Round - 8") == "Qualifying Round - 8"


def test_football_data_result_falls_back_to_score() -> None:
    assert _fd_result(None, 1, 2) == "away"

