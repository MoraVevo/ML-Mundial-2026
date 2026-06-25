from kinela.normalize import _goal_type


def test_goal_type_penalty() -> None:
    event = {"shot": {"type": {"name": "Penalty"}}, "location": [108, 40]}
    assert _goal_type(event) == "penalty"


def test_goal_type_own_half() -> None:
    event = {"shot": {"type": {"name": "Open Play"}}, "location": [49, 40]}
    assert _goal_type(event) == "own_half_or_halfway"


def test_goal_type_shootout() -> None:
    event = {"period": 5, "shot": {"type": {"name": "Penalty"}}}
    assert _goal_type(event) == "penalty_shootout"
