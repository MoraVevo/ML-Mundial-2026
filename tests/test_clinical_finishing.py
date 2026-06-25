from kinela.clinical_finishing import clinical_finishing_summary


def test_clinical_summary_aligns_goals_with_observed_shots() -> None:
    observed = [
        {
            "goals_for": 2,
            "shots_on_goal": 4,
            "total_shots": 10,
        }
    ]
    with_unobserved_goals = [
        *observed,
        {
            "goals_for": 5,
            "shots_on_goal": None,
            "total_shots": None,
        },
    ]

    baseline = clinical_finishing_summary(observed)
    candidate = clinical_finishing_summary(with_unobserved_goals)

    assert candidate["clinical_signal"] == baseline["clinical_signal"]
    assert candidate["clinical_coverage"] == baseline["clinical_coverage"]


def test_clinical_summary_caps_goals_above_recorded_shots_on_goal() -> None:
    impossible = clinical_finishing_summary(
        [
            {
                "goals_for": 4,
                "shots_on_goal": 1,
                "total_shots": 3,
            }
        ]
    )
    capped = clinical_finishing_summary(
        [
            {
                "goals_for": 1,
                "shots_on_goal": 1,
                "total_shots": 3,
            }
        ]
    )

    assert impossible["clinical_signal"] == capped["clinical_signal"]
