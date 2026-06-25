from datetime import date

import joblib
import numpy as np

from kinela.worldcup_2026 import WorldCup2026Simulator


class _FakeModel:
    def __init__(self, values: list[float] | list[list[float]]) -> None:
        self.values = np.asarray(values)

    def predict(self, _frame):
        return self.values

    def predict_proba(self, _frame):
        return self.values


def test_lightgbm_prediction_does_not_apply_untrained_squad_multiplier() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.prediction_cache = {}
    simulator.lightgbm_model = {
        "team_a_goals_model": _FakeModel([0.8]),
        "team_b_goals_model": _FakeModel([1.4]),
        "result_model": _FakeModel([[0.2, 0.3, 0.5]]),
    }
    simulator._cache_key = lambda *args: args
    simulator._lightgbm_features = lambda *args, **kwargs: {"feature": 1.0}
    simulator._lightgbm_frame = lambda rows: rows
    simulator._squad_strength_multipliers = lambda *args: (_ for _ in ()).throw(
        AssertionError("squad multiplier must not be called")
    )

    prediction = simulator.lightgbm_prediction(
        "Sweden",
        "Tunisia",
        date(2026, 6, 15),
        "GROUP_STAGE",
    )

    assert prediction["team_a_goals"] == 0.8
    assert prediction["team_b_goals"] == 1.4
    np.testing.assert_allclose(
        prediction["probabilities"],
        np.array([0.2, 0.3, 0.5]),
    )


def test_lightgbm_prediction_bypasses_cache_when_group_table_is_provided() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.lightgbm_model = {
        "team_a_goals_model": _FakeModel([1.1]),
        "team_b_goals_model": _FakeModel([0.7]),
        "result_model": _FakeModel([[0.55, 0.25, 0.20]]),
    }
    cache_key = ("sweden", "tunisia", "2026-06-15", "GROUP_STAGE")
    simulator.prediction_cache = {
        cache_key: {
            "team_a_goals": 9.0,
            "team_b_goals": 9.0,
            "probabilities": np.array([0.0, 1.0, 0.0]),
        }
    }
    calls = []
    simulator._cache_key = lambda *args: cache_key
    simulator._lightgbm_features = lambda *args, **kwargs: calls.append(kwargs) or {"feature": 1.0}
    simulator._lightgbm_frame = lambda rows: rows

    prediction = simulator.lightgbm_prediction(
        "Sweden",
        "Tunisia",
        date(2026, 6, 15),
        "GROUP_STAGE",
        group_table={"Sweden": {"played": 1, "points": 3}},
    )

    assert calls
    assert prediction["team_a_goals"] == 1.1
    assert prediction["team_b_goals"] == 0.7
    np.testing.assert_allclose(
        prediction["probabilities"],
        np.array([0.55, 0.25, 0.20]),
    )


def test_lightgbm_stage_context_uses_training_canonical_labels() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.group_match_count = 10
    simulator.group_goal_avg = 1.2
    simulator.knockout_match_count = 8
    simulator.knockout_goal_avg = 1.0

    assert simulator._stage_context("GROUP_STAGE")["stage_or_round"] == "GROUP_STAGE"
    assert simulator._stage_context("Round of 32")["stage_or_round"] == "ROUND_OF_32"
    assert simulator._stage_context("LAST_16")["stage_or_round"] == "LAST_16"
    assert simulator._stage_context("Quarter-finals")["stage_or_round"] == "QUARTER_FINALS"


def test_simulator_prefers_all_played_future_lightgbm_model(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(
        {"feature_recipe": "standard-heldout"},
        models_dir / "lightgbm_neutral_model.joblib",
    )
    joblib.dump(
        {"feature_recipe": "all-played-future"},
        models_dir / "lightgbm_neutral_all_played_wc2026.joblib",
    )
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.data_root = tmp_path

    model = simulator._load_lightgbm_model()

    assert model["feature_recipe"] == "all-played-future"
