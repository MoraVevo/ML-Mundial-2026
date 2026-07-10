import csv
from collections import Counter
from datetime import date

import joblib
import numpy as np

import scripts.run_worldcup2026_consensus_bracket as consensus
from kinela.model import DETAIL_STAT_FEATURES
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
    simulator._lightgbm_prediction_frames = lambda rows: {
        "team_a_goals": rows,
        "team_b_goals": rows,
        "result": rows,
        "xg_result": rows,
    }
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
    simulator._lightgbm_prediction_frames = lambda rows: {
        "team_a_goals": rows,
        "team_b_goals": rows,
        "result": rows,
        "xg_result": rows,
    }

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


def test_lightgbm_prediction_blends_xg_result_classifier() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.prediction_cache = {}
    simulator.lightgbm_model = {
        "team_a_goals_model": _FakeModel([1.3]),
        "team_b_goals_model": _FakeModel([0.9]),
        "result_model": _FakeModel([[0.2, 0.3, 0.5]]),
        "xg_result_model": _FakeModel([[0.4, 0.2, 0.4]]),
        "result_probability_blend_weight": 0.5,
    }
    simulator._cache_key = lambda *args: args
    simulator._lightgbm_features = lambda *args, **kwargs: {"feature": 1.0}
    simulator._lightgbm_prediction_frames = lambda rows: {
        "team_a_goals": rows,
        "team_b_goals": rows,
        "result": rows,
        "xg_result": rows,
    }

    prediction = simulator.lightgbm_prediction(
        "France",
        "Morocco",
        date(2026, 7, 10),
        "QUARTER_FINALS",
    )

    assert prediction["team_a_goals"] == 1.3
    assert prediction["team_b_goals"] == 0.9
    np.testing.assert_allclose(
        prediction["probabilities"],
        np.array([0.3, 0.25, 0.45]),
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


def test_tournament_state_resets_live_fifa_points_between_runs() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.tournament_start_fifa_point_overrides = {
        "france": 1800.0,
        "morocco": 1650.0,
    }
    simulator.base_fifa_point_overrides = {"france": 1850.0, "morocco": 1700.0}
    simulator.fifa_point_overrides = {"france": 1901.0, "morocco": 1649.0}
    simulator.simulated_histories = {"france": [{"gf": 1}]}
    simulator.current_match_records = [{"match_id": 103}]
    simulator.prediction_cache = {"stale": {"team_a_goals": 9.0}}

    simulator._reset_tournament_state()

    assert simulator.fifa_point_overrides == simulator.tournament_start_fifa_point_overrides
    assert simulator.fifa_point_overrides is not simulator.tournament_start_fifa_point_overrides
    assert not simulator.simulated_histories
    assert simulator.current_match_records == []
    assert simulator.prediction_cache == {}


def test_fifa_point_baselines_separate_direct_prediction_from_tournament_replay() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.fifa_rankings = {
        "france": {"points": 1800.0},
        "morocco": {"points": 1650.0},
    }
    simulator.fallback_fifa_points = 800.0
    simulator.history = [
        {
            "source": "manual-worldcup-2026",
            "home_team": "France",
            "away_team": "Morocco",
            "home_goals": 1,
            "away_goals": 0,
        }
    ]

    tournament_start = simulator._build_fifa_point_overrides(
        include_manual_worldcup=False,
    )
    direct_prediction = simulator._build_fifa_point_overrides(
        include_manual_worldcup=True,
    )

    assert tournament_start == {"france": 1800.0, "morocco": 1650.0}
    assert direct_prediction != tournament_start
    assert direct_prediction["france"] > tournament_start["france"]
    assert direct_prediction["morocco"] < tournament_start["morocco"]


def test_refresh_derived_state_rebuilds_fifa_baselines_and_clears_cache() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.fifa_rankings = {
        "france": {"points": 1800.0},
        "morocco": {"points": 1650.0},
    }
    simulator.fallback_fifa_points = 800.0
    simulator.history = [
        {
            "source": "manual-worldcup-2026",
            "home_team": "France",
            "away_team": "Morocco",
            "home_goals": 1,
            "away_goals": 0,
        }
    ]
    simulator.simulated_histories = {"france": [{"gf": 1}]}
    simulator.current_match_records = [{"match_id": 90}]
    simulator.prediction_cache = {"stale": {"team_a_goals": 9.0}}
    simulator._index_team_histories = lambda: {"france": []}
    simulator._index_head_to_head_histories = lambda: {}
    simulator._index_worldcup_histories = lambda: {}
    simulator._build_elo_ratings = lambda: {"france": 1510.0}
    simulator._build_confederation_contexts = lambda: ({}, {})
    simulator._goal_contexts = lambda: {
        "global_avg": 2.4,
        "major_avg": 2.1,
        "group_avg": 2.0,
        "knockout_avg": 1.8,
        "major_matches": 10,
        "group_matches": 6,
        "knockout_matches": 4,
    }

    simulator._refresh_derived_state()

    assert not simulator.simulated_histories
    assert simulator.current_match_records == []
    assert simulator.prediction_cache == {}
    assert simulator.tournament_start_fifa_point_overrides == {
        "france": 1800.0,
        "morocco": 1650.0,
    }
    assert simulator.base_fifa_point_overrides["france"] > 1800.0
    assert simulator.fifa_point_overrides == simulator.base_fifa_point_overrides
    assert simulator.knockout_match_count == 4


def test_missing_actual_fotmob_detail_is_not_filled_from_recent_average() -> None:
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.late85_points_swing_metrics = {}
    simulator.score_timing_metrics = {}
    row = {
        "match_id": "fd:999",
        "date_obj": date(2026, 7, 6),
        "source": "manual-worldcup-2026",
        "competition_name": "FIFA World Cup",
        "home_team": "France",
        "away_team": "Morocco",
        "home_goals": 1,
        "away_goals": 0,
        "home_opponent_elo_pre": 1700.0,
        "away_opponent_elo_pre": 1850.0,
        "home_elo_pre": 1850.0,
        "away_elo_pre": 1700.0,
    }
    for side in ("home", "away"):
        for feature in DETAIL_STAT_FEATURES:
            row[f"{side}_actual_{feature}"] = ""
            row[f"{side}_recent6_{feature}_avg"] = "0.75"
    simulator.history = [row]

    histories = simulator._index_team_histories()

    france_detail = histories["france"][0]["detail_stats"]
    assert france_detail["total_shots"] == 0.75
    assert france_detail["fotmob_expected_goals"] is None
    assert france_detail["fotmob_detail_coverage"] is None


def test_simulator_prefers_worldcup_holdout_lightgbm_model(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(
        {"model_id": "standard-heldout"},
        models_dir / "lightgbm_neutral_model.joblib",
    )
    joblib.dump(
        {"model_id": "worldcup-holdout"},
        models_dir / "lightgbm_neutral_worldcup_holdout.joblib",
    )
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.data_root = tmp_path

    model = simulator._load_lightgbm_model()

    assert model["model_id"] == "worldcup-holdout"


def test_simulator_prefers_all_played_worldcup_model_when_available(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    joblib.dump(
        {"model_id": "standard-heldout"},
        models_dir / "lightgbm_neutral_model.joblib",
    )
    joblib.dump(
        {"model_id": "worldcup-holdout"},
        models_dir / "lightgbm_neutral_worldcup_holdout.joblib",
    )
    joblib.dump(
        {"model_id": "all-played"},
        models_dir / "lightgbm_neutral_all_played_wc2026.joblib",
    )
    simulator = WorldCup2026Simulator.__new__(WorldCup2026Simulator)
    simulator.data_root = tmp_path

    model = simulator._load_lightgbm_model()

    assert model["model_id"] == "all-played"


def test_consensus_script_prefers_all_played_model_path(tmp_path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    for name in (
        "lightgbm_neutral_model.joblib",
        "lightgbm_neutral_worldcup_holdout.joblib",
        "lightgbm_neutral_all_played_wc2026.joblib",
    ):
        (models_dir / name).write_text("placeholder", encoding="utf-8")

    selected = consensus._selected_model_path(tmp_path, "lightgbm")

    assert selected.endswith("lightgbm_neutral_all_played_wc2026.joblib")


def test_consensus_worker_run_counts_cover_all_runs() -> None:
    counts = consensus._worker_run_counts(10_000, 8)

    assert counts == [1250] * 8
    assert sum(counts) == 10_000
    assert consensus._worker_run_counts(3, 8) == [1, 1, 1]


def test_consensus_merge_keeps_matchup_conditional_winners() -> None:
    target = consensus._empty_counts()
    source = consensus._empty_counts()
    source["slot_matchups"]["93"][("Spain", "Belgium")] = 3
    source["slot_winners"]["93"]["Spain"] = 2
    source["slot_winners"]["93"]["Belgium"] = 1
    source["slot_matchup_winners"]["93"][("Spain", "Belgium")] = Counter(
        {"Spain": 2, "Belgium": 1}
    )
    source["full_brackets"][(("93", "LAST_16", "Spain", "Belgium", "Spain"),)] = 3

    consensus._merge_counts(target, source)
    summary = consensus._summarize_counts(
        target,
        runs=3,
        seed=42,
        engine="lightgbm",
        selected_model_path="model.joblib",
        model_label="test",
        fast_mode=False,
        workers=1,
        started=consensus.datetime(2026, 7, 7, 12, 0, 0),
    )

    matchup = summary["slot_summary"][0]["top_matchups"][0]
    assert matchup["top_winner"] == "Spain"
    assert matchup["top_winner_count"] == 2
    assert matchup["top_winner_probability_given_matchup"] == 0.666667


def test_matchup_csv_writes_winner_conditional_on_matchup(tmp_path) -> None:
    result = {
        "team_summary": [
            {
                "equipo": "Spain",
                "veces_campeon": 1,
                "prob_campeon": 1.0,
            }
        ],
        "slot_summary": [
            {
                "match_id": "93",
                "top_matchups": [
                    {
                        "team_a": "Spain",
                        "team_b": "Belgium",
                        "count": 3,
                        "probability": 0.3,
                        "top_winner": "Belgium",
                        "top_winner_count": 2,
                        "top_winner_probability_given_matchup": 0.666667,
                    }
                ],
                "top_winners": [{"item": "Spain", "count": 5, "probability": 0.5}],
            }
        ],
    }

    paths = consensus._write_outputs(result, tmp_path / "summary.json")

    rows = list(csv.DictReader(open(paths["matchup_csv_path"], encoding="utf-8")))
    assert rows[0]["winner"] == "Belgium"
    assert rows[0]["winner_probability_given_matchup"] == "0.666667"
