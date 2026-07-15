import pandas as pd

import scripts.generate_model_evaluation_report as report
import scripts.update_worldcup2026_manual_detail_from_espn as espn_update
import scripts.worldcup2026_model_metrics as model_metrics
import scripts.worldcup2026_out_of_sample_comparison as oos_comparison


def test_worldcup_split_uses_only_prior_training_rows() -> None:
    training = pd.DataFrame(
        [
            {
                "date": "2026-06-01",
                "source": "provider",
                "competition_name": "Friendly",
                "is_friendly": True,
            },
            {
                "date": "2026-06-10",
                "source": "provider",
                "competition_name": "World Cup - Qualification Europe",
                "is_friendly": False,
            },
            {
                "date": "2026-06-11",
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
                "is_friendly": False,
            },
            {
                "date": "2026-06-12",
                "source": "provider",
                "competition_name": "World Cup - Qualification Europe",
                "is_friendly": False,
            },
        ]
    )
    clean = pd.DataFrame({"split": ["train", "train", "train", "train"]})

    split, info = report._split_worldcup_2026(training, clean)

    assert split["split"].tolist() == ["train", "train", "test", "excluded"]
    assert "before the first World Cup 2026 match" in info["policy"]


def test_external_random_split_excludes_same_and_later_nonselected_rows() -> None:
    rows = []
    for index in range(12):
        rows.append(
            {
                "row_index": index,
                "date": f"2025-01-{index + 1:02d}",
                "source": "provider",
                "competition_name": "UEFA Nations League",
                "is_friendly": False,
            }
        )
    training = pd.DataFrame(rows)
    clean = pd.DataFrame({"split": ["train"] * len(training)})

    split, _ = report._split_external_random_temporal(
        training,
        clean,
        test_matches=4,
        seed=7,
    )
    test_dates = pd.to_datetime(training.loc[split["split"].eq("test"), "date"])
    cutoff = test_dates.min()
    later_non_test = pd.to_datetime(training["date"]).ge(cutoff) & ~split["split"].eq("test")

    assert set(split["split"].unique()) == {"train", "test", "excluded"}
    assert split.loc[later_non_test, "split"].eq("excluded").all()


def test_latest_manual_result_date_uses_manual_file(tmp_path) -> None:
    static_dir = tmp_path / "static"
    static_dir.mkdir()
    (static_dir / "worldcup_2026_manual_results.csv").write_text(
        "\n".join(
            [
                "match_id,date,team_a,team_b,team_a_goals,team_b_goals,winner",
                "1,2026-06-22,A,B,1,0,A",
                "2,2026-06-24,C,D,0,0,Draw",
                "3,2026-06-23,E,F,2,1,E",
            ]
        ),
        encoding="utf-8",
    )

    assert model_metrics._latest_manual_result_date(tmp_path) == "2026-06-24"


def test_espn_knockout_sync_does_not_add_unfinished_events() -> None:
    events = {
        espn_update._team_key("Argentina", "Egypt"): {
            "id": "760508",
            "competitions": [
                {
                    "status": {"type": {"completed": False}},
                    "competitors": [
                        {"team": {"displayName": "Argentina"}, "score": "1"},
                        {"team": {"displayName": "Egypt"}, "score": "0"},
                    ],
                }
            ],
        }
    }

    rows, added = espn_update._sync_finished_knockout_results([], events)

    assert rows == []
    assert added == []


def test_espn_knockout_sync_treats_string_false_completed_as_unfinished() -> None:
    events = {
        espn_update._team_key("Argentina", "Egypt"): {
            "id": "760508",
            "competitions": [
                {
                    "status": {"type": {"completed": "false"}},
                    "competitors": [
                        {"team": {"displayName": "Argentina"}, "score": "1"},
                        {"team": {"displayName": "Egypt"}, "score": "0"},
                    ],
                }
            ],
        }
    }

    rows, added = espn_update._sync_finished_knockout_results([], events)

    assert rows == []
    assert added == []


def test_espn_knockout_sync_adds_only_completed_events() -> None:
    events = {
        espn_update._team_key("Argentina", "Egypt"): {
            "id": "760508",
            "competitions": [
                {
                    "status": {"type": {"completed": True}},
                    "competitors": [
                        {
                            "team": {"displayName": "Argentina"},
                            "score": "2",
                            "winner": True,
                        },
                        {"team": {"displayName": "Egypt"}, "score": "0"},
                    ],
                }
            ],
        }
    }

    rows, added = espn_update._sync_finished_knockout_results([], events)

    assert len(rows) == 1
    assert added == rows
    assert rows[0]["match_id"] == "95"
    assert rows[0]["winner"] == "Argentina"
    assert rows[0]["source"] == "ESPN event 760508; elimination bracket metadata"


def test_walkforward_split_uses_prior_rows_and_target_only() -> None:
    training = pd.DataFrame(
        [
            {
                "row_index": 10,
                "date": "2026-06-10",
                "source": "provider",
                "competition_name": "World Cup - Qualification Europe",
            },
            {
                "row_index": 11,
                "date": "2026-06-11",
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
            },
            {
                "row_index": 12,
                "date": "2026-06-12",
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
            },
            {
                "row_index": 13,
                "date": "2026-06-13",
                "source": "manual-worldcup-2026",
                "competition_name": "FIFA World Cup",
            },
        ]
    )
    clean = pd.DataFrame({"split": ["train"] * len(training)})

    split, info = oos_comparison.split_worldcup_2026_walkforward(
        training,
        clean,
        target_index=2,
    )

    assert split["split"].tolist() == ["train", "train", "test", "excluded"]
    assert "incluyendo partidos previos del Mundial 2026" in info["policy"]
