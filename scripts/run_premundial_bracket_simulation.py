from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402

from kinela.worldcup_2026 import (  # noqa: E402
    QUARTER_FINALS,
    R32_SLOTS,
    ROUND_OF_16,
    SEMI_FINALS,
    WorldCup2026Simulator,
)
from kinela.worldcup_2026 import _normalise_name  # noqa: E402


LEFT_R32_MATCHES = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_MATCHES = [76, 78, 79, 80, 86, 88, 85, 87]
R16_BY_MATCH = {match_id: (previous_a, previous_b) for match_id, previous_a, previous_b in ROUND_OF_16}
QF_BY_MATCH = {match_id: (previous_a, previous_b) for match_id, previous_a, previous_b in QUARTER_FINALS}
SF_BY_MATCH = {match_id: (previous_a, previous_b) for match_id, previous_a, previous_b in SEMI_FINALS}
MATCH_DATES = {
    "ROUND_OF_32": date(2026, 6, 28),
    "LAST_16": date(2026, 7, 4),
    "QUARTER_FINALS": date(2026, 7, 9),
    "SEMI_FINALS": date(2026, 7, 14),
    "FINAL": date(2026, 7, 19),
}


def _reset_simulator_to_premundial(
    simulator: WorldCup2026Simulator,
    data_root: Path,
) -> dict[str, Any]:
    model_path = data_root / "models" / "lightgbm_neutral_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            "Missing pre-World-Cup holdout model: "
            f"{model_path}. Run `kinela train lightgbm-neutral` first."
        )
    simulator.lightgbm_model = joblib.load(model_path)

    original_history = len(simulator.history)
    simulator.manual_results = {}
    simulator.history = [
        row
        for row in simulator.history
        if row.get("source") != "manual-worldcup-2026"
        and not (
            str(row.get("competition_name", "")).casefold() == "fifa world cup"
            and row["date_obj"] >= date(2026, 6, 11)
        )
    ]
    for team, rows in list(simulator.player_availability.items()):
        simulator.player_availability[team] = [
            row for row in rows if row.get("source") != "manual-worldcup-2026"
        ]

    simulator.simulated_histories = defaultdict(list)
    simulator.current_match_records = []
    simulator.prediction_cache.clear()
    simulator.team_histories = simulator._index_team_histories()
    simulator.head_to_head_histories = simulator._index_head_to_head_histories()
    simulator.worldcup_histories = simulator._index_worldcup_histories()
    simulator.elo_ratings = simulator._build_elo_ratings()
    (
        simulator.confederation_stats,
        simulator.team_cross_confederation_stats,
    ) = simulator._build_confederation_contexts()
    simulator.fifa_point_overrides = simulator._build_fifa_point_overrides()
    contexts = simulator._goal_contexts()
    simulator.global_goal_avg = contexts["global_avg"]
    simulator.major_goal_avg = contexts["major_avg"]
    simulator.group_goal_avg = contexts["group_avg"]
    simulator.knockout_goal_avg = contexts["knockout_avg"]
    simulator.major_match_count = contexts["major_matches"]
    simulator.group_match_count = contexts["group_matches"]
    simulator.knockout_match_count = contexts["knockout_matches"]
    simulator.prediction_cache.clear()

    return {
        "model_path": str(model_path),
        "manual_results_used": False,
        "history_rows_before_filter": original_history,
        "history_rows_after_filter": len(simulator.history),
        "excluded_worldcup_2026_history_rows": original_history - len(simulator.history),
    }


def _pick_unused(counter: Counter[str], used: set[str]) -> tuple[str, int]:
    for team, count in counter.most_common():
        if team not in used:
            used.add(team)
            return team, count
    team, count = counter.most_common(1)[0]
    return team, count


def _winner_prediction(
    simulator: WorldCup2026Simulator,
    match_id: int,
    stage: str,
    team_a: str,
    team_b: str,
    support_a: int,
    support_b: int,
) -> dict[str, Any]:
    prediction = simulator.lightgbm_prediction(
        team_a,
        team_b,
        MATCH_DATES[stage],
        stage,
    )
    probabilities = prediction["probabilities"]
    probability_a = float(probabilities[0])
    probability_draw = float(probabilities[1])
    probability_b = float(probabilities[2])
    penalty_probability_a = simulator.penalty_model.team_a_probability(
        team_a,
        team_b,
        MATCH_DATES[stage],
    )
    advance_a = probability_a + probability_draw * penalty_probability_a
    advance_b = probability_b + probability_draw * (1 - penalty_probability_a)
    winner = team_a if advance_a >= advance_b else team_b
    return {
        "match_id": match_id,
        "stage": stage,
        "team_a": team_a,
        "team_b": team_b,
        "support_a": support_a,
        "support_b": support_b,
        "probability_a": round(probability_a, 4),
        "probability_draw": round(probability_draw, 4),
        "probability_b": round(probability_b, 4),
        "advance_probability_a": round(advance_a, 4),
        "advance_probability_b": round(advance_b, 4),
        "expected_goals_a": round(float(prediction["team_a_goals"]), 3),
        "expected_goals_b": round(float(prediction["team_b_goals"]), 3),
        "winner": winner,
        "winner_advance_probability": round(max(advance_a, advance_b), 4),
    }


def _build_consensus_bracket(
    simulator: WorldCup2026Simulator,
    slot_side_counts: dict[str, dict[str, Counter[str]]],
    slot_team_counts: dict[str, Counter[str]],
) -> dict[str, dict[str, Any]]:
    r32_assignments: dict[tuple[str, str], tuple[str, int]] = {}
    used_r32_teams: set[str] = set()
    r32_sides = []
    for match_id, _, _ in R32_SLOTS:
        for side in ("team_a", "team_b"):
            counter = slot_side_counts[str(match_id)][side]
            top_count = counter.most_common(1)[0][1]
            r32_sides.append((top_count, str(match_id), side, counter))
    for _, match_id, side, counter in sorted(r32_sides, reverse=True):
        r32_assignments[(match_id, side)] = _pick_unused(counter, used_r32_teams)

    rows: dict[str, dict[str, Any]] = {}
    winners: dict[int, str] = {}
    for match_id, _, _ in R32_SLOTS:
        team_a, support_a = r32_assignments[(str(match_id), "team_a")]
        team_b, support_b = r32_assignments[(str(match_id), "team_b")]
        row = _winner_prediction(
            simulator,
            match_id,
            "ROUND_OF_32",
            team_a,
            team_b,
            support_a,
            support_b,
        )
        rows[str(match_id)] = row
        winners[match_id] = row["winner"]

    for match_id, previous_a, previous_b in ROUND_OF_16:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        row = _winner_prediction(
            simulator,
            match_id,
            "LAST_16",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        rows[str(match_id)] = row
        winners[match_id] = row["winner"]

    for match_id, previous_a, previous_b in QUARTER_FINALS:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        row = _winner_prediction(
            simulator,
            match_id,
            "QUARTER_FINALS",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        rows[str(match_id)] = row
        winners[match_id] = row["winner"]

    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        row = _winner_prediction(
            simulator,
            match_id,
            "SEMI_FINALS",
            team_a,
            team_b,
            slot_team_counts[str(match_id)][team_a],
            slot_team_counts[str(match_id)][team_b],
        )
        rows[str(match_id)] = row
        winners[match_id] = row["winner"]

    final_row = _winner_prediction(
        simulator,
        103,
        "FINAL",
        winners[101],
        winners[102],
        slot_team_counts["103"][winners[101]],
        slot_team_counts["103"][winners[102]],
    )
    rows["103"] = final_row
    return rows


def _prediction_from_cache(
    simulator: WorldCup2026Simulator,
    team_a: str,
    team_b: str,
    match_date: date,
    stage: str,
) -> dict[str, Any]:
    key = (
        _normalise_name(team_a),
        _normalise_name(team_b),
        match_date.isoformat(),
        stage,
    )
    prediction = simulator.prediction_cache.get(key)
    if prediction is None:
        prediction = simulator.lightgbm_prediction(team_a, team_b, match_date, stage, None)
    return prediction


def _align_scores(
    goals_a: np.ndarray,
    goals_b: np.ndarray,
    outcomes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a_win = outcomes == 0
    draw = outcomes == 1
    b_win = outcomes == 2
    goals_a = goals_a.copy()
    goals_b = goals_b.copy()
    goals_a[a_win & (goals_a <= goals_b)] = goals_b[a_win & (goals_a <= goals_b)] + 1
    goals_b[b_win & (goals_b <= goals_a)] = goals_a[b_win & (goals_b <= goals_a)] + 1
    goals_b[draw] = goals_a[draw]
    return goals_a, goals_b


def _rank_group(
    rng: np.random.Generator,
    teams: list[str],
    points: dict[str, int],
    goals_for: dict[str, int],
    goals_against: dict[str, int],
) -> list[str]:
    return sorted(
        teams,
        key=lambda team: (
            points[team],
            goals_for[team] - goals_against[team],
            goals_for[team],
            float(rng.random()),
        ),
        reverse=True,
    )


def _resolve_slot(
    slot: tuple[str, str | tuple[str, ...]],
    rankings: dict[str, list[str]],
    thirds: list[tuple[str, str]],
    assignments: dict[int, str],
    match_id: int,
) -> str:
    kind, value = slot
    if kind == "1":
        return rankings[str(value)][0]
    if kind == "2":
        return rankings[str(value)][1]
    assigned_group = f"GROUP_{assignments[match_id]}"
    for group, team in thirds:
        if group == assigned_group:
            return team
    raise ValueError(f"Missing assigned third-place team for {match_id}: {assigned_group}")


def _sample_knockout_winner(
    simulator: WorldCup2026Simulator,
    rng: np.random.Generator,
    team_a: str,
    team_b: str,
    stage: str,
) -> str:
    prediction = _prediction_from_cache(simulator, team_a, team_b, MATCH_DATES[stage], stage)
    probs = np.asarray(prediction["probabilities"], dtype=float)
    probs = probs / probs.sum()
    outcome = int(rng.choice(3, p=probs))
    if outcome == 0:
        return team_a
    if outcome == 2:
        return team_b
    penalty_probability_a = simulator.penalty_model.team_a_probability(team_a, team_b, MATCH_DATES[stage])
    return team_a if float(rng.random()) < penalty_probability_a else team_b


def _run_fast_tournaments(
    simulator: WorldCup2026Simulator,
    *,
    simulations: int,
    seed: int,
    progress_every: int,
) -> tuple[Counter[str], Counter[str], Counter[str], dict[str, dict[str, Counter[str]]], dict[str, Counter[str]]]:
    rng = np.random.default_rng(seed)
    teams = sorted({team for group in simulator.groups.values() for team in group})
    team_index = {team: index for index, team in enumerate(teams)}
    points = np.zeros((simulations, len(teams)), dtype=np.int16)
    goals_for = np.zeros_like(points)
    goals_against = np.zeros_like(points)

    for match in simulator.group_matches:
        team_a = match["homeTeam"]["name"]
        team_b = match["awayTeam"]["name"]
        match_date = date.fromisoformat(match["utcDate"][:10])
        prediction = _prediction_from_cache(simulator, team_a, team_b, match_date, "GROUP_STAGE")
        probs = np.asarray(prediction["probabilities"], dtype=float)
        probs = probs / probs.sum()
        outcomes = rng.choice(3, size=simulations, p=probs)
        goals_a = rng.poisson(max(0.05, float(prediction["team_a_goals"])), size=simulations)
        goals_b = rng.poisson(max(0.05, float(prediction["team_b_goals"])), size=simulations)
        goals_a, goals_b = _align_scores(goals_a, goals_b, outcomes)
        idx_a = team_index[team_a]
        idx_b = team_index[team_b]
        goals_for[:, idx_a] += goals_a
        goals_against[:, idx_a] += goals_b
        goals_for[:, idx_b] += goals_b
        goals_against[:, idx_b] += goals_a
        points[:, idx_a] += np.where(outcomes == 0, 3, np.where(outcomes == 1, 1, 0)).astype(np.int16)
        points[:, idx_b] += np.where(outcomes == 2, 3, np.where(outcomes == 1, 1, 0)).astype(np.int16)

    champions: Counter[str] = Counter()
    finalists: Counter[str] = Counter()
    semifinalists: Counter[str] = Counter()
    slot_side_counts: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    slot_team_counts: dict[str, Counter[str]] = defaultdict(Counter)

    for sim_index in range(simulations):
        rankings: dict[str, list[str]] = {}
        third_candidates: list[tuple[tuple[int, int, int, float], str, str]] = []
        for group, group_teams in simulator.groups.items():
            p = {team: int(points[sim_index, team_index[team]]) for team in group_teams}
            gf = {team: int(goals_for[sim_index, team_index[team]]) for team in group_teams}
            ga = {team: int(goals_against[sim_index, team_index[team]]) for team in group_teams}
            ranking = _rank_group(rng, list(group_teams), p, gf, ga)
            rankings[group] = ranking
            third = ranking[2]
            third_candidates.append(
                (
                    (p[third], gf[third] - ga[third], gf[third], float(rng.random())),
                    group,
                    third,
                )
            )
        thirds = [
            (group, team)
            for _score, group, team in sorted(third_candidates, reverse=True)[:8]
        ]
        qualified_groups = "".join(sorted(group.replace("GROUP_", "") for group, _team in thirds))
        assignments = simulator.third_place_assignment_table[qualified_groups]
        winners: dict[int, str] = {}

        for match_id, slot_a, slot_b in R32_SLOTS:
            team_a = _resolve_slot(slot_a, rankings, thirds, assignments, match_id)
            team_b = _resolve_slot(slot_b, rankings, thirds, assignments, match_id)
            slot_side_counts[str(match_id)]["team_a"][team_a] += 1
            slot_side_counts[str(match_id)]["team_b"][team_b] += 1
            slot_team_counts[str(match_id)][team_a] += 1
            slot_team_counts[str(match_id)][team_b] += 1
            winners[match_id] = _sample_knockout_winner(
                simulator,
                rng,
                team_a,
                team_b,
                "ROUND_OF_32",
            )

        for match_id, previous_a, previous_b in ROUND_OF_16:
            team_a = winners[previous_a]
            team_b = winners[previous_b]
            slot_side_counts[str(match_id)]["team_a"][team_a] += 1
            slot_side_counts[str(match_id)]["team_b"][team_b] += 1
            slot_team_counts[str(match_id)][team_a] += 1
            slot_team_counts[str(match_id)][team_b] += 1
            winners[match_id] = _sample_knockout_winner(simulator, rng, team_a, team_b, "LAST_16")

        for match_id, previous_a, previous_b in QUARTER_FINALS:
            team_a = winners[previous_a]
            team_b = winners[previous_b]
            slot_side_counts[str(match_id)]["team_a"][team_a] += 1
            slot_side_counts[str(match_id)]["team_b"][team_b] += 1
            slot_team_counts[str(match_id)][team_a] += 1
            slot_team_counts[str(match_id)][team_b] += 1
            winners[match_id] = _sample_knockout_winner(
                simulator,
                rng,
                team_a,
                team_b,
                "QUARTER_FINALS",
            )
            semifinalists[winners[match_id]] += 1

        for match_id, previous_a, previous_b in SEMI_FINALS:
            team_a = winners[previous_a]
            team_b = winners[previous_b]
            slot_side_counts[str(match_id)]["team_a"][team_a] += 1
            slot_side_counts[str(match_id)]["team_b"][team_b] += 1
            slot_team_counts[str(match_id)][team_a] += 1
            slot_team_counts[str(match_id)][team_b] += 1
            winners[match_id] = _sample_knockout_winner(simulator, rng, team_a, team_b, "SEMI_FINALS")

        team_a = winners[101]
        team_b = winners[102]
        slot_side_counts["103"]["team_a"][team_a] += 1
        slot_side_counts["103"]["team_b"][team_b] += 1
        slot_team_counts["103"][team_a] += 1
        slot_team_counts["103"][team_b] += 1
        champion = _sample_knockout_winner(simulator, rng, team_a, team_b, "FINAL")
        champions[champion] += 1
        finalists[team_a] += 1
        finalists[team_b] += 1

        if progress_every and (sim_index + 1) % progress_every == 0:
            print(f"{sim_index + 1}/{simulations} premundial simulations complete", flush=True)

    return champions, finalists, semifinalists, slot_side_counts, slot_team_counts


def _short_name(name: str) -> str:
    aliases = {
        "United States": "USA",
        "Cape Verde Islands": "Cape Verde",
        "Curaçao": "Curacao",
        "Bosnia and Herzegovina": "Bosnia-Herz.",
        "Czech Republic": "Czechia",
        "Ivory Coast": "Cote d'Ivoire",
        "Korea Republic": "South Korea",
        "United Arab Emirates": "UAE",
    }
    return aliases.get(name, name)


def _draw_bracket(
    bracket: dict[str, dict[str, Any]],
    champion_probability: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10.5), dpi=180)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#050606")
    ax.set_facecolor("#050606")

    # Subtle pitch/poster texture.
    for index, color in enumerate(["#1be7ff", "#ff3f7f", "#ff5a2d", "#7dff35"]):
        ax.add_patch(
            Rectangle(
                (index * 0.25, 0),
                0.25,
                1,
                color=color,
                alpha=0.12,
                linewidth=0,
            )
        )
    ax.add_patch(
        FancyBboxPatch(
            (0.015, 0.025),
            0.97,
            0.95,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="#050606",
            edgecolor="#7dff35",
            linewidth=1.2,
            alpha=0.97,
        )
    )

    line = "#86f24b"
    box = "#8bff3c"
    text = "#071007"
    white = "#f8fff2"
    muted = "#b7c8b0"

    ax.text(
        0.5,
        0.93,
        "BRACKET PREMUNDIAL 2026",
        ha="center",
        va="center",
        color=white,
        fontsize=30,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.885,
        f"10,000 simulaciones sin partidos disputados del Mundial | Campeon consenso: {_short_name(bracket['103']['winner'])} ({champion_probability:.1%})",
        ha="center",
        va="center",
        color=line,
        fontsize=13,
        fontweight="bold",
    )

    left_leaf_y = [0.84 - i * 0.047 for i in range(16)]
    right_leaf_y = left_leaf_y
    left_r32_y = [(left_leaf_y[i] + left_leaf_y[i + 1]) / 2 for i in range(0, 16, 2)]
    right_r32_y = left_r32_y
    left_r16_y = [(left_r32_y[i] + left_r32_y[i + 1]) / 2 for i in range(0, 8, 2)]
    right_r16_y = left_r16_y
    left_qf_y = [(left_r16_y[i] + left_r16_y[i + 1]) / 2 for i in range(0, 4, 2)]
    right_qf_y = left_qf_y
    left_sf_y = sum(left_qf_y) / 2
    right_sf_y = left_sf_y
    final_y = 0.5

    nodes: dict[int | str, tuple[float, float]] = {}

    def draw_box(x: float, y: float, label: str, prob: float | None = None, *, side: str = "left") -> None:
        width = 0.135
        height = 0.028
        x0 = x if side == "left" else x - width
        ax.add_patch(
            FancyBboxPatch(
                (x0, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=box,
                edgecolor=box,
                linewidth=0,
            )
        )
        suffix = "" if prob is None else f" {prob:.0%}"
        ax.text(
            x0 + width / 2,
            y,
            f"{_short_name(label)}{suffix}",
            ha="center",
            va="center",
            color=text,
            fontsize=7.8,
            fontweight="bold",
        )

    def connect(x1: float, y1: float, x2: float, y2: float, *, side: str) -> None:
        mid = (x1 + x2) / 2
        ax.plot([x1, mid], [y1, y1], color=line, linewidth=1.7)
        ax.plot([mid, mid], [y1, y2], color=line, linewidth=1.7)
        ax.plot([mid, x2], [y2, y2], color=line, linewidth=1.7)

    def add_match(
        match_id: int,
        y: float,
        x: float,
        input_nodes: list[tuple[float, float]],
        *,
        side: str,
    ) -> None:
        row = bracket[str(match_id)]
        winner_prob = row["winner_advance_probability"]
        draw_box(x, y, row["winner"], winner_prob, side=side)
        nodes[match_id] = (x if side == "left" else x - 0.135, y)
        target_x = x if side == "left" else x - 0.135
        for input_x, input_y in input_nodes:
            connect(input_x, input_y, target_x, y, side=side)

    for idx, match_id in enumerate(LEFT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = left_leaf_y[idx * 2], left_leaf_y[idx * 2 + 1]
        draw_box(0.04, y1, row["team_a"], row["support_a"] / 10000, side="left")
        draw_box(0.04, y2, row["team_b"], row["support_b"] / 10000, side="left")
        add_match(match_id, left_r32_y[idx], 0.20, [(0.175, y1), (0.175, y2)], side="left")

    for idx, match_id in enumerate(RIGHT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = right_leaf_y[idx * 2], right_leaf_y[idx * 2 + 1]
        draw_box(0.96, y1, row["team_a"], row["support_a"] / 10000, side="right")
        draw_box(0.96, y2, row["team_b"], row["support_b"] / 10000, side="right")
        add_match(match_id, right_r32_y[idx], 0.80, [(0.825, y1), (0.825, y2)], side="right")

    left_r16_matches = [89, 90, 93, 94]
    right_r16_matches = [91, 92, 95, 96]
    for idx, match_id in enumerate(left_r16_matches):
        previous_a, previous_b = R16_BY_MATCH[match_id]
        add_match(
            match_id,
            left_r16_y[idx],
            0.34,
            [nodes[previous_a], nodes[previous_b]],
            side="left",
        )
    for idx, match_id in enumerate(right_r16_matches):
        previous_a, previous_b = R16_BY_MATCH[match_id]
        add_match(
            match_id,
            right_r16_y[idx],
            0.66,
            [nodes[previous_a], nodes[previous_b]],
            side="right",
        )

    for idx, match_id in enumerate([97, 98]):
        previous_a, previous_b = QF_BY_MATCH[match_id]
        add_match(match_id, left_qf_y[idx], 0.46, [nodes[previous_a], nodes[previous_b]], side="left")
    for idx, match_id in enumerate([99, 100]):
        previous_a, previous_b = QF_BY_MATCH[match_id]
        add_match(match_id, right_qf_y[idx], 0.54, [nodes[previous_a], nodes[previous_b]], side="right")

    add_match(101, left_sf_y, 0.49, [nodes[97], nodes[98]], side="left")
    add_match(102, right_sf_y, 0.51, [nodes[99], nodes[100]], side="right")

    final = bracket["103"]
    ax.plot([0.49, 0.49, 0.50], [left_sf_y, final_y, final_y], color=line, linewidth=2.2)
    ax.plot([0.51, 0.51, 0.50], [right_sf_y, final_y, final_y], color=line, linewidth=2.2)
    ax.add_patch(
        FancyBboxPatch(
            (0.395, final_y - 0.045),
            0.21,
            0.09,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="#f9fff5",
            edgecolor=line,
            linewidth=2.0,
            zorder=20,
        )
    )
    ax.text(
        0.5,
        final_y + 0.018,
        "CAMPEON",
        ha="center",
        va="center",
        color=text,
        fontsize=10,
        zorder=21,
    )
    ax.text(
        0.5,
        final_y - 0.012,
        _short_name(final["winner"]),
        ha="center",
        va="center",
        color=text,
        fontsize=20,
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        0.5,
        0.08,
        "Bracket consenso: clasificados y cruces mas frecuentes en simulacion; cada cruce se reevalua con el modelo premundial.",
        ha="center",
        va="center",
        color=muted,
        fontsize=9,
    )

    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def run_premundial_simulation(
    data_root: Path,
    *,
    simulations: int,
    seed: int,
    output_dir: Path,
    progress_every: int,
) -> dict[str, Any]:
    simulator = WorldCup2026Simulator(data_root, seed=seed, engine="poisson")
    simulator.engine = "lightgbm"
    metadata = _reset_simulator_to_premundial(simulator, data_root)
    original_lightgbm_prediction = simulator.lightgbm_prediction

    def frozen_premundial_prediction(
        team_a: str,
        team_b: str,
        match_date: date,
        stage: str,
        group_table: dict[str, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return original_lightgbm_prediction(team_a, team_b, match_date, stage, None)

    def skip_simulated_history_update(*_args: Any, **_kwargs: Any) -> None:
        return None

    simulator.lightgbm_prediction = frozen_premundial_prediction  # type: ignore[method-assign]
    simulator._record_simulated_match = skip_simulated_history_update  # type: ignore[method-assign]
    metadata["premundial_context"] = "frozen_pre_tournament_predictions"
    metadata["simulated_results_update_model_state"] = False
    simulator._precompute_lightgbm_cache()
    metadata["precomputed_prediction_cache_entries"] = len(simulator.prediction_cache)

    started = datetime.now()
    (
        champions,
        finalists,
        semifinalists,
        slot_side_counts,
        slot_team_counts,
    ) = _run_fast_tournaments(
        simulator,
        simulations=simulations,
        seed=seed,
        progress_every=progress_every,
    )

    bracket = _build_consensus_bracket(simulator, slot_side_counts, slot_team_counts)
    champion = bracket["103"]["winner"]
    champion_probability = champions[champion] / simulations
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "worldcup_2026_premundial_bracket.png"
    summary_path = output_dir / "worldcup_2026_premundial_bracket.json"
    _draw_bracket(bracket, champion_probability, image_path)

    payload = {
        "simulations": simulations,
        "seed": seed,
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        **metadata,
        "champion_consensus": champion,
        "champion_consensus_probability": round(champion_probability, 6),
        "top_champions": [
            {"team": team, "probability": round(count / simulations, 6)}
            for team, count in champions.most_common(12)
        ],
        "top_finalists": [
            {"team": team, "probability": round(count / simulations, 6)}
            for team, count in finalists.most_common(12)
        ],
        "top_semifinalists": [
            {"team": team, "probability": round(count / simulations, 6)}
            for team, count in semifinalists.most_common(12)
        ],
        "bracket": bracket,
        "image_path": str(image_path),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--runs", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets"))
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    payload = run_premundial_simulation(
        args.data_root,
        simulations=args.runs,
        seed=args.seed,
        output_dir=args.output_dir,
        progress_every=args.progress_every,
    )
    print(
        json.dumps(
            {
                "simulations": payload["simulations"],
                "champion_consensus": payload["champion_consensus"],
                "champion_consensus_probability": payload["champion_consensus_probability"],
                "image_path": payload["image_path"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
