from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict
from datetime import date
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
    TeamStanding,
    WorldCup2026Simulator,
)

LEFT_R32_MATCHES = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_MATCHES = [76, 78, 79, 80, 86, 88, 85, 87]
R16_BY_MATCH = {match_id: (previous_a, previous_b) for match_id, previous_a, previous_b in ROUND_OF_16}
QF_BY_MATCH = {match_id: (previous_a, previous_b) for match_id, previous_a, previous_b in QUARTER_FINALS}
MATCH_DATES = {
    "ROUND_OF_32": date(2026, 6, 28),
    "LAST_16": date(2026, 7, 4),
    "QUARTER_FINALS": date(2026, 7, 9),
    "SEMI_FINALS": date(2026, 7, 14),
    "THIRD_PLACE": date(2026, 7, 18),
    "FINAL": date(2026, 7, 19),
}


def _short_name(name: str) -> str:
    aliases = {
        "United States": "USA",
        "Cape Verde Islands": "Cape Verde",
        "Bosnia and Herzegovina": "Bosnia-Herz.",
        "Czech Republic": "Czechia",
        "Korea Republic": "South Korea",
        "United Arab Emirates": "UAE",
        "New Zealand": "N. Zealand",
    }
    return aliases.get(name, name)


def _force_premundial_state(simulator: WorldCup2026Simulator, data_root: Path) -> dict[str, Any]:
    model_path = data_root / "models" / "lightgbm_neutral_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing premundial model: {model_path}")

    original_history = len(simulator.history)
    simulator.engine = "lightgbm"
    simulator.lightgbm_model = joblib.load(model_path)
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
        "model_path": str(model_path.resolve().relative_to(ROOT)),
        "manual_results_used": False,
        "history_rows_before_filter": original_history,
        "history_rows_after_filter": len(simulator.history),
        "excluded_worldcup_2026_history_rows": original_history - len(simulator.history),
        "bracket_mode": "deterministic_model_path",
        "third_place_assignment": "exact_495_combination_table",
    }


def _pick_result(probabilities: Any) -> str:
    if probabilities is None:
        return "draw"
    labels = ["team_a", "draw", "team_b"]
    return labels[int(np.argmax(np.asarray(probabilities, dtype=float)))]


def _score_from_prediction(prediction: dict[str, Any], result: str) -> tuple[int, int]:
    goals_a = max(0, int(round(float(prediction["team_a_goals"]))))
    goals_b = max(0, int(round(float(prediction["team_b_goals"]))))
    if result == "team_a" and goals_a <= goals_b:
        goals_a = goals_b + 1
    elif result == "team_b" and goals_b <= goals_a:
        goals_b = goals_a + 1
    elif result == "draw" and goals_a != goals_b:
        goals = min(goals_a, goals_b)
        goals_a = goals
        goals_b = goals
    return goals_a, goals_b


def _rank(table: dict[str, dict[str, Any]]) -> list[TeamStanding]:
    standings = [
        TeamStanding(
            team=row["team"],
            group=row["group"],
            points=int(row["points"]),
            goals_for=int(row["gf"]),
            goals_against=int(row["ga"]),
        )
        for row in table.values()
    ]
    return sorted(
        standings,
        key=lambda item: (item.points, item.goal_difference, item.goals_for, item.team),
        reverse=True,
    )


def _points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _play_group_stage(
    simulator: WorldCup2026Simulator,
) -> tuple[dict[str, list[TeamStanding]], list[dict[str, Any]]]:
    simulator.simulated_histories = defaultdict(list)
    simulator.current_match_records = []
    standings: dict[str, list[TeamStanding]] = {}
    match_rows: list[dict[str, Any]] = []

    for group, teams in simulator.groups.items():
        table = {
            team: {"team": team, "group": group, "points": 0, "gf": 0, "ga": 0}
            for team in teams
        }
        matches = sorted(
            [match for match in simulator.group_matches if match["group"] == group],
            key=lambda item: item["utcDate"],
        )
        for match in matches:
            team_a = match["homeTeam"]["name"]
            team_b = match["awayTeam"]["name"]
            match_date = date.fromisoformat(match["utcDate"][:10])
            prediction = simulator.lightgbm_prediction(team_a, team_b, match_date, "GROUP_STAGE", table)
            result = _pick_result(prediction["probabilities"])
            goals_a, goals_b = _score_from_prediction(prediction, result)
            winner = team_a if goals_a > goals_b else team_b if goals_b > goals_a else None

            table[team_a]["gf"] += goals_a
            table[team_a]["ga"] += goals_b
            table[team_a]["points"] += _points(goals_a, goals_b)
            table[team_b]["gf"] += goals_b
            table[team_b]["ga"] += goals_a
            table[team_b]["points"] += _points(goals_b, goals_a)
            simulator._record_simulated_match(team_a, team_b, match_date, goals_a, goals_b)
            simulator._record_match_trace(
                match.get("id"),
                "GROUP_STAGE",
                group,
                team_a,
                team_b,
                goals_a,
                goals_b,
                winner,
                decided_by="model_most_likely",
            )
            probabilities = prediction["probabilities"]
            match_rows.append(
                {
                    "match_id": match.get("id"),
                    "group": group,
                    "date": match_date.isoformat(),
                    "team_a": team_a,
                    "team_b": team_b,
                    "team_a_goals": goals_a,
                    "team_b_goals": goals_b,
                    "probability_a": round(float(probabilities[0]), 4),
                    "probability_draw": round(float(probabilities[1]), 4),
                    "probability_b": round(float(probabilities[2]), 4),
                    "winner": winner or "draw",
                }
            )
        standings[group] = _rank(table)
    return standings, match_rows


def _best_thirds(standings: dict[str, list[TeamStanding]]) -> list[TeamStanding]:
    thirds = [ranking[2] for ranking in standings.values()]
    return sorted(
        thirds,
        key=lambda item: (item.points, item.goal_difference, item.goals_for, item.team),
        reverse=True,
    )[:8]


def _resolve_slot(
    simulator: WorldCup2026Simulator,
    match_id: int,
    slot: tuple[str, str | tuple[str, ...]],
    standings: dict[str, list[TeamStanding]],
    third_assignments: dict[int, str],
) -> str:
    kind, value = slot
    if kind == "1":
        return standings[str(value)][0].team
    if kind == "2":
        return standings[str(value)][1].team
    group = f"GROUP_{third_assignments[match_id]}"
    allowed = set(value)
    if group not in allowed:
        raise ValueError(f"Invalid third-place assignment for match {match_id}: {group}")
    return standings[group][2].team


def _play_knockout_match(
    simulator: WorldCup2026Simulator,
    match_id: int,
    stage: str,
    team_a: str,
    team_b: str,
) -> dict[str, Any]:
    prediction = simulator.lightgbm_prediction(team_a, team_b, MATCH_DATES[stage], stage)
    probabilities = np.asarray(prediction["probabilities"], dtype=float)
    probability_a = float(probabilities[0])
    probability_draw = float(probabilities[1])
    probability_b = float(probabilities[2])
    penalty_a = simulator.penalty_model.team_a_probability(team_a, team_b, MATCH_DATES[stage])
    advance_a = probability_a + probability_draw * penalty_a
    advance_b = probability_b + probability_draw * (1 - penalty_a)
    winner = team_a if advance_a >= advance_b else team_b
    result = "team_a" if winner == team_a else "team_b"
    goals_a, goals_b = _score_from_prediction(prediction, result)
    simulator._record_simulated_match(team_a, team_b, MATCH_DATES[stage], goals_a, goals_b)
    simulator._record_match_trace(
        match_id,
        stage,
        None,
        team_a,
        team_b,
        goals_a,
        goals_b,
        winner,
        decided_by="model_advance_probability",
        penalty_team_a_probability=penalty_a,
    )
    return {
        "match_id": match_id,
        "stage": stage,
        "team_a": team_a,
        "team_b": team_b,
        "team_a_goals": goals_a,
        "team_b_goals": goals_b,
        "probability_a": round(probability_a, 4),
        "probability_draw": round(probability_draw, 4),
        "probability_b": round(probability_b, 4),
        "advance_probability_a": round(advance_a, 4),
        "advance_probability_b": round(advance_b, 4),
        "winner": winner,
        "winner_advance_probability": round(max(advance_a, advance_b), 4),
    }


def build_deterministic_bracket(
    data_root: Path,
    simulation_summary: Path | None,
) -> dict[str, Any]:
    simulator = WorldCup2026Simulator(data_root, seed=42, engine="poisson")
    metadata = _force_premundial_state(simulator, data_root)
    standings, group_matches = _play_group_stage(simulator)
    best_thirds = _best_thirds(standings)
    third_assignments = simulator._third_place_match_assignments(best_thirds)

    bracket: dict[str, dict[str, Any]] = {}
    winners: dict[int, str] = {}
    r32_teams: list[str] = []
    for match_id, slot_a, slot_b in R32_SLOTS:
        team_a = _resolve_slot(simulator, match_id, slot_a, standings, third_assignments)
        team_b = _resolve_slot(simulator, match_id, slot_b, standings, third_assignments)
        r32_teams.extend([team_a, team_b])
        row = _play_knockout_match(simulator, match_id, "ROUND_OF_32", team_a, team_b)
        bracket[str(match_id)] = row
        winners[match_id] = row["winner"]

    expected_teams = {row.team for ranking in standings.values() for row in ranking[:2]}
    expected_teams |= {row.team for row in best_thirds}
    if len(r32_teams) != 32 or len(set(r32_teams)) != 32 or set(r32_teams) != expected_teams:
        raise ValueError("Round-of-32 audit failed: bracket entrants do not match qualified teams")

    for match_id, previous_a, previous_b in ROUND_OF_16:
        row = _play_knockout_match(simulator, match_id, "LAST_16", winners[previous_a], winners[previous_b])
        bracket[str(match_id)] = row
        winners[match_id] = row["winner"]

    for match_id, previous_a, previous_b in QUARTER_FINALS:
        row = _play_knockout_match(
            simulator,
            match_id,
            "QUARTER_FINALS",
            winners[previous_a],
            winners[previous_b],
        )
        bracket[str(match_id)] = row
        winners[match_id] = row["winner"]

    semifinal_losers: list[str] = []
    for match_id, previous_a, previous_b in SEMI_FINALS:
        team_a = winners[previous_a]
        team_b = winners[previous_b]
        row = _play_knockout_match(simulator, match_id, "SEMI_FINALS", team_a, team_b)
        bracket[str(match_id)] = row
        winners[match_id] = row["winner"]
        semifinal_losers.append(team_b if row["winner"] == team_a else team_a)

    third_row = _play_knockout_match(
        simulator,
        104,
        "THIRD_PLACE",
        semifinal_losers[0],
        semifinal_losers[1],
    )
    bracket["104"] = third_row
    final_row = _play_knockout_match(simulator, 103, "FINAL", winners[101], winners[102])
    bracket["103"] = final_row

    summary: dict[str, Any] = {}
    if simulation_summary and simulation_summary.exists():
        source_summary = json.loads(simulation_summary.read_text(encoding="utf-8"))
        summary = {
            "source_output": str(simulation_summary.resolve().relative_to(ROOT)),
            "runs_completed": source_summary.get("runs_completed"),
            "top_champions": source_summary.get("champions", [])[:8],
            "top_finalists": source_summary.get("finalists", [])[:8],
            "top_semifinalists": source_summary.get("semifinalists", [])[:8],
        }

    return {
        **metadata,
        **summary,
        "graphic_type": "coherent_deterministic_premundial_bracket",
        "winner": final_row["winner"],
        "winner_advance_probability": final_row["winner_advance_probability"],
        "group_standings": {
            group: [asdict(row) for row in ranking]
            for group, ranking in sorted(standings.items())
        },
        "best_thirds": [asdict(row) for row in best_thirds],
        "third_place_match_assignments": third_assignments,
        "group_matches": group_matches,
        "bracket": bracket,
    }


def render_bracket(payload: dict[str, Any], output_png: Path) -> None:
    bracket = payload["bracket"]
    summary_top = payload.get("top_champions") or []
    champion = bracket["103"]["winner"]
    champion_probability = bracket["103"]["winner_advance_probability"]
    runs = payload.get("runs_completed")

    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(20, 11.25), dpi=190)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#050606")
    ax.set_facecolor("#050606")

    for color, x0, x1 in [
        ("#20d7ff", 0.00, 0.22),
        ("#ff3b78", 0.22, 0.50),
        ("#ff5a2f", 0.50, 0.75),
        ("#97ff3d", 0.75, 1.00),
    ]:
        ax.add_patch(Rectangle((x0, 0), x1 - x0, 1, color=color, alpha=0.11, linewidth=0))

    rng = np.random.default_rng(42)
    ax.scatter(
        rng.random(2200),
        rng.random(2200),
        s=rng.uniform(0.08, 0.65, 2200),
        color="white",
        alpha=rng.uniform(0.018, 0.055, 2200),
        linewidths=0,
    )

    line = "#86f24b"
    box = "#8bff3c"
    leaf_box = "#a8ff64"
    text = "#071007"
    white = "#f8fff2"
    muted = "#b7c8b0"

    ax.add_patch(
        FancyBboxPatch(
            (0.018, 0.032),
            0.964,
            0.925,
            boxstyle="round,pad=0.012,rounding_size=0.035",
            facecolor="#050606",
            edgecolor="#7dff35",
            linewidth=1.5,
            alpha=0.975,
        )
    )
    ax.text(0.5, 0.915, "BRACKET PREMUNDIAL 2026", ha="center", va="center", color=white, fontsize=34, fontweight="bold")
    subtitle = (
        "Ruta coherente del modelo: grupos -> 8 mejores terceros -> dieciseisavos -> final"
    )
    ax.text(0.5, 0.875, subtitle, ha="center", va="center", color=line, fontsize=13.5, fontweight="bold")
    ax.text(
        0.5,
        0.846,
        "No usa partidos disputados del Mundial. La asignacion de terceros usa la tabla exacta de 495 combinaciones.",
        ha="center",
        va="center",
        color=muted,
        fontsize=9.5,
    )

    left_leaf_y = [0.79 - i * 0.043 for i in range(16)]
    left_r32_y = [(left_leaf_y[i] + left_leaf_y[i + 1]) / 2 for i in range(0, 16, 2)]
    left_r16_y = [(left_r32_y[i] + left_r32_y[i + 1]) / 2 for i in range(0, 8, 2)]
    left_qf_y = [(left_r16_y[i] + left_r16_y[i + 1]) / 2 for i in range(0, 4, 2)]
    sf_y = sum(left_qf_y) / 2
    final_y = 0.485
    nodes: dict[int, tuple[float, float]] = {}

    def draw_box(
        x: float,
        y: float,
        label: str,
        prob: float | None = None,
        *,
        side: str,
        winner: bool = False,
        width: float = 0.142,
    ) -> tuple[float, float]:
        height = 0.030
        x0 = x if side == "left" else x - width
        face = box if winner else leaf_box
        ax.add_patch(
            FancyBboxPatch(
                (x0, y - height / 2),
                width,
                height,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                facecolor=face,
                edgecolor=face,
                linewidth=0,
                zorder=5,
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
            fontsize=7.9,
            fontweight="bold",
            zorder=6,
        )
        return (x0 + width, y) if side == "left" else (x0, y)

    def connect(start: tuple[float, float], end: tuple[float, float]) -> None:
        x1, y1 = start
        x2, y2 = end
        mid = (x1 + x2) / 2
        ax.plot([x1, mid], [y1, y1], color=line, linewidth=1.65, alpha=0.94, zorder=2)
        ax.plot([mid, mid], [y1, y2], color=line, linewidth=1.65, alpha=0.94, zorder=2)
        ax.plot([mid, x2], [y2, y2], color=line, linewidth=1.65, alpha=0.94, zorder=2)

    def add_match(match_id: int, y: float, x: float, inputs: list[tuple[float, float]], *, side: str) -> None:
        row = bracket[str(match_id)]
        target_edge = draw_box(
            x,
            y,
            row["winner"],
            row["winner_advance_probability"],
            side=side,
            winner=True,
        )
        box_edge = (x, y) if side == "left" else (x - 0.142, y)
        for input_node in inputs:
            connect(input_node, box_edge)
        nodes[match_id] = target_edge

    for idx, match_id in enumerate(LEFT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = left_leaf_y[idx * 2], left_leaf_y[idx * 2 + 1]
        e1 = draw_box(0.045, y1, row["team_a"], None, side="left")
        e2 = draw_box(0.045, y2, row["team_b"], None, side="left")
        add_match(match_id, left_r32_y[idx], 0.205, [e1, e2], side="left")

    for idx, match_id in enumerate(RIGHT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = left_leaf_y[idx * 2], left_leaf_y[idx * 2 + 1]
        e1 = draw_box(0.955, y1, row["team_a"], None, side="right")
        e2 = draw_box(0.955, y2, row["team_b"], None, side="right")
        add_match(match_id, left_r32_y[idx], 0.795, [e1, e2], side="right")

    for idx, match_id in enumerate([89, 90, 93, 94]):
        add_match(match_id, left_r16_y[idx], 0.345, [nodes[R16_BY_MATCH[match_id][0]], nodes[R16_BY_MATCH[match_id][1]]], side="left")
    for idx, match_id in enumerate([91, 92, 95, 96]):
        add_match(match_id, left_r16_y[idx], 0.655, [nodes[R16_BY_MATCH[match_id][0]], nodes[R16_BY_MATCH[match_id][1]]], side="right")
    for idx, match_id in enumerate([97, 98]):
        add_match(match_id, left_qf_y[idx], 0.465, [nodes[QF_BY_MATCH[match_id][0]], nodes[QF_BY_MATCH[match_id][1]]], side="left")
    for idx, match_id in enumerate([99, 100]):
        add_match(match_id, left_qf_y[idx], 0.535, [nodes[QF_BY_MATCH[match_id][0]], nodes[QF_BY_MATCH[match_id][1]]], side="right")

    add_match(101, sf_y, 0.345, [nodes[97], nodes[98]], side="left")
    add_match(102, sf_y, 0.655, [nodes[99], nodes[100]], side="right")
    connect(nodes[101], (0.388, final_y))
    connect(nodes[102], (0.612, final_y))

    ax.add_patch(
        FancyBboxPatch(
            (0.388, final_y - 0.048),
            0.224,
            0.096,
            boxstyle="round,pad=0.011,rounding_size=0.022",
            facecolor="#f8fff2",
            edgecolor=line,
            linewidth=2.2,
            zorder=20,
        )
    )
    ax.text(
        0.5,
        final_y + 0.026,
        "CAMPEON EN RUTA",
        ha="center",
        va="center",
        color=text,
        fontsize=10.0,
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        0.5,
        final_y - 0.004,
        _short_name(champion),
        ha="center",
        va="center",
        color=text,
        fontsize=21,
        fontweight="bold",
        zorder=21,
    )
    ax.text(
        0.5,
        final_y - 0.031,
        f"prob. final {champion_probability:.0%}",
        ha="center",
        va="center",
        color=text,
        fontsize=9.2,
        fontweight="bold",
        zorder=21,
    )

    if summary_top:
        label = f"Top campeon en {runs:,} simulaciones" if runs else "Top campeon en simulaciones"
        ax.text(0.16, 0.083, label, ha="center", va="center", color=white, fontsize=9.1, fontweight="bold")
        for index, item in enumerate(summary_top[:5]):
            ax.text(
                0.055 + index * 0.052,
                0.057,
                f"{_short_name(item['team'])}\n{float(item['probability']):.1%}",
                ha="center",
                va="center",
                color=line if index == 0 else muted,
                fontsize=8.2,
                fontweight="bold" if index == 0 else "normal",
            )

    ax.text(
        0.5,
        0.108,
        f"Final deterministica: {_short_name(bracket['103']['team_a'])} vs {_short_name(bracket['103']['team_b'])}",
        ha="center",
        va="center",
        color=muted,
        fontsize=9,
    )
    ax.text(
        0.5,
        0.085,
        f"Tercer lugar deterministico: {_short_name(bracket['104']['winner'])}",
        ha="center",
        va="center",
        color=muted,
        fontsize=9,
    )
    ax.text(
        0.84,
        0.096,
        "Ruta unica y auditable\n32 clasificados unicos",
        ha="center",
        va="center",
        color=line,
        fontsize=9.2,
        fontweight="bold",
    )

    fig.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a coherent premundial World Cup 2026 bracket.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--simulation-summary", type=Path, default=Path("outputs/premundial_full_1000_seed42.json"))
    parser.add_argument("--output-png", type=Path, default=Path("docs/assets/worldcup_2026_premundial_bracket.png"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/assets/worldcup_2026_premundial_bracket.json"))
    args = parser.parse_args()

    payload = build_deterministic_bracket(args.data_root, args.simulation_summary)
    render_bracket(payload, args.output_png)
    payload["image_path"] = str(args.output_png.resolve().relative_to(ROOT))
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Rendered {args.output_png} | champion={payload['winner']} "
        f"{payload['winner_advance_probability']:.0%}"
    )


if __name__ == "__main__":
    main()
