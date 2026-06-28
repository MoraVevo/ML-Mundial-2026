from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

LEFT_R32_MATCHES = [74, 77, 73, 75, 83, 84, 81, 82]
RIGHT_R32_MATCHES = [76, 78, 79, 80, 86, 88, 85, 87]
R16_BY_MATCH = {
    89: (74, 77),
    90: (73, 75),
    91: (76, 78),
    92: (79, 80),
    93: (83, 84),
    94: (81, 82),
    95: (86, 88),
    96: (85, 87),
}
QF_BY_MATCH = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}


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


def _top_entry(table: dict[str, list[dict[str, Any]]], match_id: int, index: int = 0) -> dict[str, Any]:
    rows = table[str(match_id)]
    row = rows[index] if index < len(rows) else {"team": "TBD", "count": 0, "probability": 0.0}
    return {"team": row["team"], "count": int(row["count"]), "probability": float(row["probability"])}


def _build_bracket(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    teams = data["slot_team_counts"]
    winners = data["slot_winner_counts"]
    return {
        str(match_id): {
            "match_id": match_id,
            "team_a": _top_entry(teams, match_id, 0),
            "team_b": _top_entry(teams, match_id, 1),
            "winner": _top_entry(winners, match_id, 0),
        }
        for match_id in range(73, 105)
    }


def render_consensus_bracket(source: Path, output_png: Path, output_json: Path) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    data = json.loads(source.read_text(encoding="utf-8"))
    bracket = _build_bracket(data)
    runs = int(data["runs_completed"])
    champion = bracket["103"]["winner"]
    runner = data["runners_up"][0]
    third = data["third_place"][0]
    top_champions = data["champions"][:8]

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
    ax.text(
        0.5,
        0.875,
        f"{runs:,} simulaciones completas | sin partidos disputados del Mundial | campeon consenso: "
        f"{_short_name(champion['team'])} ({champion['probability']:.1%})",
        ha="center",
        va="center",
        color=line,
        fontsize=13.5,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.846,
        "Cada porcentaje es frecuencia dentro de la simulacion completa: aparicion en cruce o victoria del slot.",
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
        target_edge = draw_box(x, y, row["winner"]["team"], row["winner"]["probability"], side=side, winner=True)
        box_edge = (x, y) if side == "left" else (x - 0.142, y)
        for input_node in inputs:
            connect(input_node, box_edge)
        nodes[match_id] = target_edge

    for idx, match_id in enumerate(LEFT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = left_leaf_y[idx * 2], left_leaf_y[idx * 2 + 1]
        e1 = draw_box(0.045, y1, row["team_a"]["team"], row["team_a"]["probability"], side="left")
        e2 = draw_box(0.045, y2, row["team_b"]["team"], row["team_b"]["probability"], side="left")
        add_match(match_id, left_r32_y[idx], 0.205, [e1, e2], side="left")

    for idx, match_id in enumerate(RIGHT_R32_MATCHES):
        row = bracket[str(match_id)]
        y1, y2 = left_leaf_y[idx * 2], left_leaf_y[idx * 2 + 1]
        e1 = draw_box(0.955, y1, row["team_a"]["team"], row["team_a"]["probability"], side="right")
        e2 = draw_box(0.955, y2, row["team_b"]["team"], row["team_b"]["probability"], side="right")
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
    ax.text(0.5, final_y + 0.020, "CAMPEON CONSENSO", ha="center", va="center", color=text, fontsize=10.5, fontweight="bold", zorder=21)
    ax.text(
        0.5,
        final_y - 0.012,
        f"{_short_name(champion['team'])} {champion['probability']:.1%}",
        ha="center",
        va="center",
        color=text,
        fontsize=21,
        fontweight="bold",
        zorder=21,
    )

    ax.text(0.15, 0.130, "Top campeon", ha="center", va="center", color=white, fontsize=10, fontweight="bold")
    for index, item in enumerate(top_champions[:5]):
        ax.text(
            0.055 + index * 0.047,
            0.101,
            f"{_short_name(item['team'])}\n{item['probability']:.1%}",
            ha="center",
            va="center",
            color=line if index == 0 else muted,
            fontsize=8.2,
            fontweight="bold" if index == 0 else "normal",
        )
    ax.text(0.5, 0.118, f"Finalista mas frecuente: {_short_name(data['finalists'][0]['team'])} ({data['finalists'][0]['probability']:.1%})", ha="center", va="center", color=muted, fontsize=9)
    ax.text(
        0.5,
        0.094,
        f"Subcampeon mas frecuente: {_short_name(runner['team'])} ({runner['probability']:.1%}) | "
        f"Tercer lugar mas frecuente: {_short_name(third['team'])} ({third['probability']:.1%})",
        ha="center",
        va="center",
        color=muted,
        fontsize=9,
    )
    ax.text(
        0.84,
        0.108,
        "Contexto premundial congelado\n0 resultados del Mundial usados",
        ha="center",
        va="center",
        color=line,
        fontsize=9.2,
        fontweight="bold",
    )

    fig.savefig(output_png, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    payload = {
        "source_output": str(source.resolve().relative_to(project_root)),
        "runs_completed": runs,
        "seed": data["seed"],
        "manual_results_used": data["manual_results_used"],
        "excluded_worldcup_2026_history_rows": data["excluded_worldcup_2026_history_rows"],
        "simulation_mode": data["simulation_mode"],
        "graphic_type": "full_simulation_consensus_bracket",
        "note": "Top two entrants and slot winner are frequency leaders from the complete premundial simulation output.",
        "champion_consensus": champion,
        "top_champions": top_champions,
        "top_finalists": data["finalists"][:8],
        "top_semifinalists": data["semifinalists"][:8],
        "bracket": bracket,
        "image_path": str(output_png.resolve().relative_to(project_root)),
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a consensus bracket from the full premundial simulation output.")
    parser.add_argument("--source", type=Path, default=Path("outputs/premundial_full_1000_seed42.json"))
    parser.add_argument("--output-png", type=Path, default=Path("docs/assets/worldcup_2026_premundial_bracket.png"))
    parser.add_argument("--output-json", type=Path, default=Path("docs/assets/worldcup_2026_premundial_bracket.json"))
    args = parser.parse_args()
    payload = render_consensus_bracket(args.source, args.output_png, args.output_json)
    champion = payload["champion_consensus"]
    print(f"Rendered {args.output_png} | champion={champion['team']} {champion['probability']:.1%}")


if __name__ == "__main__":
    main()
