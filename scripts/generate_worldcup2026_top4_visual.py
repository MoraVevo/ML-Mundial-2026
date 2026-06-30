from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

POSITION_FIELDS = [
    ("prob_campeon", "Campeon"),
    ("prob_segundo", "Segundo lugar"),
    ("prob_tercero", "Tercer lugar"),
    ("prob_cuarto", "Cuarto lugar"),
]


def _top_by_position(payload: dict[str, Any], limit: int) -> dict[str, list[dict[str, Any]]]:
    teams = payload["team_summary"]
    result: dict[str, list[dict[str, Any]]] = {}
    for field, _label in POSITION_FIELDS:
        result[field] = [
            {
                "team": row["equipo"],
                "probability": float(row[field]),
                "count": int(row[field.replace("prob_", "veces_")]),
            }
            for row in sorted(teams, key=lambda item: float(item[field]), reverse=True)[:limit]
        ]
    return result


def _render_panel(ax: plt.Axes, rows: list[dict[str, Any]], title: str, color: str) -> None:
    names = [row["team"] for row in rows][::-1]
    values = [100.0 * float(row["probability"]) for row in rows][::-1]
    bars = ax.barh(names, values, color=color, edgecolor="#102016", linewidth=0.8)
    ax.set_title(title, color="#f4fff2", fontsize=16, fontweight="bold", pad=10)
    ax.set_xlim(0, max(values) * 1.22 if values else 1)
    ax.tick_params(axis="x", colors="#d8ead1", labelsize=9)
    ax.tick_params(axis="y", colors="#f4fff2", labelsize=10)
    ax.grid(axis="x", color="#2b402d", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#79f14f")
        spine.set_linewidth(0.9)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            bar.get_width() + max(values) * 0.025,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.1f}%",
            va="center",
            ha="left",
            color="#f4fff2",
            fontsize=10,
            fontweight="bold",
        )


def render_top4_visual(
    payload: dict[str, Any],
    output: Path,
    *,
    title: str,
    subtitle: str,
    limit: int,
) -> dict[str, list[dict[str, Any]]]:
    top_by_position = _top_by_position(payload, limit)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10), facecolor="#07110b")
    colors = ["#7CFF36", "#36D872", "#2FB8A0", "#A6FF63"]
    for ax, (field, label), color in zip(axes.flat, POSITION_FIELDS, colors, strict=True):
        ax.set_facecolor("#0b1610")
        _render_panel(ax, top_by_position[field], label, color)

    fig.suptitle(title, x=0.5, y=0.98, color="#f7fff2", fontsize=28, fontweight="bold")
    fig.text(0.5, 0.935, subtitle, ha="center", color="#7CFF36", fontsize=13, fontweight="bold")
    fig.text(
        0.5,
        0.035,
        "Modelo neutral LightGBM entrenado sin partidos del Mundial 2026; resultados jugados usados solo como estado del torneo.",
        ha="center",
        color="#cfe8c5",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.03, 0.06, 0.97, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return top_by_position


def main() -> None:
    parser = argparse.ArgumentParser(description="Render World Cup 2026 top-four probability panels.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/worldcup_2026_top4_probabilities.png"))
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("docs/assets/worldcup_2026_top4_probabilities.json"),
    )
    parser.add_argument("--limit", type=int, default=12)
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    runs = int(metadata["runs"])
    mode_label = (
        "FULL CONTEXT"
        if metadata.get("simulation_mode") == "full_context"
        else "FAST"
    )
    top_by_position = render_top4_visual(
        payload,
        args.output,
        title="Mundial 2026: probabilidades Top 4",
        subtitle=(
            f"{runs:,} simulaciones | {mode_label} | holdout out-of-sample | "
            "terceros exactos | penales"
        ),
        limit=args.limit,
    )
    summary = {
        "source": str(args.input),
        "runs": runs,
        "seed": metadata["seed"],
        "model_label": metadata.get("model_label", ""),
        "model_path": metadata.get("model_path", ""),
        "model_policy": "worldcup_2026_out_of_sample_holdout",
        "simulation_mode": metadata.get("simulation_mode", ""),
        "third_place_assignment": metadata.get("third_place_assignment", ""),
        "top_by_position": top_by_position,
    }
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"image": str(args.output), "summary": str(args.summary_output)}, indent=2))


if __name__ == "__main__":
    main()
