from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

SIMULATION_PATHS = [
    "data/static",
    "data/processed",
    "data/manifests",
    "data/models/lightgbm_neutral_all_played_wc2026.joblib",
    "data/models/lightgbm_neutral_worldcup_holdout.joblib",
    "data/models/lightgbm_neutral_model.joblib",
    "data/models/lightgbm_neutral_metrics.json",
    "data/models/README_CURRENT_DEFAULT.md",
    "data/raw/api_football/fixtures",
    "data/raw/api_football/players",
    "data/raw/espn/worldcup_2026",
    "data/raw/football_data/competitions/WC",
    "outputs/worldcup2026_all_played_model_metadata.json",
]

OPTIONAL_OUTPUT_GLOBS = [
    "outputs/worldcup2026_consensus_bracket_*",
    "outputs/worldcup2026_last16_*",
    "outputs/worldcup2026_r32_*",
]

EXCLUDED_NAMES = {
    ".DS_Store",
    "Thumbs.db",
}


def _iter_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    files: list[Path] = []
    for child in path.rglob("*"):
        if not child.is_file():
            continue
        if child.name in EXCLUDED_NAMES:
            continue
        if "__pycache__" in child.parts:
            continue
        files.append(child)
    return files


def _simulation_files(include_outputs: bool) -> list[Path]:
    files: list[Path] = []
    for relative in SIMULATION_PATHS:
        files.extend(_iter_files(ROOT / relative))
    if include_outputs:
        for pattern in OPTIONAL_OUTPUT_GLOBS:
            for path in ROOT.glob(pattern):
                files.extend(_iter_files(path))
    return files


def _full_cache_files(include_outputs: bool) -> list[Path]:
    files = _iter_files(ROOT / "data")
    if include_outputs:
        files.extend(_iter_files(ROOT / "outputs"))
    return files


def _dedupe(paths: list[Path]) -> list[Path]:
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique, key=lambda item: item.relative_to(ROOT).as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(
    *,
    mode: str,
    include_outputs: bool,
    output: Path,
    files: list[Path],
    hashes: bool,
) -> dict[str, Any]:
    file_rows = []
    total_bytes = 0
    for path in files:
        size = path.stat().st_size
        total_bytes += size
        row: dict[str, Any] = {
            "path": path.relative_to(ROOT).as_posix(),
            "bytes": size,
        }
        if hashes:
            row["sha256"] = _sha256(path)
        file_rows.append(row)
    return {
        "bundle_type": "kinela-worldcup2026-portable-data",
        "mode": mode,
        "include_outputs": include_outputs,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "output": str(output),
        "file_count": len(file_rows),
        "total_bytes": total_bytes,
        "files": file_rows,
    }


def _default_output(mode: str) -> Path:
    date_label = datetime.now().date().isoformat()
    return ROOT / "outputs" / "portable" / f"kinela_worldcup2026_{mode}_bundle_{date_label}.zip"


def build_bundle(
    *,
    mode: str,
    output: Path,
    include_outputs: bool,
    hashes: bool,
    dry_run: bool,
) -> dict[str, Any]:
    files = (
        _simulation_files(include_outputs)
        if mode == "simulation"
        else _full_cache_files(include_outputs)
    )
    files = _dedupe(files)
    missing = [
        relative
        for relative in SIMULATION_PATHS
        if mode == "simulation" and not (ROOT / relative).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required portable inputs:\n"
            + "\n".join(f"- {item}" for item in missing)
        )
    manifest = _manifest(
        mode=mode,
        include_outputs=include_outputs,
        output=output,
        files=files,
        hashes=hashes,
    )
    if dry_run:
        return manifest

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        archive.writestr(
            "PORTABLE_BUNDLE_MANIFEST.json",
            json.dumps(manifest, indent=2, ensure_ascii=False),
        )
        for path in files:
            archive.write(path, path.relative_to(ROOT).as_posix())
    manifest["zip_bytes"] = output.stat().st_size
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulation", "full-cache"], default="simulation")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include-outputs", action="store_true")
    parser.add_argument("--hashes", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output = args.output or _default_output(args.mode)
    if not output.is_absolute():
        output = ROOT / output
    manifest = build_bundle(
        mode=args.mode,
        output=output,
        include_outputs=args.include_outputs,
        hashes=args.hashes,
        dry_run=args.dry_run,
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
