from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_NAME = "PORTABLE_BUNDLE_MANIFEST.json"


def _safe_target(root: Path, member: str) -> Path:
    if member.endswith("/"):
        return root / member
    target = (root / member).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"Unsafe archive member outside target root: {member}") from exc
    return target


def restore_bundle(bundle: Path, target_root: Path, *, overwrite: bool) -> dict[str, object]:
    if not bundle.exists():
        raise FileNotFoundError(f"Bundle not found: {bundle}")
    target_root.mkdir(parents=True, exist_ok=True)
    target_root = target_root.resolve()

    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        if MANIFEST_NAME not in names:
            raise ValueError(f"Missing {MANIFEST_NAME}; this does not look like a Kinela bundle.")
        manifest = json.loads(archive.read(MANIFEST_NAME).decode("utf-8"))
        collisions: list[str] = []
        for name in names:
            if name == MANIFEST_NAME or name.endswith("/"):
                continue
            target = _safe_target(target_root, name)
            if target.exists() and not overwrite:
                collisions.append(name)
        if collisions:
            preview = "\n".join(f"- {item}" for item in collisions[:20])
            extra = "" if len(collisions) <= 20 else f"\n... and {len(collisions) - 20} more"
            raise FileExistsError(
                "Bundle would overwrite existing files. Re-run with --overwrite.\n"
                f"{preview}{extra}"
            )
        for name in names:
            if name == MANIFEST_NAME or name.endswith("/"):
                continue
            target = _safe_target(target_root, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(name) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination)
        manifest_path = target_root / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {
        "bundle": str(bundle),
        "target_root": str(target_root),
        "manifest_path": str(manifest_path),
        "mode": manifest.get("mode"),
        "file_count": manifest.get("file_count"),
        "total_bytes": manifest.get("total_bytes"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, default=ROOT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    result = restore_bundle(args.bundle, args.target_root, overwrite=args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
