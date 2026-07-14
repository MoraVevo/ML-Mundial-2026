from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ("results.csv", "shootouts.csv", "former_names.csv", "LICENSE")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def collect(output_dir: Path, *, refresh: bool = False) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files: list[dict[str, object]] = []
    for name in FILES:
        path = output_dir / name
        url = f"{BASE_URL}/{name}"
        if path.exists() and not refresh:
            content = path.read_bytes()
            status = "cached"
        else:
            request = Request(url, headers={"User-Agent": "kinela-penalty-model/1.0"})
            with urlopen(request, timeout=60) as response:  # noqa: S310
                content = response.read()
            path.write_bytes(content)
            status = "downloaded"
        files.append(
            {
                "name": name,
                "path": str(path),
                "url": url,
                "status": status,
                "bytes": len(content),
                "sha256": _sha256(content),
            }
        )
    metadata = {
        "source": "martj42/international_results",
        "license": "CC0-1.0",
        "collected_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": files,
    }
    metadata_path = output_dir / "collection_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {**metadata, "metadata_path": str(metadata_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache free international shootout data.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/open_international_results"),
    )
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(collect(args.output_dir, refresh=args.refresh), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
