from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from kinela.providers.statsbomb import StatsBombOpenData  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache open knockout events for the extra-time model."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    result = StatsBombOpenData(args.data_root).collect_men_extra_time_tournaments(
        refresh=args.refresh,
        workers=max(1, args.workers),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
