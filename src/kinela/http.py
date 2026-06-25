from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


class CachedJsonClient:
    def __init__(
        self,
        base_url: str,
        cache_dir: Path,
        headers: dict[str, str] | None = None,
        min_interval_seconds: float = 0.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.cache_dir = cache_dir
        self.headers = headers or {}
        self.min_interval_seconds = min_interval_seconds
        self._last_request_at = 0.0

    def get(
        self,
        endpoint: str,
        params: dict[str, Any] | None = None,
        *,
        cache_name: str | None = None,
        refresh: bool = False,
    ) -> Any:
        query = urllib.parse.urlencode(
            sorted((key, value) for key, value in (params or {}).items() if value is not None)
        )
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        if query:
            url = f"{url}?{query}"

        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
        filename = cache_name or f"{endpoint.strip('/').replace('/', '__')}-{digest}.json"
        cache_path = self.cache_dir / filename
        if cache_path.exists() and not refresh:
            return json.loads(cache_path.read_text(encoding="utf-8"))

        wait = self.min_interval_seconds - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

        request = urllib.request.Request(
            url,
            headers={"User-Agent": "kinela/0.1", **self.headers},
        )
        for attempt in range(4):
            try:
                with urllib.request.urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if exc.code == 429 and attempt < 3:
                    retry_after = float(exc.headers.get("Retry-After", 15 * (attempt + 1)))
                    time.sleep(retry_after)
                    continue
                raise RuntimeError(
                    f"HTTP {exc.code} requesting {url}: {body[:500]}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < 3:
                    time.sleep(10 * (attempt + 1))
                    continue
                raise RuntimeError(f"Network error requesting {url}: {exc}") from exc

        self._last_request_at = time.monotonic()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(cache_path)
        return payload
