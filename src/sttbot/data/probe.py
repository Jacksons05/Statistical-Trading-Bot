"""Re-measure which direct-egress data hosts are reachable from this machine.

Reachability is environment-specific: a sandbox with an allowlisting proxy
returns 403 for every host regardless of upstream status, which is easy to
mistake for the venue blocking us. Run this to get a current answer::

    python -m sttbot.data.probe

Nothing here is imported by library code, and no test invokes it -- the test
suite stays hermetic.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass

from .datasets import BLOCKED_SOURCES, LIVE_ENDPOINTS, user_agent


@dataclass(frozen=True)
class ProbeResult:
    name: str
    host: str
    status: int | None
    detail: str

    @property
    def reachable(self) -> bool:
        # 2xx and 3xx both prove egress reached the origin. 401/403 are
        # ambiguous -- the host answered, but we were refused.
        return self.status is not None and self.status < 400


def probe(url: str, *, timeout: float = 10.0) -> tuple[int | None, str]:
    """Return ``(status_code, detail)`` for a HEAD-ish GET of ``url``."""
    request = urllib.request.Request(url, headers={"User-Agent": user_agent()})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, "ok"
    except urllib.error.HTTPError as exc:
        return exc.code, exc.reason or "http error"
    except urllib.error.URLError as exc:
        return None, f"unreachable: {exc.reason}"
    except OSError as exc:  # pragma: no cover - network-dependent
        return None, f"unreachable: {exc}"


def probe_all(timeout: float = 10.0) -> list[ProbeResult]:
    results = []
    for name, endpoint in LIVE_ENDPOINTS.items():
        status, detail = probe(endpoint.probe_url, timeout=timeout)
        results.append(ProbeResult(name, endpoint.host, status, detail))
    return results


def main() -> None:  # pragma: no cover - manual/network tool
    print("Live endpoints (expected reachable):")
    for result in probe_all():
        mark = "ok " if result.reachable else "FAIL"
        status = result.status if result.status is not None else "---"
        print(f"  [{mark}] {status:>4}  {result.name:<18} {result.host}")
        if not result.reachable:
            print(f"          {result.detail}")
            print(f"          {LIVE_ENDPOINTS[result.name].notes}")

    print("\nKnown-blocked (expected to fail):")
    for host, reason in BLOCKED_SOURCES.items():
        status, detail = probe(f"https://{host}")
        print(f"  [{status or '---'}] {host}  {detail}")
        print(f"          {reason}")


if __name__ == "__main__":  # pragma: no cover
    main()
