"""
Launch the Model Health page on this machine only.

The server binds to 127.0.0.1 (Streamlit otherwise listens on all
interfaces) and answers only to the host names 127.0.0.1 and localhost, so
a web page whose name is rebound to this machine cannot read it. Usage
statistics are off. The page reads two summaries:

    --ci-summary     a summary downloaded from the daily health workflow;
                     default: the newest reports/health/ci/*/health_summary.json
    --local-summary  the local monitor's summary;
                     default: ~/stock-agent-private/health/latest.json

A summary that does not exist is shown as "no summary yet" with the command
that creates it. The page is read-only and computes nothing.

Usage:
    .venv/bin/python scripts/run_model_health.py [--port 8503]
        [--ci-summary FILE] [--local-summary FILE]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "src" / "stock_agent" / "health" / "app.py"
CI_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_CI_SUMMARY"
LOCAL_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_LOCAL_SUMMARY"
CI_DOWNLOADS = PROJECT_ROOT / "reports" / "health" / "ci"
DEFAULT_LOCAL = Path.home() / "stock-agent-private" / "health" / "latest.json"


def newest_ci_summary(downloads: Path = CI_DOWNLOADS) -> Path | None:
    """The most recently modified downloaded CI summary, if any."""

    found = [path for path in downloads.glob("*/health_summary.json") if path.is_file()]
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
        # DNS-rebinding protection: Streamlit accepts the app's WebSocket under
        # any Host header unless this list names the ones the server answers to.
        "--server.allowedHosts",
        "127.0.0.1",
        "--server.allowedHosts",
        "localhost",
        "--server.port",
        str(port),
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "viewer",
        "--theme.base",
        "light",
    ]


def environment(ci_summary: Path | None, local_summary: Path | None) -> dict[str, str]:
    values = dict(os.environ)
    for variable, path in (
        (CI_SUMMARY_VARIABLE, ci_summary),
        (LOCAL_SUMMARY_VARIABLE, local_summary),
    ):
        values.pop(variable, None)
        if path is not None:
            values[variable] = str(path.expanduser().resolve())
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", type=int, default=8503)
    parser.add_argument("--ci-summary", type=Path, default=None)
    parser.add_argument("--local-summary", type=Path, default=DEFAULT_LOCAL)
    args = parser.parse_args(argv)
    ci_summary = args.ci_summary or newest_ci_summary()
    return subprocess.call(
        command(args.port), cwd=PROJECT_ROOT, env=environment(ci_summary, args.local_summary)
    )


if __name__ == "__main__":
    raise SystemExit(main())
