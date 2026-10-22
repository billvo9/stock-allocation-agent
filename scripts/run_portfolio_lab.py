"""
Launch the Portfolio Lab (the product app) on this machine only.

The server binds to 127.0.0.1 (Streamlit otherwise listens on all
interfaces) and answers only to the host names 127.0.0.1 and localhost, so
a web page whose name is rebound to this machine cannot read it. Usage
statistics are off. The lab runs in synthetic demo mode: every fund is
synthetic demo data, not real fund performance, and the metrics are the
approved descriptive calculations (OD8 group A).

The Lab's editorial theme (light and dark variants), its bundled fonts and
static file serving come from the script-level config next to the app
(src/stock_agent/app/.streamlit/config.toml); no flag overrides it, so the
Settings menu offers System, Light and Dark, and System follows the
operating system. This checkout's `src` is put first on PYTHONPATH, so the
app always runs this checkout's code.

The Model Health page reads two summaries, passed in environment variables
(the page itself makes no network call and runs no check):

    --health-summary     the local monitor's summary;
                         default: ~/stock-agent-private/health/latest.json
    --ci-health-summary  a summary downloaded from the daily health workflow;
                         default: the newest reports/health/ci/*/health_summary.json

A summary that does not exist is shown as "not yet collected", with the
command that creates it.

Usage:
    .venv/bin/python scripts/run_portfolio_lab.py [--port 8502]
        [--health-summary FILE] [--ci-health-summary FILE]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "src"
APP = SOURCE / "stock_agent" / "app" / "run.py"
# The variables `stock_agent.app.health_summary` reads.
LOCAL_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_LOCAL_SUMMARY"
CI_SUMMARY_VARIABLE = "STOCK_AGENT_HEALTH_CI_SUMMARY"
DEFAULT_LOCAL_SUMMARY = Path.home() / "stock-agent-private" / "health" / "latest.json"
CI_DOWNLOADS = PROJECT_ROOT / "reports" / "health" / "ci"


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
    ]


def newest_ci_summary(downloads: Path = CI_DOWNLOADS) -> Path | None:
    """The most recently modified downloaded CI summary, if any."""

    found = [path for path in downloads.glob("*/health_summary.json") if path.is_file()]
    return max(found, key=lambda path: path.stat().st_mtime) if found else None


def environment(
    health_summary: Path | None = None, ci_health_summary: Path | None = None
) -> dict[str, str]:
    """
    The current environment with this checkout's `src` first on PYTHONPATH and
    the Model Health summary paths set (or removed when there is none).
    """

    paths = [str(SOURCE), *filter(None, os.environ.get("PYTHONPATH", "").split(os.pathsep))]
    values = {**os.environ, "PYTHONPATH": os.pathsep.join(dict.fromkeys(paths))}
    for variable, path in (
        (LOCAL_SUMMARY_VARIABLE, health_summary),
        (CI_SUMMARY_VARIABLE, ci_health_summary),
    ):
        values.pop(variable, None)
        if path is not None:
            values[variable] = str(path.expanduser().resolve())
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--port", type=int, default=8502)
    parser.add_argument("--health-summary", type=Path, default=DEFAULT_LOCAL_SUMMARY)
    parser.add_argument("--ci-health-summary", type=Path, default=None)
    args = parser.parse_args(argv)
    env = environment(args.health_summary, args.ci_health_summary or newest_ci_summary())
    return subprocess.call(command(args.port), cwd=PROJECT_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
