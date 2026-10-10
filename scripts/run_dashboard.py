"""
Launch the read-only diagnostics dashboard on this machine only.

The server binds to 127.0.0.1 (Streamlit otherwise listens on all
interfaces) and usage statistics are off. The dashboard reads saved runs
under --root (default reports/), which must contain runs/<run_id>/.

Usage:
    .venv/bin/python scripts/run_dashboard.py [--root reports] [--port 8501]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "src" / "stock_agent" / "dashboard" / "app.py"
ROOT_ENVIRONMENT_VARIABLE = "STOCK_AGENT_DASHBOARD_ROOT"


def command(port: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP),
        "--server.address",
        "127.0.0.1",
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--root", default=str(PROJECT_ROOT / "reports"))
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args(argv)
    environment = {**os.environ, ROOT_ENVIRONMENT_VARIABLE: str(Path(args.root).resolve())}
    return subprocess.call(command(args.port), cwd=PROJECT_ROOT, env=environment)


if __name__ == "__main__":
    raise SystemExit(main())
