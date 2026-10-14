"""
Launch the Portfolio Lab (the product app) on this machine only.

The server binds to 127.0.0.1 (Streamlit otherwise listens on all
interfaces) and usage statistics are off. The lab runs in synthetic demo
mode: every fund is synthetic demo data, not real fund performance, and every
metric is unavailable until the descriptive formulas (OD8 group A) are
approved and implemented.

Usage:
    .venv/bin/python scripts/run_portfolio_lab.py [--port 8502]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP = PROJECT_ROOT / "src" / "stock_agent" / "app" / "run.py"


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
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args(argv)
    return subprocess.call(command(args.port), cwd=PROJECT_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
