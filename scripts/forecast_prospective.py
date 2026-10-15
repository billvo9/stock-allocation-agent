"""
The prospective forecasting job for VTI, AGG and SHV (docs/research/ml_prospective.md).

Subcommands, run from the repository root:

    freeze     develop, refit, write and register the frozen forecasters (once,
               before the first forecast; a later run is a new, counted variant)
        .venv/bin/python scripts/forecast_prospective.py freeze --research-store <F1 store>
    forecast   record one forecast as of the last complete session
        .venv/bin/python scripts/forecast_prospective.py forecast \
            --bundle reports/ml/prospective/frozen/<bundle_id> \
            --store-root <prospective store> --forecast-close YYYY-MM-DD
    score      score every matured, unscored forecast row
        .venv/bin/python scripts/forecast_prospective.py score \
            --store-root <prospective store> --score-close YYYY-MM-DD
    report     cumulative band coverage and errors, with small-sample caveats
        .venv/bin/python scripts/forecast_prospective.py report

Training reads the F1 research store only (sessions up to 2024-12-31); the
prospective store is read only through the real-etf-data loader. The ledger
(default data/raw/prospective_forecasts/ledger.jsonl) and the outputs
(default reports/ml/prospective) are private and ignored by Git. Each
forecast is labelled "unvalidated, no demonstrated skill": these are research
records, never recommendations. Nothing here contacts a network.
"""

from __future__ import annotations

import argparse
import importlib
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from stock_agent.forecasting.prospective import forecast, freeze, frozen, ledger
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.forecasting.provenance import code_provenance, environment
from stock_agent.registry import open_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "prospective" / "etf_prospective_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / freeze.DEFAULT_OUTPUT
DEFAULT_LEDGER = PROJECT_ROOT / ledger.DEFAULT_LEDGER
# The prospective-store loader and the NYSE calendar beyond 2024, both built
# by the real-etf-data ticket. Update these lines if their final names differ.
# The calendar is the independent session schedule every window is checked
# against (nyse.sessions refuses dates after 2024 until it is extended).
LOADER = ("stock_agent.data.prospective", "load_inference_window")
CALENDAR = ("stock_agent.calendars.nyse", "sessions")


def _resolve(target: tuple[str, str], what: str):
    module, name = target
    try:
        return getattr(importlib.import_module(module), name)
    except (ModuleNotFoundError, AttributeError):
        raise SystemExit(
            f"The {what} {module}.{name} is not available on this branch. It comes with the "
            "real-etf-data ticket; nothing was read or written."
        ) from None


def _loader():
    return _resolve(LOADER, "prospective-store loader")


def _calendar():
    return _resolve(CALENDAR, "exchange calendar")


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def _freeze(args) -> int:
    config = load_config(args.config)
    provenance = {"code": code_provenance(PROJECT_ROOT), "environment": environment()}
    result = freeze.freeze(
        config,
        research_store_root=args.research_store,
        output_root=args.output_root,
        provenance=provenance,
        started_at=datetime.now(UTC),
    )
    registry = open_registry(args.output_root / freeze.REGISTRY_DIRECTORY, args.output_root)
    print(f"config {config.experiment} sha256 {config.sha256}")
    print(f"run {result.run_id}; checks {result.record['checks_summary']['families']}")
    print(f"bundle {result.bundle.path} (manifest {result.bundle.manifest_sha256})")
    for entry in result.entries:
        print(f"registered {entry.ref}: stage {registry.stage(entry.ref)}; {frozen.STATUS}")
    return 0


def _forecast(args) -> int:
    output_root = args.bundle.resolve().parents[1]
    registry = open_registry(output_root / freeze.REGISTRY_DIRECTORY, output_root)
    models = forecast.load_models(args.bundle, registry)
    entry = forecast.record_forecast(
        models,
        loader=_loader(),
        store_root=args.store_root,
        forecast_close=pd.Timestamp(args.forecast_close, tz="UTC"),
        ledger_path=args.ledger,
        created_at=_now(),
        code=code_provenance(PROJECT_ROOT),
        calendar=_calendar(),
    )
    rows = pd.DataFrame(entry.payload["rows"])
    columns = ["asset", "horizon", "target", "point", "band_low", "band_high", "null", "stage"]
    print(f"forecast {entry.payload['forecast_id']} as of {entry.payload['as_of_session']}")
    print(", ".join(entry.payload["labels"]))
    print(rows.loc[:, [c for c in columns if c in rows]].to_string(index=False))
    print(f"ledger head {entry.entry_sha256}: note it outside the ledger to detect truncation")
    return 0


def _score(args) -> int:
    entry, left = forecast.score_matured(
        ledger_path=args.ledger,
        loader=_loader(),
        store_root=args.store_root,
        score_close=pd.Timestamp(args.score_close, tz="UTC"),
        scored_at=_now(),
        code=code_provenance(PROJECT_ROOT),
        calendar=_calendar(),
    )
    print(f"scored {0 if entry is None else len(entry.payload['scores'])} rows")
    if left:
        print(pd.DataFrame(left)["reason"].value_counts().to_string())
    if entry is not None:
        print(f"ledger head {entry.entry_sha256}: note it outside the ledger to detect truncation")
    return _report(args)


def _report(args) -> int:
    entries = ledger.read_ledger(args.ledger)
    if getattr(args, "expect_head", None):
        ledger.check_head(entries, args.expect_head)
        print(f"kept head {args.expect_head[:12]} is in the ledger")
    report = forecast.coverage_report(entries)
    print(f"{sum(e.kind == 'forecast' for e in entries)} forecasts; head {ledger.head(entries)}")
    pd.set_option("display.width", 200)
    print(report.drop(columns="caveat").to_string(index=False) if len(report) else "no scores yet")
    for text in sorted(set(report["caveat"])):
        print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freezing.add_argument("--research-store", type=Path, required=True)
    freezing.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    record = commands.add_parser("forecast")
    record.add_argument("--bundle", type=Path, required=True)
    record.add_argument("--store-root", type=Path, required=True)
    record.add_argument("--forecast-close", required=True)
    scoring = commands.add_parser("score")
    scoring.add_argument("--store-root", type=Path, required=True)
    scoring.add_argument("--score-close", required=True)
    reporting = commands.add_parser("report")
    reporting.add_argument(
        "--expect-head", help="a head hash kept outside the ledger; refused if it is missing"
    )
    args = parser.parse_args(argv)
    handlers = {"freeze": _freeze, "forecast": _forecast, "score": _score, "report": _report}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
