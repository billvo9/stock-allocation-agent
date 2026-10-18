"""
The prospective forecasting job for VTI, AGG and SHV (docs/research/ml_prospective.md).

Subcommands, run from the repository root:

    freeze     develop, refit, write and register the frozen forecasters (once,
               before the first forecast; a later run is a new, counted variant)
        .venv/bin/python scripts/forecast_prospective.py freeze --research-store <F1 store>
    registration-record
               write the registration record file a registration pull request adds
               (docs/research/registrations/<bundle_id>.json; nothing is registered)
        .venv/bin/python scripts/forecast_prospective.py registration-record \
            --bundle reports/ml/prospective/frozen/<bundle_id> \
            --registration-doc-commit <commit of docs/research/prospective_registration.md>
    register   append the registration entry, the first entry of a new ledger, after
               the owner merged the registration pull request
        .venv/bin/python scripts/forecast_prospective.py register \
            --bundle reports/ml/prospective/frozen/<bundle_id> \
            --record docs/research/registrations/<bundle_id>.json \
            --registration-commit <merge commit> --registered-at <GitHub merge time, UTC>
    forecast   record one forecast as of the last complete session
        .venv/bin/python scripts/forecast_prospective.py forecast \
            --bundle reports/ml/prospective/frozen/<bundle_id> \
            --store-root <prospective store> --forecast-close YYYY-MM-DD
    score      score every matured, unscored forecast row
        .venv/bin/python scripts/forecast_prospective.py score \
            --store-root <prospective store> --score-close YYYY-MM-DD
    report     cumulative band coverage and errors, with small-sample caveats
        .venv/bin/python scripts/forecast_prospective.py report

Registration (docs/research/prospective_registration.md, sections 1 and 4). The
registration time is GitHub's merge time of the registration pull request,
typed in by the owner: this script never fetches it. `register` checks that the
record file is in the merge commit, byte for byte, from the local repository.
A forecast or score is refused before the ledger's registration, from code
whose registered paths differ from the registered commit's or are uncommitted,
and from a bundle, config or rule other than the record's. No option selects a
zero-volatility rule: the registered rule is the only one.

Training reads the F1 research store only (sessions up to 2024-12-31); the
prospective store is read only through `load_inference_window`, with the
record's decision time as `retrieved_by`, and every window is checked
against the rules-based NYSE calendar. An asset with a close missing in its
window is refused (`incomplete_inference_window`), never median-filled; the
first eligible close needs every fund's window complete and spot-checked. The
ledger (default data/raw/prospective_forecasts/ledger.jsonl) and the outputs
(default reports/ml/prospective) are private and ignored by Git. Each
forecast is labelled "unvalidated, no demonstrated skill": these are research
records, never recommendations. Nothing here contacts a network.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# The reviewed consumer of the prospective store: only the loader and the rules
# calendar (tests/test_prospective_boundaries.py allows exactly these names).
from stock_agent.calendars.nyse_rules import CALENDAR_ID, sessions
from stock_agent.data.prospective.inference import load_inference_window
from stock_agent.forecasting.prospective import (
    forecast,
    freeze,
    frozen,
    ledger,
    registration_record,
)
from stock_agent.forecasting.prospective.config import load_config
from stock_agent.forecasting.prospective.refusal import ForecastRefused
from stock_agent.forecasting.provenance import (
    code_provenance,
    code_trees,
    commit_time,
    committed_file,
    environment,
    is_ancestor,
    paths_dirty,
)
from stock_agent.registry import open_registry

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "prospective" / "etf_prospective_v1.yaml"
DEFAULT_OUTPUT = PROJECT_ROOT / freeze.DEFAULT_OUTPUT
DEFAULT_LEDGER = PROJECT_ROOT / ledger.DEFAULT_LEDGER


def exchange_calendar(start, end) -> pd.DatetimeIndex:
    """The rules-based NYSE sessions in [start, end]: every window is checked against it."""

    return sessions(start, end)


exchange_calendar.calendar_id = CALENDAR_ID


def _now() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(UTC))


def _imported_here() -> None:
    """
    Refuse unless the imported package is this checkout's `src`: a worktree
    whose interpreter imports another checkout would record or check one tree
    and run another.
    """

    imported = Path(forecast.__file__).resolve().parents[3]  # src/stock_agent/forecasting/...
    if imported != (PROJECT_ROOT / "src").resolve():
        raise ForecastRefused(
            "registration_mismatch",
            f"stock_agent is imported from {imported}, not {PROJECT_ROOT / 'src'}: run with "
            "PYTHONPATH=src from the checkout being checked",
        )


def _code() -> dict[str, object]:
    """The running code: its commit, and the tree ids of the registered code paths."""

    _imported_here()
    paths = registration_record.CODE_PATHS
    try:
        return {
            **code_provenance(PROJECT_ROOT),
            "trees": code_trees(PROJECT_ROOT, "HEAD", paths),
            "trees_dirty": paths_dirty(PROJECT_ROOT, paths),
        }
    except subprocess.CalledProcessError:
        raise ForecastRefused(
            "registration_mismatch",
            f"the running code's tree ids for {list(paths)} could not be read from Git",
        ) from None


def _code_trees(commit: str) -> dict[str, str]:
    """The registered code paths' tree ids at the bundle's commit (local repository)."""

    registration_record.commit_id(commit, "the bundle's code commit")
    try:
        return code_trees(PROJECT_ROOT, commit, registration_record.CODE_PATHS)
    except subprocess.CalledProcessError:
        raise ForecastRefused(
            "registration_invalid",
            f"{commit} or a registered path is not in the local repository (fetch it first)",
        ) from None


def _commit_time(commit: str) -> pd.Timestamp:
    try:
        return pd.Timestamp(commit_time(PROJECT_ROOT, commit))
    except subprocess.CalledProcessError:
        raise ForecastRefused(
            "registration_invalid", f"{commit} is not in the local repository (fetch it first)"
        ) from None


def _is_ancestor(older: str, newer: str) -> bool:
    try:
        return is_ancestor(PROJECT_ROOT, older, newer)
    except subprocess.CalledProcessError:
        raise ForecastRefused(
            "registration_invalid",
            f"{older} or {newer} is not in the local repository (fetch it first)",
        ) from None


def _committed_record(entries) -> None:
    """
    Refuse a ledger whose registration record differs from the record file in
    its registration commit: a consistent rewrite of the whole ledger could
    change the one, never the other.
    """

    found = registration_record.current(entries)
    if found is None:
        return
    path = found.entry.payload["record_path"]
    try:
        committed = json.loads(_committed_file(found.registration_commit, path))
        same = registration_record.record_sha256(
            registration_record.validate_record(committed)
        ) == registration_record.record_sha256(found.record)
    except (json.JSONDecodeError, UnicodeDecodeError, ForecastRefused):
        same = False
    if not same:
        raise ForecastRefused(
            "registration_tampered",
            f"the ledger's registration record differs from {path} in {found.registration_commit}",
        )


def _committed_file(commit: str, path: str) -> bytes:
    try:
        return committed_file(PROJECT_ROOT, commit, path)
    except subprocess.CalledProcessError:
        raise ForecastRefused(
            "record_not_committed",
            f"{path} is not in commit {commit} of the local repository (fetch it first)",
        ) from None


def _models(bundle: Path) -> forecast.Models:
    output_root = bundle.resolve().parents[1]
    registry = open_registry(output_root / freeze.REGISTRY_DIRECTORY, output_root)
    return forecast.load_models(bundle, registry)


def _freeze(args) -> int:
    _imported_here()  # the bundle's code commit must describe the code that trains it
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


def _registration_record(args) -> int:
    models = _models(args.bundle)
    bundle = models.bundle
    record = registration_record.build_record(
        models,
        code_trees=_code_trees(bundle.document.get("code", {}).get("git_sha")),
        registration_document_commit=args.registration_doc_commit,
    )
    path = registration_record.write_record(args.registrations / f"{bundle.bundle_id}.json", record)
    print(f"registration record {path}")
    print(f"record sha256 {registration_record.record_sha256(record)}")
    print(f"bundle {bundle.bundle_id} manifest {bundle.manifest_sha256}")
    print(
        f"config {record['config_sha256']}; zero-volatility rule {record['zero_volatility_rule']}"
    )
    print("nothing is registered: add this file in a registration pull request for the owner")
    return 0


def _register(args) -> int:
    models = _models(args.bundle)
    path = args.record.resolve()
    try:
        relative = path.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        raise ForecastRefused(
            "registration_invalid", f"{path} is not inside the repository {PROJECT_ROOT}"
        ) from None
    record = registration_record.read_record(path)
    commit = registration_record.commit_id(args.registration_commit, "--registration-commit")
    if _committed_file(commit, relative) != path.read_bytes():
        raise ForecastRefused(
            "record_not_committed",
            f"{relative} differs from the file in the registration commit {commit}",
        )
    trees = record["code"]["trees"]
    if _code_trees(record["code"]["git_sha"]) != trees:
        raise ForecastRefused(
            "registration_mismatch",
            "the record's tree ids are not those of the bundle's commit "
            f"{record['code']['git_sha']}",
        )
    if _code_trees(commit) != trees:
        raise ForecastRefused(
            "registration_mismatch",
            f"the registration commit {commit} changes a registered code path since the freeze: "
            "freeze again from it",
        )
    document = record["registration_document"]["commit"]
    if not _is_ancestor(document, commit):
        raise ForecastRefused(
            "registration_invalid",
            f"the registration document's commit {document} is not in {commit}'s history",
        )
    try:
        registered_at = pd.Timestamp(args.registered_at)
    except ValueError:
        raise ForecastRefused(
            "registration_invalid", f"--registered-at {args.registered_at!r} is not a time"
        ) from None
    if registered_at.tzinfo is not None and registered_at < _commit_time(commit):
        raise ForecastRefused(
            "registered_before_commit",
            f"the registration time {registered_at.isoformat()} is before the merge commit "
            f"{commit} itself ({_commit_time(commit).isoformat()})",
        )
    entry = registration_record.register(
        args.ledger,
        record,
        models,
        record_path=relative,
        registration_commit=commit,
        registered_at=registered_at,
        recorded_at=_now(),
        code=_code(),
    )
    print(f"registered {entry.payload['record_sha256']} at {entry.payload['registered_at']}")
    print(f"ledger head {entry.entry_sha256}: keep a copy outside the ledger")
    return 0


def _forecast(args) -> int:
    models = _models(args.bundle)
    _committed_record(ledger.read_ledger(args.ledger))
    entry = forecast.record_forecast(
        models,
        loader=load_inference_window,
        store_root=args.store_root,
        forecast_close=pd.Timestamp(args.forecast_close, tz="UTC"),
        ledger_path=args.ledger,
        created_at=_now(),
        code=_code(),
        calendar=exchange_calendar,
    )
    session = entry.payload["session"]
    print(f"session {session['number']} ({session['phase']}; session 1 {session['first_session']})")
    if entry.kind == "refusal":
        print(f"refused as of {entry.payload['as_of_session']}: {entry.payload['refusals']}")
        print(f"ledger head {entry.entry_sha256}: keep a copy outside the ledger")
        return 1
    rows = pd.DataFrame(entry.payload["rows"])
    columns = ["asset", "horizon", "target", "point", "band_low", "band_high", "null", "stage"]
    print(f"forecast {entry.payload['forecast_id']} as of {entry.payload['as_of_session']}")
    print(", ".join(entry.payload["labels"]))
    print(rows.loc[:, [c for c in columns if c in rows]].to_string(index=False))
    for refusal in entry.payload["refusals"]:
        print(
            f"refused {refusal['asset']}: {refusal['reason']} {refusal['sessions_without_close']}"
        )
    print(f"ledger head {entry.entry_sha256}: note it outside the ledger to detect truncation")
    return 0


def _score(args) -> int:
    _committed_record(ledger.read_ledger(args.ledger))
    entry, left = forecast.score_matured(
        ledger_path=args.ledger,
        loader=load_inference_window,
        store_root=args.store_root,
        score_close=pd.Timestamp(args.score_close, tz="UTC"),
        scored_at=_now(),
        code=_code(),
        calendar=exchange_calendar,
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
    try:
        found = registration_record.current(entries)
    except ForecastRefused as refused:
        print(f"registration refused: {refused.reason}")
    else:
        if found is None:
            print("not registered")
        else:
            print(
                f"registered {found.registered_at.isoformat()} at {found.registration_commit} "
                f"(bundle {found.record['bundle']['bundle_id']})"
            )
    report = forecast.coverage_report(entries)
    print(f"{sum(e.kind == 'forecast' for e in entries)} forecasts; head {ledger.head(entries)}")
    pd.set_option("display.width", 200)
    print(report.drop(columns="caveat").to_string(index=False) if len(report) else "no scores yet")
    for text in sorted(set(report["caveat"])):
        print(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    commands = parser.add_subparsers(dest="command", required=True)
    freezing = commands.add_parser("freeze")
    freezing.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    freezing.add_argument("--research-store", type=Path, required=True)
    freezing.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    recording = commands.add_parser("registration-record")
    recording.add_argument("--bundle", type=Path, required=True)
    recording.add_argument(
        "--registration-doc-commit",
        required=True,
        help="the commit of docs/research/prospective_registration.md the owner approved",
    )
    recording.add_argument(
        "--registrations",
        type=Path,
        default=PROJECT_ROOT / registration_record.RECORD_DIRECTORY,
        help="the directory of registration records (in the repository)",
    )
    registering = commands.add_parser("register")
    registering.add_argument("--bundle", type=Path, required=True)
    registering.add_argument("--record", type=Path, required=True)
    registering.add_argument(
        "--registration-commit", required=True, help="the registration pull request's merge commit"
    )
    registering.add_argument(
        "--registered-at",
        required=True,
        help="GitHub's merge time of the registration pull request, with its timezone",
    )
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "freeze": _freeze,
        "registration-record": _registration_record,
        "register": _register,
        "forecast": _forecast,
        "score": _score,
        "report": _report,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
