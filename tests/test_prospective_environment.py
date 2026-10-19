"""
The environment manifest of the forecast and score jobs: the Python version,
implementation and executable, the platform, the dependency pins and every
distribution the jobs import (version and RECORD hash). Freeze writes it into
the bundle, so the bundle's manifest SHA-256 (in the registration record)
registers it; forecast and score refuse a Python minor-version or
key-package mismatch and report the rest.
"""

from __future__ import annotations

import copy

import pytest
from forecast_prospective_helpers import RUN_ENVIRONMENT

from stock_agent.forecasting import provenance
from stock_agent.forecasting.prospective import environment
from stock_agent.forecasting.prospective.refusal import ForecastRefused


def _changed(**edits):
    manifest = copy.deepcopy(RUN_ENVIRONMENT)
    for path, value in edits.items():
        target = manifest
        *parents, last = path.split("__")
        for key in parents:
            target = target[key]
        if value is None:
            target.pop(last)
        else:
            target[last] = value
    return manifest


def test_the_fixture_manifest_is_valid_and_hashed():
    checked = environment.validate_manifest(RUN_ENVIRONMENT)
    assert checked == RUN_ENVIRONMENT
    assert len(environment.manifest_sha256(RUN_ENVIRONMENT)) == 64


@pytest.mark.parametrize(
    "edit",
    [
        {"python": None},
        {"python__minor": 3.12},
        {"distributions__numpy__content_sha256": "not a hash"},
        {"pins": "a string"},
        {"manifest_schema_version": "0.9"},
        {"extra": "key"},
        {"distributions": {}},
        {"pins__setup.cfg": "4" * 64},
        {"python__executable_sha256": "A" * 64},
        {"distributions__scipy": None},  # a key distribution must be present
        {"distributions__numpy__content_sha256": None},  # with a RECORD hash
    ],
)
def test_a_malformed_manifest_is_refused(edit):
    with pytest.raises(ForecastRefused) as refused:
        environment.validate_manifest(_changed(**edit))
    assert refused.value.reason == "environment_invalid"


def test_the_same_environment_passes_with_nothing_to_report():
    report = environment.check(RUN_ENVIRONMENT, RUN_ENVIRONMENT)
    assert report == {
        "registered_sha256": environment.manifest_sha256(RUN_ENVIRONMENT),
        "running_sha256": environment.manifest_sha256(RUN_ENVIRONMENT),
        "differences": {},
    }


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        ({"python__minor": "3.13", "python__version": "3.13.0"}, "python.minor"),
        ({"python__implementation": "pypy"}, "python.implementation"),
        ({"distributions__numpy__version": "9.9.9"}, "distributions.numpy.version"),
        (
            {"distributions__pandas__content_sha256": "f" * 64},
            "distributions.pandas.content_sha256",
        ),
    ],
)
def test_a_python_minor_or_key_package_change_is_refused(edit, named):
    with pytest.raises(ForecastRefused) as refused:
        environment.check(RUN_ENVIRONMENT, _changed(**edit))
    assert refused.value.reason == "environment_mismatch"
    assert named in refused.value.detail


@pytest.mark.parametrize(
    ("edit", "named"),
    [
        ({"python__version": "3.12.9"}, "python.version"),
        ({"python__executable_sha256": "f" * 64}, "python.executable_sha256"),
        ({"platform__release": "28.0.0"}, "platform.release"),
        ({"pins__requirements.txt": "f" * 64}, "pins.requirements.txt"),
        ({"distributions__tzdata__version": "2099.1"}, "distributions.tzdata.version"),
        ({"distributions__tzdata": None}, "distributions.tzdata"),  # removed: named once
        (
            {"distributions__six": {"version": "1.17.0", "content_sha256": "9" * 64}},
            "distributions.six",  # added: named once
        ),
        ({"timezone__sha256": "e" * 64}, "timezone.sha256"),  # a new tz database release
    ],
)
def test_other_changes_are_reported_not_refused(edit, named):
    report = environment.check(RUN_ENVIRONMENT, _changed(**edit))
    assert list(report["differences"]) == [named]


def test_a_reported_difference_keeps_both_values():
    report = environment.check(RUN_ENVIRONMENT, _changed(python__version="3.12.9"))
    assert report["differences"] == {
        "python.version": {"registered": "3.12.2", "running": "3.12.9"}
    }


@pytest.mark.parametrize("accelerator", ["bottleneck", "numexpr", "numba"])
def test_an_added_pandas_accelerator_is_refused(accelerator):
    added = {"version": "1.0.0", "content_sha256": "9" * 64}
    with pytest.raises(ForecastRefused) as refused:
        environment.check(RUN_ENVIRONMENT, _changed(**{f"distributions__{accelerator}": added}))
    assert refused.value.reason == "environment_mismatch"


def test_the_key_distributions_are_the_ones_the_arithmetic_uses():
    assert set(environment.KEY_DISTRIBUTIONS) == {
        "numpy",
        "pandas",
        "pyarrow",
        "python-dateutil",
        "pyyaml",
        "scikit-learn",
        "scipy",
    }
    assert set(environment.GUARDED_OPTIONAL) == {"bottleneck", "numexpr", "numba"}


@pytest.mark.parametrize(
    "name", ["numpy", "pandas", "pyarrow", "python-dateutil", "pyyaml", "scikit-learn", "scipy"]
)
@pytest.mark.parametrize("field", ["version", "content_sha256"])
def test_every_key_distribution_change_is_refused(name, field):
    value = "0.0.1" if field == "version" else "e" * 64
    with pytest.raises(ForecastRefused) as refused:
        environment.check(RUN_ENVIRONMENT, _changed(**{f"distributions__{name}__{field}": value}))
    assert refused.value.reason == "environment_mismatch"
    assert f"distributions.{name}.{field}" in refused.value.detail


def test_a_bundle_without_a_manifest_is_refused():
    with pytest.raises(ForecastRefused) as refused:
        environment.check(None, RUN_ENVIRONMENT)
    assert refused.value.reason == "environment_unregistered"


# --- reading the real environment (offline: the interpreter and installed metadata) ---------


def test_the_real_manifest_names_the_jobs_imported_distributions(tmp_path):
    from forecast_prospective_helpers import REPO

    modules = (
        "stock_agent.forecasting.prospective.forecast",
        "stock_agent.forecasting.prospective.freeze",
        "stock_agent.data.prospective.inference",
    )
    first = provenance.environment_manifest(REPO, modules, required=environment.KEY_DISTRIBUTIONS)
    environment.validate_manifest(first)
    assert first["python"]["minor"] == "3.12"
    assert set(environment.KEY_DISTRIBUTIONS) <= set(first["distributions"])
    assert all(first["distributions"][k]["content_sha256"] for k in environment.KEY_DISTRIBUTIONS)
    # Read in a fresh interpreter: nothing this pytest process loaded leaks in.
    for absent in ("pytest", "pluggy", "streamlit", "stock-allocation-agent"):
        assert absent not in first["distributions"], absent
    assert first["timezone"]["key"] == "America/New_York" and len(first["timezone"]["sha256"]) == 64
    assert set(first["pins"]) == {"requirements.txt", "pyproject.toml"}
    # Deterministic: a second reading of the same environment is identical.
    again = provenance.environment_manifest(REPO, modules, required=environment.KEY_DISTRIBUTIONS)
    assert again == first


# --- freeze, forecast, score and the preflight ----------------------------------------------


def test_freezing_writes_the_manifest_into_the_bundle(prospective_frozen):
    document = prospective_frozen.result.bundle.document
    assert document["environment_manifest"] == RUN_ENVIRONMENT


def test_freezing_refuses_a_missing_manifest_before_reading_data(prospective_frozen, tmp_path):
    from forecast_prospective_helpers import PROVENANCE, STARTED

    from stock_agent.forecasting.prospective import freeze

    provenance_ = {k: v for k, v in PROVENANCE.items() if k != "environment_manifest"}
    with pytest.raises(ForecastRefused) as refused:
        freeze.freeze(
            prospective_frozen.config,
            research_store_root=tmp_path / "never_read",
            output_root=tmp_path / "out",
            provenance=provenance_,
            started_at=STARTED,
        )
    assert refused.value.reason == "environment_invalid"
    assert not (tmp_path / "out").exists()


@pytest.fixture
def ledger_path(tmp_path):
    return tmp_path / "ledger.jsonl"


def test_a_forecast_from_another_python_minor_is_refused(
    prospective_frozen, ledger_path, prospective_store_root
):
    from forecast_prospective_helpers import record

    from stock_agent.forecasting.prospective import ledger

    running = _changed(python__minor="3.13", python__version="3.13.0")
    with pytest.raises(ForecastRefused) as refused:
        record(prospective_frozen, ledger_path, prospective_store_root, environment=running)
    assert refused.value.reason == "environment_mismatch"
    assert [e.kind for e in ledger.read_ledger(ledger_path)] == ["registration"]


def test_a_forecast_records_its_environment_and_what_differs(
    prospective_frozen, ledger_path, prospective_store_root
):
    from forecast_prospective_helpers import record

    running = _changed(python__version="3.12.9")
    entry = record(prospective_frozen, ledger_path, prospective_store_root, environment=running)
    assert entry.payload["forecast_schema_version"] == "1.3"
    assert entry.payload["environment"] == {
        "registered_sha256": environment.manifest_sha256(RUN_ENVIRONMENT),
        "running_sha256": environment.manifest_sha256(running),
        "differences": {"python.version": {"registered": "3.12.2", "running": "3.12.9"}},
        "manifest": RUN_ENVIRONMENT,  # the bundle's, so scoring can apply the same rule
    }


def test_a_forecast_from_a_bundle_without_a_manifest_is_refused(
    prospective_frozen, ledger_path, prospective_store_root
):
    import dataclasses

    from forecast_prospective_helpers import record

    from stock_agent.forecasting.prospective import forecast

    models = prospective_frozen.models()
    document = {k: v for k, v in models.bundle.document.items() if k != "environment_manifest"}
    older = forecast.Models(dataclasses.replace(models.bundle, document=document), models.entries)
    with pytest.raises(ForecastRefused) as refused:
        record(prospective_frozen, ledger_path, prospective_store_root, models=older)
    assert refused.value.reason == "environment_unregistered"


def test_scoring_refuses_another_key_package(
    prospective_frozen, ledger_path, prospective_store_root
):
    import pandas as pd
    from forecast_prospective_helpers import RUN_CODE, exchange_calendar, record

    from stock_agent.data.prospective.inference import load_inference_window
    from stock_agent.forecasting.prospective import forecast

    record(prospective_frozen, ledger_path, prospective_store_root)
    common = {
        "ledger_path": ledger_path,
        "loader": load_inference_window,
        "store_root": prospective_store_root,
        "score_close": "2026-10-23",
        "scored_at": pd.Timestamp("2026-10-23T23:00:00Z"),
        "code": RUN_CODE,
        "calendar": exchange_calendar,
    }
    with pytest.raises(ForecastRefused) as refused:
        forecast.score_matured(
            **common, environment=_changed(distributions__numpy__version="9.0.0")
        )
    assert refused.value.reason == "environment_mismatch"
    written, _ = forecast.score_matured(**common, environment=RUN_ENVIRONMENT)
    assert written.payload["score_schema_version"] == "1.3"
    assert written.payload["environment"]["differences"] == {}


def test_the_preflight_compares_the_running_environment_with_the_bundles():
    from stock_agent.forecasting.prospective import preflight

    assert preflight.environment_manifest(RUN_ENVIRONMENT, RUN_ENVIRONMENT).status == "pass"
    item = preflight.environment_manifest(RUN_ENVIRONMENT, _changed(python__minor="3.13"))
    assert (item.status, item.reason) == ("fail", "environment_mismatch")
    missing = preflight.environment_manifest(None, RUN_ENVIRONMENT)
    assert (missing.status, missing.reason) == ("fail", "environment_unregistered")
    reported = preflight.environment_manifest(RUN_ENVIRONMENT, _changed(python__version="3.12.9"))
    assert reported.status == "pass"
    assert list(reported.evidence["differences"]) == ["python.version"]


def _other():
    return _changed(distributions__tzdata__version="2099.1")


def _scoring_ledger(prospective_frozen, ledger_path, store, change):
    """A registered ledger with the fixture's forecast, then a planted forecast `change`d."""

    import pandas as pd
    from forecast_prospective_helpers import record

    from stock_agent.forecasting.prospective import ledger

    first = record(prospective_frozen, ledger_path, store)
    planted = copy.deepcopy(dict(first.payload))
    planted["as_of_session"] = "2026-10-15"
    planted["forecast_id"] = "planted"
    change(planted)
    ledger.append_entry(
        ledger_path, "forecast", planted, recorded_at=pd.Timestamp("2026-10-15T23:00:00Z")
    )
    return first


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (lambda p: p.pop("environment"), "environment_unregistered"),  # a mixed ledger
        (
            lambda p: p["environment"].update(
                manifest=_other(), registered_sha256=environment.manifest_sha256(_other())
            ),
            "environment_mismatch",  # two (each self-consistent) manifests in one ledger
        ),
    ],
    ids=["entry_without_environment", "two_manifests"],
)
def test_scoring_refuses_a_ledger_whose_forecasts_disagree_on_the_environment(
    prospective_frozen, ledger_path, prospective_store_root, change, reason
):
    import pandas as pd
    from forecast_prospective_helpers import RUN_CODE, exchange_calendar

    from stock_agent.data.prospective.inference import load_inference_window
    from stock_agent.forecasting.prospective import forecast, ledger

    _scoring_ledger(prospective_frozen, ledger_path, prospective_store_root, change)
    with pytest.raises(ForecastRefused) as refused:
        forecast.score_matured(
            ledger_path=ledger_path,
            loader=load_inference_window,
            store_root=prospective_store_root,
            score_close="2026-10-23",
            scored_at=pd.Timestamp("2026-10-23T23:00:00Z"),
            code=RUN_CODE,
            calendar=exchange_calendar,
            environment=RUN_ENVIRONMENT,
        )
    assert refused.value.reason == reason
    assert "score" not in [e.kind for e in ledger.read_ledger(ledger_path)]


# --- the content digest: the installed files, not where or how they were installed ---------

RECORD = """numpy/__init__.py,sha256=AAAA,100
numpy/core/_multiarray.so,sha256=BBBB,2000
numpy/__pycache__/__init__.cpython-312.pyc,,
numpy-2.5.1.dist-info/METADATA,sha256=CCCC,50
"""
INSTALL_ROWS = """../../../bin/f2py,sha256=DDDD,213
numpy-2.5.1.dist-info/INSTALLER,sha256=EEEE,4
numpy-2.5.1.dist-info/REQUESTED,sha256=FFFF,0
numpy-2.5.1.dist-info/direct_url.json,sha256=GGGG,80
numpy-2.5.1.dist-info/RECORD,,
"""


def test_install_specific_rows_never_enter_the_digest():
    plain = provenance.content_paths(RECORD)
    assert plain == [
        "numpy-2.5.1.dist-info/METADATA",
        "numpy/__init__.py",
        "numpy/core/_multiarray.so",
    ]
    assert provenance.content_paths(RECORD + INSTALL_ROWS) == plain


def test_the_digest_follows_the_files_content(tmp_path):
    for path in provenance.content_paths(RECORD):
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_bytes(path.encode())
    before = provenance.content_digest(provenance.content_paths(RECORD), tmp_path.joinpath)
    assert before == provenance.content_digest(
        provenance.content_paths(RECORD + INSTALL_ROWS), tmp_path.joinpath
    )
    (tmp_path / "numpy" / "__init__.py").write_bytes(b"edited in place")
    edited = provenance.content_digest(provenance.content_paths(RECORD), tmp_path.joinpath)
    assert edited != before
    (tmp_path / "numpy" / "__init__.py").unlink()
    assert provenance.content_digest(provenance.content_paths(RECORD), tmp_path.joinpath) != edited


@pytest.mark.parametrize(
    ("name", "canonical"),
    [("PyYAML", "pyyaml"), ("scikit_learn", "scikit-learn"), ("zope.interface", "zope-interface")],
)
def test_distribution_names_are_normalized(name, canonical):
    assert provenance.canonical_name(name) == canonical


def test_the_jobs_script_imports_nothing_beyond_its_job_modules():
    import json
    import subprocess
    import sys

    from forecast_prospective_helpers import REPO

    script = REPO / "scripts" / "forecast_prospective.py"
    probe = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('job', {str(script)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print(json.dumps([sorted({m.split('.')[0] for m in sys.modules}), module.JOB_MODULES]))"
    )
    env = {"PYTHONPATH": str(REPO / "src"), "PATH": "/usr/bin:/bin"}
    loaded, modules = json.loads(
        subprocess.run(
            [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
        ).stdout
    )
    job = provenance.environment_manifest(REPO, modules, required=environment.KEY_DISTRIBUTIONS)
    from importlib.metadata import packages_distributions

    owners = packages_distributions()
    script_distributions = {
        provenance.canonical_name(d) for m in loaded for d in owners.get(m, ())
    } - {"stock-allocation-agent"}
    assert script_distributions <= set(job["distributions"])


def test_scoring_refuses_a_manifest_that_does_not_match_its_recorded_hash(
    prospective_frozen, ledger_path, prospective_store_root
):
    import pandas as pd
    from forecast_prospective_helpers import RUN_CODE, exchange_calendar

    from stock_agent.data.prospective.inference import load_inference_window
    from stock_agent.forecasting.prospective import forecast

    def edit(payload):  # the planted entry's manifest no longer matches its own hash
        payload["environment"]["manifest"] = _changed(python__version="3.12.3")

    _scoring_ledger(prospective_frozen, ledger_path, prospective_store_root, edit)
    with pytest.raises(ForecastRefused) as refused:
        forecast.score_matured(
            ledger_path=ledger_path,
            loader=load_inference_window,
            store_root=prospective_store_root,
            score_close="2026-10-23",
            scored_at=pd.Timestamp("2026-10-23T23:00:00Z"),
            code=RUN_CODE,
            calendar=exchange_calendar,
            environment=RUN_ENVIRONMENT,
        )
    assert refused.value.reason == "environment_invalid"
