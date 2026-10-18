"""
The Stock Lab's inference store (decision A2): a private, write-once store of
protected-period observations, separate from the research store, with strict
retrieval-time records. Fixtures only.
"""

from __future__ import annotations

import sqlite3
import stat

import pandas as pd
import pytest
from stock_lab_inference_helpers import (
    AS_OF,
    CALENDAR_ID,
    make_bars,
    ny,
    open_store,
    record,
    weekdays,
    window_start,
)

from stock_agent.model_validation.folds import MODEL_HOLDOUT_START
from stock_agent.stock_lab.inference.schema import (
    APPLICATION_ID,
    DB_NAME,
    HOME_ENV,
    PROTECTED_START,
    InferenceRefused,
    Refusal,
    default_path,
)
from stock_agent.stock_lab.inference.store import InferenceStore
from stock_agent.stock_lab.store import DB_NAME as RESEARCH_DB_NAME
from stock_agent.stock_lab.store import StockLabStore, StoreRefused
from stock_agent.stock_lab.symbols import parse_tickers

FIRST = window_start()
EVENING = ny(AS_OF, "21:00")


def refused(reason: Refusal):
    return pytest.raises(InferenceRefused, match=f"^{reason.value}: ")


def test_the_protected_period_starts_where_development_data_ends():
    assert pd.Timestamp(PROTECTED_START, tz="UTC") == pd.Timestamp(MODEL_HOLDOUT_START)


def test_the_default_location_is_private_and_separate(monkeypatch, tmp_path):
    monkeypatch.delenv(HOME_ENV, raising=False)
    path = default_path()
    assert path.name == DB_NAME and path.parent.name == "stock_lab_inference"
    assert path.parent.parent.name == "stock-agent-private"
    monkeypatch.setenv(HOME_ENV, str(tmp_path / "elsewhere"))
    assert default_path() == tmp_path / "elsewhere" / DB_NAME


def test_a_batch_is_stored_once_with_its_retrieval_and_recording_times(tmp_path):
    store, clock = open_store(tmp_path)
    assert not store.exists
    batch = record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    assert store.exists
    assert batch.request_start == FIRST and batch.request_end == AS_OF
    assert batch.row_count == 252 and batch.calendar_id == CALENDAR_ID
    assert batch.retrieved_at == "2026-10-16T01:00:00.000000Z"  # 21:00 New York, EDT
    assert batch.recorded_at == "2026-10-16T01:01:00.000000Z"  # the store's clock
    assert store.batches("NVDA") == (batch,)
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE(store.path.parent.stat().st_mode) == 0o700


def test_missing_parent_folders_are_created_private(tmp_path):
    store, clock = open_store(tmp_path / "a" / "b")
    record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    for folder in (tmp_path / "a", tmp_path / "a" / "b", store.path.parent):
        assert stat.S_IMODE(folder.stat().st_mode) == 0o700, folder
    with sqlite3.connect(store.path) as connection:
        assert connection.execute("PRAGMA application_id").fetchone()[0] == APPLICATION_ID


def test_an_identical_retrieval_of_the_same_sessions_is_not_stored_twice(tmp_path):
    store, clock = open_store(tmp_path)
    first = record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    again = record(store, clock, FIRST, AS_OF, retrieved_at=EVENING + pd.Timedelta(minutes=20))
    assert again == first and len(store.batches("NVDA")) == 1


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE bars SET close = 1.0",
        "DELETE FROM bars",
        "UPDATE batches SET retrieved_at = '2020-01-01T00:00:00.000000Z'",
        "DELETE FROM batches",
        "UPDATE meta SET value = '2'",
        "DELETE FROM meta",
        "INSERT OR REPLACE INTO meta VALUES ('store_kind', 'other')",
    ],
)
def test_stored_rows_can_never_be_rewritten(tmp_path, statement):
    store, clock = open_store(tmp_path)
    record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    with sqlite3.connect(store.path) as connection, pytest.raises(sqlite3.DatabaseError):
        connection.execute(statement)


def test_a_complete_batch_cannot_gain_rows_and_a_row_cannot_be_replaced(tmp_path):
    store, clock = open_store(tmp_path)
    batch = record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    with sqlite3.connect(store.path) as connection:
        for statement in (
            "INSERT INTO bars VALUES (?, '2026-10-16', 1, 1, 1, 1, 1, 0, 1)",
            f"INSERT OR REPLACE INTO bars VALUES (?, '{AS_OF}', 1, 1, 1, 1, 1, 0, 1)",
        ):
            with pytest.raises(sqlite3.DatabaseError):
                connection.execute(statement, (batch.batch_id,))


def test_development_sessions_belong_to_the_research_store_not_here(tmp_path):
    store, clock = open_store(tmp_path)
    with refused(Refusal.BEFORE_PROTECTED_PERIOD):
        record(store, clock, "2024-12-30", "2025-01-03", retrieved_at=EVENING)
    assert not store.exists  # nothing was created by a refused write


def test_the_rows_must_be_exactly_the_calendar_sessions_of_the_request(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    with refused(Refusal.NOT_ON_CALENDAR):  # a session left out instead of kept as a gap
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, bars=bars.drop(index=100))
    with refused(Refusal.NOT_ON_CALENDAR):  # rows ending before the requested end
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, bars=bars.iloc[:-1])
    weekend = bars.copy()
    weekend.loc[weekend.index[-1], "session"] = pd.Timestamp("2026-10-17", tz="UTC")
    with refused(Refusal.NOT_ON_CALENDAR):
        record(store, clock, FIRST, "2026-10-16", retrieved_at=EVENING, bars=weekend)
    with refused(Refusal.INVALID_REQUEST):  # a request that does not start on a session
        record(store, clock, "2025-10-18", AS_OF, retrieved_at=EVENING)
    gap = bars.copy()
    gap.loc[100, ["open", "high", "low", "close", "volume"]] = float("nan")
    batch = record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, bars=gap)
    assert batch.row_count == 252  # a session the provider did not serve stays a row


def test_a_batch_retrieved_before_its_last_close_was_available_is_refused(tmp_path):
    store, clock = open_store(tmp_path)
    with refused(Refusal.RETRIEVAL_TIME):
        record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "19:59"))
    record(store, clock, FIRST, AS_OF, retrieved_at=ny(AS_OF, "20:00"))


def test_a_retrieval_time_cannot_be_backdated_or_in_the_future(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)

    def write(retrieved_at):
        return store.record_batch(
            "NVDA",
            bars,
            source="fixture",
            retrieved_at=retrieved_at,
            request_start=FIRST,
            request_end=AS_OF,
            sessions=weekdays,
            calendar_id=CALENDAR_ID,
            data_class="real",
            vintage_note="fixture",
        )

    clock.now = EVENING + pd.Timedelta(hours=2)
    with refused(Refusal.RETRIEVAL_TIME):  # retrieved two hours before it is recorded
        write(EVENING)
    clock.now = EVENING
    with refused(Refusal.RETRIEVAL_TIME):
        write(EVENING + pd.Timedelta(minutes=10))
    with refused(Refusal.INVALID_REQUEST):
        write(pd.Timestamp("2026-10-16 01:00"))  # naive
    assert not store.exists


def test_synthetic_bars_only_under_demo_identifiers_and_real_never(tmp_path):
    store, clock = open_store(tmp_path)
    with refused(Refusal.DATA_CLASS):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, data_class="synthetic")
    with refused(Refusal.DATA_CLASS):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, ticker="DEMO_CALM")
    with refused(Refusal.INVALID_REQUEST):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, ticker="nvda")
    record(
        store, clock, FIRST, AS_OF, retrieved_at=EVENING, ticker="DEMO_CALM", data_class="synthetic"
    )


def test_invalid_bars_are_refused(tmp_path):
    store, clock = open_store(tmp_path)
    bars = make_bars(FIRST, AS_OF)
    bars.loc[5, "close"] = -1.0
    with refused(Refusal.INVALID_BARS):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING, bars=bars)


def test_a_store_inside_a_working_tree_or_beside_the_research_store_is_refused(tmp_path):
    repository = tmp_path / "repo"
    (repository / ".git").mkdir(parents=True)
    with refused(Refusal.LOCATION):
        InferenceStore(repository / "private" / DB_NAME)
    with refused(Refusal.LOCATION):  # stricter than the research store: not even data/raw/
        InferenceStore(repository / "data" / "raw" / "inference" / DB_NAME)
    shared = tmp_path / "shared"
    research = StockLabStore(shared / RESEARCH_DB_NAME)
    research.add(parse_tickers("NVDA"))
    with refused(Refusal.LOCATION):
        InferenceStore(shared / DB_NAME)
    with refused(Refusal.LOCATION):
        InferenceStore(shared / RESEARCH_DB_NAME)  # never under another name


def test_a_directory_readable_by_others_is_refused(tmp_path):
    directory = tmp_path / "open"
    directory.mkdir(mode=0o755)
    directory.chmod(0o755)
    store, clock = open_store(directory)
    store.path.parent.mkdir(mode=0o755)
    store.path.parent.chmod(0o755)
    with refused(Refusal.LOCATION):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)


def test_a_foreign_database_is_never_written(tmp_path):
    directory = tmp_path / "inference"
    directory.mkdir(mode=0o700)
    path = directory / DB_NAME
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE other (x)")
    path.chmod(0o600)
    before = path.read_bytes()
    store, clock = open_store(tmp_path)
    with refused(Refusal.NOT_INFERENCE_STORE):
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    assert path.read_bytes() == before


def test_the_research_store_refuses_the_inference_store_file(tmp_path):
    store, clock = open_store(tmp_path)
    record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    before = store.path.read_bytes()
    research = StockLabStore(store.path, clock=clock)
    for read in (
        research.watched,
        research.events,
        lambda: research.batches("NVDA"),
        lambda: research.bars("NVDA"),
        lambda: research.forecasts("NVDA"),
    ):
        with pytest.raises(StoreRefused, match="not a research store"):
            read()
    with pytest.raises(StoreRefused, match="not a research store"):
        research.add(parse_tickers("NVDA"))
    assert store.path.read_bytes() == before  # no research table was added


def test_a_file_that_is_not_a_database_is_refused_with_a_reason(tmp_path):
    directory = tmp_path / "inference"
    directory.mkdir(mode=0o700)
    (directory / DB_NAME).write_bytes(b"not a database at all" * 50)
    (directory / DB_NAME).chmod(0o600)
    store, clock = open_store(tmp_path)
    with pytest.raises(InferenceRefused) as caught:
        record(store, clock, FIRST, AS_OF, retrieved_at=EVENING)
    assert caught.value.reason in (Refusal.NOT_INFERENCE_STORE, Refusal.DATABASE_ERROR)
