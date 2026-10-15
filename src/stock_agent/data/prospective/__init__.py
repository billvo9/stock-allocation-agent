"""
The prospective store: a private, write-once store of real ETF observations
for the prospective forecasting track, isolated from the F1 research store.

Owner decision of 2026-10-14: a separate prospective store may use the last 63
sessions before H_end (2026-10-14) only as inputs to frozen models, never to
train, tune or evaluate, and research code must not be able to read it.

Modules:
    window      H_END, LOOKBACK_SESSIONS, WINDOW_START and the complete-session
                request rule (18:00 America/New_York, the OD9 descriptive convention)
    validation  the store marker, the batch record and the batch checks shared by
                the writer and the reader
    store       the write-once writer (the only module that writes)
    inference   the only reader: `load_inference_window` for the forecast job

Isolation from the research store (F1, `stock_agent.data.funds`):
    - a root holds a `PROSPECTIVE_STORE.json` marker, and batch directories are
      named `P<retrieved_at>-<hash>`, so F1's `read_store` and `read_batch`
      refuse a prospective root or batch;
    - a prospective batch is not an F1 `RawBatch`, and its dates fall after
      F1's lockbox cap, so F1's `ingest_batch` refuses it;
    - tests forbid research packages from importing this package and require
      every research reader to refuse a prospective root
      (`tests/test_prospective_boundaries.py`).

Nothing here touches the network, adjusts a close, fills a gap or computes a
return.
"""
