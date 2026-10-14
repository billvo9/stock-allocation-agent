"""
Fund and cash data (F1): an auditable private store of source observations.

Modules:
    schema      closed enums, typed tables, the batch record, structured refusals
    validation  deterministic batch checks shared by ingest and the reader
    adapter     the replaceable `FundDataAdapter` seam and a synthetic adapter
    ingest      the write-once store writer (the only module that writes)
    reader      the fail-closed reader (re-verifies hashes and validation)

Stored: closes with their adjustment basis, dated distributions with their
share basis, splits, the raw provider payload and the retrieval time. Never
stored or computed: returns, adjusted series, filled gaps, availability
timestamps. See `docs/research/f1_fund_data.md`.
"""
