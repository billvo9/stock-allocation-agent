"""
Publication contract, version 1: versioned, write-once, hash-manifested
publications with a fail-closed reader.

Modules:
    envelope   enums, the envelope record, strict JSON, timestamps, ids,
               the lockbox check and the interim knowledge-time rule
    policy     the closed per-kind policy table and the one shared
               validation used by both the writer and the reader
    synthetic  the restricted synthetic-fixture adapter (version 1's only
               producer) and its provenance check
    writer     the only module that writes files
    reader     the fail-closed reader; it never imports the writer

Scope of version 1 (owner decisions of 2026-10-12 and 2026-10-13):
    - evidence statuses `descriptive` and `rules_illustration` only;
    - synthetic kinds only: real data is refused until the production
      availability rule (OD9) and a real-data kind exist;
    - the interim knowledge-time rule, for synthetic dates only: a
      synthetic session date d is known at d 23:59:59.999999Z. It is not
      a production market-data timestamp rule.

Integrity versus provenance:
    The manifest's SHA-256 hashes prove that the stored files are the
    files the manifest describes. They detect corruption and inconsistent
    edits, but anyone who can write the directory can recompute them, so
    they do not prove who produced a publication or from what inputs.
    Provenance is a separate, per-kind check in the policy table. For the
    synthetic kinds it is the adapter's regeneration of every row from
    the recorded specification. Real-data kinds will need their own
    provenance check; they must not rely on synthetic regeneration.
    `code_sha`, `code_dirty` and `created_at` are the producer's claims:
    hash-protected against accidental change, but not authenticated.

This docstring imports nothing, so loading the reader never loads the
writer.
"""
