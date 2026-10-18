"""
The Stock Lab's inference store: a private, write-once store of protected-
period observations (sessions from 2025-01-01 on), separate from the research
store (`stock_lab.store`). Owner decision A2 (2026-10-16): recent
observations serve "only as inputs to a previously frozen model", kept in a
separately controlled store, with strict as-of and retrieval-time records; a
scoped authorization, not an unlocking of the 2025+ holdout. Decision A6:
recent observed prices may be shown on the private page through an isolated
descriptive reader.

Modules (this file imports none of them, so loading one loads only what it
needs):
    schema   the contract shared by the writer and the readers: location,
             the SQLite identity, the batch record, the timing conventions,
             the integrity checks and the vintage rule
    store    the write-once writer (the only module that writes)
    window   the frozen-model reader: exactly INFERENCE_SESSIONS (252)
             sessions ending at the as-of close, as stored by the decision
             time; it refuses anything else
    recent   the descriptive reader for the Lab page: recent observed prices
             and their daily change, never a category, a feature or a score

Isolation (tests/test_stock_lab_inference_boundaries.py):
    - no module outside this package imports it, except the Lab page, which
      may import only the descriptive reader's names, and the owner-run
      inference fetch script; research, evaluation, training, tuning and
      backtest code never imports it or names its file;
    - the store is its own SQLite file in its own directory, marked in the
      file header (`application_id`): the research store refuses to read or
      write it, and this store refuses the research store's file and
      directory;
    - nothing here touches the network, imports a calendar or reads a clock
      except the writer's injected clock.
"""
