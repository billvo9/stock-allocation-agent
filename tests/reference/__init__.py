"""
Test-only reference implementations for the T5a engine:

    t5a_reference  a loop-based shares-and-cash ledger of the approved
                   D1-D8 rules (the oracle)
    replay         the replay-equivalence harness (engine vs RL environment)
    timing         detectors for same-close execution and future-peeking callers

Nothing here is production code. The owner approved OD8 group B (D1-D8) on
2026-10-15; the production engine is tested against these references.
"""
