"""
The Stock Lab's daily premarket briefing (read-only; standard library, numpy,
pandas and the Stock Lab itself).

A briefing is prepared for one NYSE session, ideally at 08:30 New York time
that morning, from records the Stock Lab has already checked and stored:
the watchlist, the recorded forecasts, their scores, the evaluation runs and
the daily job's runs. It never retrains, never recomputes an old forecast,
never writes to the store or its ledger, and never runs the evening job. A
probability appears only when a stored forecast holds it; anything missing
is shown as unavailable, with its reason.

Modules
    timing       which session a briefing is for, the 08:30 target, weekends
                 and holidays (the exchange calendar is passed in)
    reliability  how often past forecasts in a probability range came true,
                 with date-clustered intervals, and matched comparisons with
                 the recent-weighted baseline (B1)
    report       the briefing itself, assembled from the store, read-only
    render       the briefing as plain text and as HTML
    delivery     the delivery-adapter interface, the local preview adapter
                 and the per-market-date duplicate-send log

Nothing here sends anything anywhere: the only adapter writes files into a
private local directory.
"""
