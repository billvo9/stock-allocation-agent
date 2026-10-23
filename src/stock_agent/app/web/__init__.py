"""
The Lab app: the Portfolio Lab's simple screens for the Stock Lab (Home first;
docs/product/lab_app_design.md). A small standard-library web server on this
Mac only, with plain HTML, CSS and JavaScript, over read-only view models.

Modules:
    words   every sentence and number format the screens show
    home    the Home view model: the status line, each stock's last close and
            move, its one-month line and the next session's recorded outlook
    server  the local, read-only HTTP server: fixed routes, the DNS-rebinding
            check and the security headers
    static/ the page, its stylesheet, its script and the mark

Nothing here imports the exchange calendar or the recent-prices reader: the
launcher (scripts/run_lab_app.py) passes them in, as the premarket briefing's
script does. Nothing here writes anything, computes a value from recent prices
(decision A6) or recomputes a recorded probability.
"""
