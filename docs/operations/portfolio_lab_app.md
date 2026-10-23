# The Portfolio Lab Mac app

`Portfolio Lab.app` opens the Portfolio Lab like any other Mac application:
from Spotlight, the Apps view, Finder or the Dock, with no terminal. It is built
on this Mac by `scripts/build_mac_app.py`, with macOS's own tools and the Python
standard library only. Nothing new is installed.

It opens the **Lab app**: the simple, phone-style screens
(`docs/product/lab_app_design.md`). The detailed **research pages** (the
earlier Streamlit Lab) run beside it, one level down: More, then Research pages.

## In short

1. Update the runner copy, then build the app once (below). It lands in the
   Applications folder inside your home folder.
2. Open it with Spotlight: press Cmd-Space, type Portfolio Lab, press Return.
3. Keep it in the Dock: while it runs, Control-click its Dock icon and choose
   Options, then Keep in Dock.
4. Quit it (Cmd-Q) when you are done; that stops the Lab.

## Build and install

The app runs from the **runner copy**, `~/stock-agent-runner`: the clean copy
of `main` that the evening job already runs from. It lies outside your Desktop,
Documents and Downloads folders, so macOS never asks for folder access (owner
decision of 2026-10-23).

After this change is merged on `main`, update the runner copy, then build. Run
these two commands in Terminal, one after the other:

```bash
git -C ~/stock-agent-runner pull --ff-only
```

```bash
~/stock-agent-runner/.venv/bin/python ~/stock-agent-runner/scripts/build_mac_app.py
```

This builds the app and installs it as `~/Applications/Portfolio Lab.app`. That
is the Applications folder inside your home folder (in Finder: Go, then Home,
then Applications), not the Applications folder in Finder's sidebar. The build
says which folder the app runs from, and that macOS won't ask.
- `--dest FOLDER` installs elsewhere. The folder must be outside the
  repository.
- `--checkout FOLDER` runs the Lab from another project folder. Without it the
  build uses `~/stock-agent-runner` when it exists, otherwise the repository's
  main working tree. A folder in Desktop, Documents or Downloads gets a warning:
  macOS will ask for access, and again after each rebuild.
- Running it again replaces the earlier build. It never replaces another app
  that happens to be called Portfolio Lab.

The app records the project folder's path. Code changes reach the app with the
same `git -C ~/stock-agent-runner pull --ff-only` you already use for the
evening job, because the app runs the launcher (`scripts/portfolio_lab_app.py`)
from that folder. Quit Portfolio Lab before pulling, then open it again. Build
the app again after:
- moving or renaming the project folder (quit Portfolio Lab first);
- a change to `scripts/build_mac_app.py`;
- a change to the icon (`assets/brand/`).

## The first time

- **Folder access.** Built from the runner copy, the app needs no permission:
  macOS doesn't ask. (Built from a folder in Desktop, Documents or Downloads,
  macOS asks once whether Portfolio Lab may access it, and again after each
  rebuild. To allow it later, open System Settings > Privacy & Security > Files
  & Folders and turn on the folder under Portfolio Lab.)
- **Gatekeeper.** A build made on this Mac carries no download quarantine, so
  Gatekeeper normally does not check it. If macOS still refuses to open the app,
  open System Settings > Privacy & Security, scroll to Security, choose Open
  Anyway for Portfolio Lab, and confirm. The app is signed ad hoc for this Mac,
  not notarized; a personal local app needs neither.

## Use

- **Open.** The Lab app appears in your browser, usually within a few seconds
  (longer the first time after restarting the Mac). While it starts, the app's
  menu does not respond; that is normal. Everything runs on this Mac only
  (127.0.0.1).
- **Open again.** Each click on the Dock icon opens the Lab app in another
  browser tab; close extra tabs freely. It never starts a second copy.
- **"Open Portfolio Lab from its app".** The page shows your numbers only when
  the app opened it: each start makes a new private key (below). A bookmarked
  or copied address shows this message instead; click the Dock icon.
- **The research pages.** More, then Research pages, opens them in a new tab.
  They start with the app and stop with it.
- **The dot under the Dock icon** means the Lab is still running, even with
  every browser tab closed. Switching to the app (Cmd-Tab) shows no window; the
  Lab lives in the browser.
- **Quit.** Quit Portfolio Lab like any app: Cmd-Q, or Quit in its Dock menu.
  Quitting stops both servers; closing the browser tab does not. Logging out or
  shutting down quits it too.
- **Addresses.** The Lab app uses port 8510 when it is free (otherwise the next
  free one up to 8599), and the research pages 8502 (up to 8509), so the browser
  keeps your Light/Dark choices.
- **Which code runs.** The app runs whatever the project folder has checked out:
  for the runner copy, `main`.

## This launch's private key

Your Mac's other accounts, and other programs, can reach any server on
127.0.0.1. So each time Portfolio Lab starts, it makes a new random key. The
Lab app shows numbers only to a page that has it. (Owner decision of
2026-10-23.)
- The key goes to the browser in the part of the address after `#`. Browsers
  never send that part to a server, and the page removes it from the address at
  once. The page then keeps the key for that browser tab only.
- It is stored only in the app's owner-only record (`server.json`). It is
  passed to the Lab app privately, and never printed or logged.
- It ends when you quit Portfolio Lab.
- One brief exposure remains: for an instant, the address with the key is handed
  to macOS's `open` command. An account listing every running program at that
  moment could see it, until the app quits.

## When something goes wrong

The app shows a macOS dialog and quits; it never fails silently.

| Dialog or symptom | Cause | What to do |
|---|---|---|
| macOS isn't letting Portfolio Lab open its project folder | Folder access was denied (a project in Desktop, Documents or Downloads) | Allow it (see "The first time"), or build from the runner copy |
| can't find its project folder | The folder moved or was renamed | Build the app again from the new place |
| Python environment (.venv) is missing | The `.venv` was deleted | Recreate it as the evening runbook describes, then open the app again |
| doesn't have the Lab's launcher, or a start script is missing | The project folder is older than this app | `git -C ~/stock-agent-runner pull --ff-only`, then open the app again |
| couldn't start | A server did not start or stopped answering | Choose Show Logs, read `lab.log` or `research.log`, then open the app again |
| The page says "Open Portfolio Lab from its app" | The tab wasn't opened by the app (bookmark, copied address) | Click the Dock icon |
| The app seems stuck, or you used Force Quit | The servers may still be running | Open Portfolio Lab again, then quit it normally |

Only the "couldn't start" dialog has a Show Logs button. Quitting after it also
stops a server that started but does not answer.

## Logs and private files

Everything the app writes is in `~/stock-agent-private/portfolio_lab_app/`
(owner-only permissions):

- `lab.log` and `research.log`: each server's output. At the next start after
  one passes 5 MB, it is kept as `.1` and a new one begins.
- `launcher.log`: one line per start, reuse, stop and failure (never the key).
- `server.json`: the running servers' process ids, start times and ports, and
  this launch's key, while they run.
- `launcher.lock`: keeps two launches from overlapping.

## What it never does

- It never reads or passes the Tiingo key or any other credential.
  - Both servers get a short allowlist of variables (`HOME`, `USER`, `LOGNAME`,
    `LANG`, `LC_ALL`, `LC_CTYPE`, `TMPDIR`) and a system `PATH`, nothing else.
  - The Lab app also gets its own project code (`PYTHONPATH`) and this launch's
    key.
- The app makes no network request beyond this Mac. The launcher's only
  requests are the servers' health checks, on 127.0.0.1, and the build makes
  none. (Streamlit itself behaves as it does when started from a terminal.)
- It never starts a fetch, a forecast or any scheduled job. It starts only the
  two servers (`scripts/run_lab_app.py` and `scripts/run_portfolio_lab.py`). The
  unattended evening job (`local.stock-lab.daily`) is separate; the app neither
  reads nor changes it, even when both run from the runner copy.
- It never writes inside the project folder. The evening job refuses a checkout
  with untracked files, and both servers run with `PYTHONDONTWRITEBYTECODE=1`,
  so not even a bytecode cache lands there. The build writes only into the
  destination folder.
- It stops only the servers it started. Before a stop, each recorded process
  must still have the same process id and start time, so a process that reused
  the id is never signalled.
- A server left running by the earlier, single-server app is stopped on the next
  open or quit, when it belongs to this project folder.

## How it is built

- **The app.** An AppleScript applet compiled by `osacompile` that stays open
  while the Lab runs. Its handlers are open (run and reopen) and quit:
  - opening checks that the project folder can be read and that
    `.venv/bin/python` and the launcher exist, then runs `.venv/bin/python -I
    -B scripts/portfolio_lab_app.py open`;
  - quitting runs `... stop`.

  The Info.plist names the app (`local.stock-agent.portfolio-lab`) and records
  the project folder and commit. It drops the applet's default privacy prompts
  (camera, contacts, controlling other apps), which the app never needs, and
  keeps one sentence explaining the Desktop and Documents folder prompt, for a
  project folder there. `codesign --sign -` reseals the bundle after these
  edits.
- **The icon.** When `assets/brand/portfolio-lab-icon.svg` exists, `sips`
  renders it (and `portfolio-lab-icon-small.svg` up to 32 pixels) at every
  iconset size and `iconutil` packs the `.icns`. Without it, the app keeps
  AppleScript's applet icon.

## The Lab app from a terminal

The same simple screens also run from one command in a project folder, beside
the app, on their own port:

```bash
.venv/bin/python scripts/run_lab_app.py --open
```

- It shows this Mac's Stock Lab, read-only. It writes nothing, fetches nothing,
  and runs no job.
- It makes its own key for that run. `--open` opens the browser with it; without
  `--open`, the terminal prints the address that holds it.
- To try it with made-up stocks first, add `--demo`. That takes about a minute
  to prepare, and a purple ribbon says the numbers are synthetic.
- Stop it with Control-C in the terminal window that started it.

## Remove

Quit the app, drag it out of the Dock, then drag `~/Applications/Portfolio
Lab.app` to the Trash. The logs in `~/stock-agent-private/portfolio_lab_app/`
stay until you delete that folder. If an earlier build from the Desktop was
allowed into your Desktop folder, forget that permission too:

```bash
tccutil reset SystemPolicyDesktopFolder local.stock-agent.portfolio-lab
```

## Possible later steps (not built)

- A native window instead of a browser tab (for example pywebview: a new
  dependency, an owner decision).
- A menu-bar item with the evening job's state (last run, next run, missed
  evenings, from Model Health) and Open Lab / Show Logs (for example rumps:
  also a new dependency).
