# The Portfolio Lab Mac app

`Portfolio Lab.app` opens the Portfolio Lab like any other Mac application:
from Spotlight, the Apps view, Finder or the Dock, with no terminal. It is built
on this Mac by `scripts/build_mac_app.py`, with macOS's own tools and the Python
standard library only. Nothing new is installed.

## In short

1. Build it once (below). It lands in the Applications folder inside your home
   folder.
2. Open it with Spotlight: press Cmd-Space, type Portfolio Lab, press Return.
3. Keep it in the Dock: while it runs, Control-click its Dock icon and choose
   Options, then Keep in Dock.
4. Quit it (Cmd-Q) when you are done; that stops the Lab.

## Build and install

From the main checkout, after this change is on `main`:

```bash
.venv/bin/python scripts/build_mac_app.py
```

This builds the app and installs it as `~/Applications/Portfolio Lab.app`. That
is the Applications folder inside your home folder (in Finder: Go, then Home,
then Applications), not the Applications folder in Finder's sidebar.
- `--dest FOLDER` installs elsewhere. The folder must be outside the
  repository.
- `--checkout FOLDER` runs the Lab from another project folder. The default is
  the repository's main working tree (the one with the `.venv`), even when the
  script runs from a linked worktree.
- Running it again replaces the earlier build. It never replaces another app
  that happens to be called Portfolio Lab.

The app records the project folder's path. Most code changes reach the app with
a normal `git pull`, because the app runs the launcher
(`scripts/portfolio_lab_app.py`) from the project folder. Build the app again
after:
- moving or renaming the project folder (quit Portfolio Lab first);
- a change to `scripts/build_mac_app.py`;
- a change to the icon (`assets/brand/`).

## The first time

- **Folder access.** The project lives in your Desktop folder, so macOS asks
  once whether Portfolio Lab may access files in your Desktop folder. The
  prompt explains: "Portfolio Lab runs the Lab from its project folder. It only
  reads the folder." Choose Allow. If you chose Don't Allow, the app says macOS
  isn't letting it open the project folder. To allow it later, open System
  Settings > Privacy & Security > Files & Folders and turn on Desktop Folder
  under Portfolio Lab. macOS ties this permission to the app's signature, so it
  asks again after each rebuild.

  What Allow grants: Portfolio Lab, and every program it starts (the project's
  Python, Streamlit and its packages), may read your Desktop folder while it
  runs. The app's files in `~/Applications` are yours to change, so a program
  already running as you could alter the app to use that permission. A project
  folder outside Desktop, Documents and Downloads needs no permission at all;
  moving it is your decision (the evening job's paths would move too).
- **Gatekeeper.** A build made on this Mac carries no download quarantine, so
  Gatekeeper normally does not check it. If macOS still refuses to open the app,
  open System Settings > Privacy & Security, scroll to Security, choose Open
  Anyway for Portfolio Lab, and confirm. The app is signed ad hoc for this Mac,
  not notarized; a personal local app needs neither.

## Use

- **Open.** Nothing appears until the browser opens, usually within a second
  or two (longer the first time after restarting the Mac). While the Lab
  starts, the app's menu does not respond; that is normal. The Lab runs on this
  Mac only (127.0.0.1).
- **Open again.** Each click on the Dock icon opens the same Lab in another
  browser tab; close extra tabs freely. It never starts a second copy.
- **The dot under the Dock icon** means the Lab is still running, even with
  every browser tab closed. Switching to the app (Cmd-Tab) shows no window; the
  Lab lives in the browser.
- **Quit.** Quit Portfolio Lab like any app: Cmd-Q, or Quit in its Dock menu.
  Quitting stops the Lab's server; closing the browser tab does not. Logging out
  or shutting down quits it too.
- **Address.** The Lab uses port 8502 when it is free, like the terminal
  command, so the browser keeps your Light/Dark choice. When 8502 is busy (for
  example a Lab you started in a terminal), it takes the next free port, up to
  8599.
- **Which code runs.** The app runs whatever branch the project folder has
  checked out.

## When something goes wrong

The app shows a macOS dialog and quits; it never fails silently.

| Dialog or symptom | Cause | What to do |
|---|---|---|
| macOS isn't letting Portfolio Lab open its project folder | Folder access was denied | Allow it (see "The first time"), then open the app again |
| can't find its project folder | The folder moved or was renamed | Build the app again from the new place |
| Python environment (.venv) is missing | The `.venv` was deleted | `python3.12 -m venv .venv`, then `.venv/bin/pip install -r requirements.txt` |
| doesn't have the Lab's launcher | The project folder is older than this app | `git pull` on `main` |
| couldn't start | The Lab did not start or stopped answering | Choose Show Logs, read `lab.log`, then open the app again |
| The app seems stuck, or you used Force Quit | The Lab may still be running | Open Portfolio Lab again, then quit it normally |

Only the "couldn't start" dialog has a Show Logs button. Quitting after it also
stops a server that started but does not answer.

## Logs and private files

Everything the app writes is in `~/stock-agent-private/portfolio_lab_app/`
(owner-only permissions):

- `lab.log`: the Lab server's output. At the next start after it passes 5 MB,
  it is kept as `lab.log.1` and a new one begins.
- `launcher.log`: one line per start, reuse, stop and failure.
- `server.json`: the running server's process id, start time and port, while it
  runs.
- `launcher.lock`: keeps two launches from overlapping.

## What it never does

- It never reads or passes the Tiingo key or any other secret. The Lab's server
  gets a short allowlist of variables (`HOME`, `USER`, `LOGNAME`, `LANG`,
  `LC_ALL`, `LC_CTYPE`, `TMPDIR`) and a system `PATH`, nothing else.
- The app makes no network request beyond this Mac: the launcher's only request
  is the Lab's health check, on 127.0.0.1, and the build makes none. (Streamlit
  itself behaves as it does when started from a terminal.)
- It never starts a fetch, a forecast or any scheduled job. It starts only the
  Lab (`scripts/run_portfolio_lab.py`). The unattended evening job
  (`local.stock-lab.daily`) is separate, and the app neither reads nor changes
  it.
- It never writes inside the project folder. The evening job refuses a checkout
  with untracked files, and the Lab runs with `PYTHONDONTWRITEBYTECODE=1`, so
  not even a bytecode cache lands there. The build writes only into the
  destination folder.
- It stops only the server it started. Before a stop, the recorded process must
  still have the same process id and start time, so a process that reused the
  id is never signalled.

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
  keeps one sentence explaining the Desktop and Documents folder prompt.
  `codesign --sign -` reseals the bundle after these edits.
- **The icon.** When `assets/brand/portfolio-lab-icon.svg` exists, `sips`
  renders it (and `portfolio-lab-icon-small.svg` up to 32 pixels) at every
  iconset size and `iconutil` packs the `.icns`. Without it, the app keeps
  AppleScript's applet icon.

## Remove

Quit the app, drag it out of the Dock, then drag `~/Applications/Portfolio
Lab.app` to the Trash. The logs in `~/stock-agent-private/portfolio_lab_app/`
stay until you delete that folder. To also forget the folder permission:

```bash
tccutil reset SystemPolicyDesktopFolder local.stock-agent.portfolio-lab
```

## Possible later steps (not built)

- A native window instead of a browser tab (for example pywebview: a new
  dependency, an owner decision).
- A menu-bar item with the evening job's state (last run, next run, missed
  evenings, from Model Health) and Open Lab / Show Logs (for example rumps:
  also a new dependency).
