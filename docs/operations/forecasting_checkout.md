# The pinned forecasting checkout (proposal)

Status (2026-10-18): a proposal for the owner. Nothing here is installed or scheduled, and no forecast job runs. It describes how the prospective forecast and score jobs would run once a registration exists (`docs/research/prospective_registration.md`).

## Why a separate checkout

The registration fixes the code that produces and scores forecasts. The job compares the Git object ids of `src`, `scripts/forecast_prospective.py`, `configs/prospective`, `pyproject.toml` and `requirements.txt` with those of the bundle's freeze commit, and refuses any difference (`registration_mismatch`).
- The identity is deliberately broad. The owner asked that it never be narrowed so far that "a modified transitive dependency could change forecasts without detection".
- Any change to `src` on `main`, even an unrelated one, therefore stops the job in a checkout that follows `main`.

A checkout pinned at the registration commit makes that broad check cheap: nothing in it ever changes, so the check always passes until a new registration replaces it.

The job also checks the installed environment against the bundle's manifest (`forecasting/prospective/environment.py`).
- **Refused (`environment_mismatch`):**
  - a different Python minor version or implementation;
  - a key distribution's version or content: numpy, pandas, pyarrow, python-dateutil, pyyaml, scikit-learn or scipy;
  - any difference in bottleneck, numexpr or numba, including their arrival, because pandas uses them when they are installed.
- **How content is measured.** It is the SHA-256 of the installed files as they are on disk now, so a file edited in place is caught. Files that depend on where or how the package was installed are left out (console scripts, INSTALLER, REQUESTED, direct_url.json), so the same wheels installed into this checkout's own venv match.
- **Recorded with each forecast, with both values:** every other difference, such as the Python patch release, the interpreter binary, the platform, the pins, the `America/New_York` zone file or another distribution.
- **Not covered:** imports made only at call time inside a function are not seen when the manifest is read. A test keeps the script's own imports within the job modules' imports.

## Three separate places

| Place | Purpose | Changes |
|---|---|---|
| Development worktrees (`.claude/worktrees/...`) and the main checkout | Building and reviewing changes | Constantly |
| The Model Health monitor clone (`~/stock-agent-monitor`, a clean `main`) | Running the daily health check (`scripts/setup_health_schedule.py`) | Follows `main` |
| The forecasting checkout (proposed: `~/stock-agent-forecast`) | Running `forecast`, `score`, `readiness` and `preflight` for one registration | Only through a new registration |

The forecasting checkout never runs development commands. It never pulls or merges, and it is never edited by hand.

## Setting it up for a registration

These are the owner's steps after the registration pull request is merged. They are not run by an agent.
1. Clone the repository and detach at the registration commit:
   ```bash
   git clone https://github.com/billvo9/stock-allocation-agent.git ~/stock-agent-forecast
   ```
   ```bash
   git -C ~/stock-agent-forecast checkout --detach <registration commit>
   ```
2. Build its own virtual environment from the bundle's manifest, with the same base interpreter:
   - Compare the interpreter's SHA-256 with the constraints header first:
     ```bash
     shasum -a 256 "$(realpath "$(command -v python3.12)")"
     ```
     A different interpreter binary is reported, not refused. A different minor version is refused.
   - Then create the venv:
     ```bash
     python3.12 -m venv ~/stock-agent-forecast/.venv
     ```
   - Print the bundle's constraints from a checkout that holds the bundle:
     ```bash
     PYTHONPATH=src .venv/bin/python scripts/forecast_prospective.py environment --bundle <bundle> > ~/stock-agent-private/forecast-constraints.txt
     ```
   - Install `requirements.txt` under those constraints:
     ```bash
     ~/stock-agent-forecast/.venv/bin/python -m pip install -r ~/stock-agent-forecast/requirements.txt -c ~/stock-agent-private/forecast-constraints.txt
     ```
     If pip cannot resolve it (the full requirements include the PPO and app stacks), it fails loudly; do not loosen the constraints.
3. Check it before any forecast:
   ```bash
   PYTHONPATH=src .venv/bin/python scripts/forecast_prospective.py preflight --bundle <bundle> --research-store <F1 store> --store-root <prospective store> --ledger-head-backup <kept head file> --out <private report path>
   ```
   Run it from `~/stock-agent-forecast`. The preflight's software checks must pass:
   - the code identity against the freeze commit;
   - the record against the bundle;
   - the environment against the bundle's manifest.
4. Always run the jobs with `PYTHONPATH=src` from this checkout. The job refuses a `stock_agent` package imported from any other checkout.

## Updating dependencies safely

Never in the forecasting checkout. A dependency update is a new registration:
1. Update and test it in a development worktree, through an ordinary reviewed pull request.
2. Freeze again from a clean commit. The new bundle carries the new manifest.
3. Open the registration pull request for the new bundle. Its merge time is the new registration time.
4. Register into a new ledger, because a registration is always its ledger's first entry. The earlier registration's record ends at its last recorded session, and the two are never pooled (section 1).
5. Build a new forecasting checkout, or replace the old one, at the new registration commit, with a new virtual environment from the new manifest.

An environment refusal on a session's evening means no record for that session. During the trial that is a void condition (section 5). Check the environment with `preflight` before each trial day, not only before registration.

A change that is only reported, such as a Python patch release or a non-key package, does not refuse forecasts. It is recorded with every forecast (`environment.differences`) and is visible for review. Do not upgrade the forecasting checkout's interpreter or packages to clear such a line. Leave it, or re-register.

## Re-registration checklist

- [ ] The change is merged to `main` through review.
- [ ] Freeze from a clean commit (`git status` empty; the job refuses a package imported from another checkout).
- [ ] `registration-record`, then the registration pull request; the owner merges it.
- [ ] `preflight` passes on the merge commit's checkout before `register`.
- [ ] `register` with `--pull-request <n>` into a new ledger path, and keep the printed head privately (section 6).
- [ ] Point the scheduled jobs at the new checkout and ledger. The forecast job is not scheduled yet.
