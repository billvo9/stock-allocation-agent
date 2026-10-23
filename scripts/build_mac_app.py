"""
Build "Portfolio Lab.app", a Mac app that opens the Portfolio Lab from the Dock.

    .venv/bin/python scripts/build_mac_app.py [--dest ~/Applications] [--checkout PATH]

The app. An AppleScript applet compiled by macOS's own `osacompile` (no new
dependency), kept open while the Lab runs:
- opening it runs scripts/portfolio_lab_app.py from the checkout with the
  checkout's .venv Python, which starts the Lab on 127.0.0.1 once and shows it
  in the default browser; opening it again only shows the running Lab;
- quitting it (Cmd-Q, the Dock's Quit, logging out) stops the Lab's server;
- when the checkout, its .venv or the launcher is missing, or macOS blocks
  access to the checkout, a macOS dialog says what to do and the app quits;
- when the launcher fails, a dialog shows its message (with Show Logs) and the
  app quits, which also stops a server that started but does not answer.

The checkout. By default the runner checkout, ~/stock-agent-runner, when it
exists: the clean copy of main the evening job already runs from (owner
decision of 2026-10-23). It lies outside Desktop, Documents and Downloads, so
macOS never asks the app for folder access. Without it, the repository's main
working tree (the one with the .venv), even when this script runs from a linked
worktree; --checkout picks another. A checkout in Desktop, Documents or
Downloads gets a warning: macOS asks for access, again after each rebuild. The
app opens the Lab app and starts the research pages beside it
(scripts/portfolio_lab_app.py); both run from the checkout and write nothing
into it. Its path is recorded in the app, so rebuild after moving the project
folder.

The icon. When assets/brand/portfolio-lab-icon.svg exists, `sips` renders it
(and portfolio-lab-icon-small.svg for 16 and 32 pixels) at every iconset size
and `iconutil` packs the .icns; otherwise the app keeps the applet's icon.

Install. The bundle is assembled in a hidden temporary folder inside --dest
(default ~/Applications), then moved into place. An earlier Portfolio Lab
built by this script is replaced; any other app with that name is refused.
--dest must be outside the checkout and this repository: nothing is written
inside them, so the evening job's clean-checkout guard is never tripped.

The app is signed ad hoc for this Mac (`codesign --sign -`, which Apple
silicon requires after the bundle is edited); it is not notarized. The build
reads no key and makes no network request. Standard library only.

Exit status: 0 built and installed, 2 refused (with the reason on stderr).
"""

from __future__ import annotations

import argparse
import os
import plistlib
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
APP_NAME = "Portfolio Lab.app"
BUNDLE_ID = "local.stock-agent.portfolio-lab"
DEFAULT_DEST = Path.home() / "Applications"
PRIVATE_DIR = Path.home() / "stock-agent-private" / "portfolio_lab_app"
LAUNCHER = Path("scripts") / "portfolio_lab_app.py"
LAB_SCRIPT = Path("scripts") / "run_portfolio_lab.py"  # the research pages (Streamlit)
LAB_APP_SCRIPT = Path("scripts") / "run_lab_app.py"  # the Lab app the app opens
RUNNER = Path.home() / "stock-agent-runner"  # the evening job's clean copy of main
# Folders macOS protects: an app reading a project folder in one of them must ask first.
PROTECTED = ("Desktop", "Documents", "Downloads")
VENV_PYTHON = Path(".venv") / "bin" / "python"
BRAND = Path("assets") / "brand"
ICON_SVG, SMALL_ICON_SVG = "portfolio-lab-icon.svg", "portfolio-lab-icon-small.svg"
# (pixels, iconset file stem); sizes up to 32 pixels use the simplified small artwork.
ICONSET = (
    (16, "icon_16x16"),
    (32, "icon_16x16@2x"),
    (32, "icon_32x32"),
    (64, "icon_32x32@2x"),
    (128, "icon_128x128"),
    (256, "icon_128x128@2x"),
    (256, "icon_256x256"),
    (512, "icon_256x256@2x"),
    (512, "icon_512x512"),
    (1024, "icon_512x512@2x"),
)
SMALL_ICON_MAX = 32
GIT = "/usr/bin/git"
LSREGISTER = Path(
    "/System/Library/Frameworks/CoreServices.framework/Frameworks/"
    "LaunchServices.framework/Support/lsregister"
)

Runner = Callable[[list[str]], None]


class BuildRefused(Exception):
    """The build cannot go ahead; the message says why and what to do."""


def run(command: list[str]) -> None:
    """Run a macOS tool, raising BuildRefused with its own message when it fails."""

    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise BuildRefused(f"{Path(command[0]).name} failed: {detail}")


def git_environment() -> dict[str, str]:
    """The caller's environment without GIT_* variables, which could point git elsewhere."""

    return {name: value for name, value in os.environ.items() if not name.startswith("GIT_")}


def main_worktree(start: Path = REPO) -> Path:
    """The repository's main working tree (where the .venv lives), seen from any worktree."""

    result = subprocess.run(
        [GIT, "-C", str(start), "rev-parse", "--path-format=absolute", "--git-common-dir"],
        capture_output=True,
        text=True,
        check=False,
        env=git_environment(),
    )
    if result.returncode != 0 or not result.stdout.strip():
        return start.resolve()
    return Path(result.stdout.strip()).resolve().parent


def default_checkout(runner: Path = RUNNER, start: Path = REPO) -> Path:
    """The runner checkout when it exists, else the repository's main working tree."""

    return runner if (runner / ".git").exists() else main_worktree(start)


def protected(checkout: Path, home: Path | None = None) -> bool:
    """Whether macOS will ask the app for access to read `checkout`."""

    home = (home or Path.home()).resolve()
    resolved = checkout.expanduser().resolve()
    return any(resolved.is_relative_to(home / folder) for folder in PROTECTED)


def commit_of(checkout: Path) -> str:
    result = subprocess.run(
        [GIT, "-C", str(checkout), "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        env=git_environment(),
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def check_checkout(checkout: Path) -> None:
    """Refuse a checkout the app could not run the Lab from."""

    if not checkout.is_dir():
        raise BuildRefused(f"No project folder at {checkout}.")
    if not (checkout / VENV_PYTHON).is_file():
        raise BuildRefused(
            f"{checkout} has no .venv, which the app runs the Lab with. "
            "Create it (python3.12 -m venv .venv, then .venv/bin/pip install -r "
            "requirements.txt) and build again."
        )
    for part in (LAUNCHER, LAB_SCRIPT, LAB_APP_SCRIPT):
        if not (checkout / part).is_file():
            raise BuildRefused(
                f"{checkout} has no {part}. Update it to the latest main "
                f"(git -C {checkout} pull --ff-only), or pass --checkout, and build again."
            )


def check_dest(dest: Path, checkout: Path) -> None:
    """Refuse a destination inside the checkout or this repository (build output stays out)."""

    resolved = dest.expanduser().resolve()
    for root in {checkout.resolve(), REPO.resolve()}:
        if resolved == root or resolved.is_relative_to(root):
            raise BuildRefused(
                f"{dest} is inside {root}. Build the app outside the repository, "
                "for example into ~/Applications (the default)."
            )


def applescript_string(text: str) -> str:
    """text as an AppleScript string literal."""

    escaped = text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


# What the app's dialogs say when something is missing: the text before and after the
# project folder's path, which the applet inserts when it shows the dialog. Dialog text
# says "then" rather than ">" because VoiceOver reads ">" as "greater than".
MISSING = {
    "blocked": (
        "macOS isn't letting Portfolio Lab open its project folder:\n\n",
        (
            "\n\nTo allow it, open System Settings, choose Privacy & Security, then Files "
            "& Folders, and turn on the switch under Portfolio Lab (Desktop Folder, for a "
            "project on the Desktop). Then open Portfolio Lab again."
        ),
    ),
    "missing": (
        "Portfolio Lab can't find its project folder:\n\n",
        (
            "\n\nIf you moved or renamed it, build the app again from the new place. In "
            "Terminal, in the project folder, run:\n\n"
            ".venv/bin/python scripts/build_mac_app.py"
        ),
    ),
    "venv": (
        "The project's Python environment (.venv) is missing in:\n\n",
        (
            "\n\nTo create it again, in Terminal, in the project folder, run:\n\n"
            "python3.12 -m venv .venv\n.venv/bin/pip install -r requirements.txt\n\n"
            "Then open Portfolio Lab again."
        ),
    ),
    "launcher": (
        f"This project folder doesn't have the Lab's launcher ({LAUNCHER.as_posix()}) yet:\n\n",
        (
            "\n\nUpdate it to the latest main (in Terminal, in the project folder: git pull "
            "--ff-only), then open Portfolio Lab again."
        ),
    ),
}
# The sentence macOS shows when it first asks whether the app may read the project folder.
FOLDER_USAGE = "Portfolio Lab runs the Lab from its project folder. It only reads the folder."


APPLESCRIPT = """\
-- Portfolio Lab: opens the local Portfolio Lab like an app.
-- Generated by scripts/build_mac_app.py; rebuild the app instead of editing this copy.

property checkoutFolder : {checkout}
property privateFolder : {private}

on run
	openLab()
end run

on reopen
	openLab()
end reopen

on quit
	stopLab()
	continue quit
end quit

-- "" when the project folder can be read, "blocked" when macOS denies access to it
-- (listing it is what macOS checks), "missing" otherwise.
on folderState()
	try
		do shell script "/bin/ls " & quoted form of checkoutFolder & " >/dev/null"
		return ""
	on error errorText
		if errorText contains "Operation not permitted" then return "blocked"
		return "missing"
	end try
end folderState

-- The dialog text for the first missing piece, or "" when everything is in place.
on missingPart()
	set state to folderState()
	if state is "blocked" then return {blocked_before} & checkoutFolder & {blocked_after}
	if state is "missing" then return {missing_before} & checkoutFolder & {missing_after}
	if not pathPasses("-x", checkoutFolder & "/{venv_python}") then
		return {venv_before} & checkoutFolder & {venv_after}
	end if
	if not pathPasses("-f", checkoutFolder & "/{launcher}") then
		return {launcher_before} & checkoutFolder & {launcher_after}
	end if
	return ""
end missingPart

on pathPasses(test, somePath)
	try
		do shell script "/bin/test " & test & " " & quoted form of somePath
		return true
	on error
		return false
	end try
end pathPasses

-- What the dialog says when the launcher fails: its own message, unless macOS blocked
-- access to the project folder.
on failureText(errorText)
	if errorText contains "Operation not permitted" then
		return {blocked_before} & checkoutFolder & {blocked_after}
	end if
	return errorText
end failureText

on launcherCommand(action)
	set python to quoted form of (checkoutFolder & "/{venv_python}")
	set launcher to quoted form of (checkoutFolder & "/{launcher}")
	return python & " -I -B " & launcher & " " & action
end launcherCommand

on openLab()
	set problem to missingPart()
	if problem is not "" then
		activate
		display alert "Portfolio Lab can't open" message problem as warning buttons {{"OK"}} default button "OK"
		tell me to quit
		return
	end if
	try
		do shell script launcherCommand("open")
	on error errorText
		activate
		set reply to display alert "Portfolio Lab couldn't start" message failureText(errorText) as warning buttons {{"Show Logs", "OK"}} default button "OK"
		if button returned of reply is "Show Logs" then
			try
				do shell script "/usr/bin/open " & quoted form of privateFolder
			on error
				display alert "There are no logs yet" message "Portfolio Lab stopped before it wrote any logs." as warning buttons {{"OK"}} default button "OK"
			end try
		end if
		-- Quitting also stops a server that started but does not answer.
		tell me to quit
	end try
end openLab

on stopLab()
	try
		do shell script launcherCommand("stop")
	end try
end stopLab
"""


def applescript(checkout: Path, private: Path = PRIVATE_DIR) -> str:
    """The applet's source, with the checkout and the private folder recorded in it."""

    messages = {
        f"{part}_{side}": applescript_string(text)
        for part, pair in MISSING.items()
        for side, text in zip(("before", "after"), pair, strict=True)
    }
    return APPLESCRIPT.format(
        checkout=applescript_string(str(checkout)),
        private=applescript_string(str(private)),
        venv_python=VENV_PYTHON.as_posix(),
        launcher=LAUNCHER.as_posix(),
        **messages,
    )


def render_icon(brand: Path, work: Path, runner: Runner = run) -> Path | None:
    """The .icns rendered from the brand SVGs with sips and iconutil, or None without them."""

    full, small = brand / ICON_SVG, brand / SMALL_ICON_SVG
    if not full.is_file():
        return None
    small = small if small.is_file() else full
    iconset = work / "PortfolioLab.iconset"
    iconset.mkdir()
    for pixels, stem in ICONSET:
        source = small if pixels <= SMALL_ICON_MAX else full
        runner(
            ["/usr/bin/sips", "-s", "format", "png", "-z", str(pixels), str(pixels)]
            + [str(source), "--out", str(iconset / f"{stem}.png")]
        )
    icns = work / "PortfolioLab.icns"
    runner(["/usr/bin/iconutil", "-c", "icns", str(iconset), "-o", str(icns)])
    return icns


def configure(bundle: Path, checkout: Path, commit: str, icon: Path | None) -> None:
    """
    Name the applet, record its source and icon, and keep only the privacy prompts it
    can need: reading a project folder in Desktop or Documents.
    """

    info = bundle / "Contents" / "Info.plist"
    with info.open("rb") as handle:
        plist = plistlib.load(handle)
    for key in [k for k in plist if k.startswith("NS") and k.endswith("UsageDescription")]:
        del plist[key]  # the applet uses no camera, contacts, Apple Events to other apps, ...
    plist.update(
        {
            "NSDesktopFolderUsageDescription": FOLDER_USAGE,
            "NSDocumentsFolderUsageDescription": FOLDER_USAGE,
            "CFBundleIdentifier": BUNDLE_ID,
            "CFBundleName": "Portfolio Lab",
            "CFBundleDisplayName": "Portfolio Lab",
            "CFBundleShortVersionString": "1.0",
            "CFBundleVersion": "1",
            "LSApplicationCategoryType": "public.app-category.finance",
            "StockAgentCheckout": str(checkout),
            "StockAgentCommit": commit,
        }
    )
    if icon is not None:
        resources = bundle / "Contents" / "Resources"
        shutil.copyfile(icon, resources / "applet.icns")
        # An asset catalog would take precedence over the .icns.
        (resources / "Assets.car").unlink(missing_ok=True)
        plist.pop("CFBundleIconName", None)
        plist["CFBundleIconFile"] = "applet"
    with info.open("wb") as handle:
        plistlib.dump(plist, handle)


def is_ours(bundle: Path) -> bool:
    try:
        with (bundle / "Contents" / "Info.plist").open("rb") as handle:
            return plistlib.load(handle).get("CFBundleIdentifier") == BUNDLE_ID
    except (OSError, plistlib.InvalidFileException):
        return False


def install(bundle: Path, dest: Path, work: Path) -> Path:
    """Move bundle to dest; an earlier copy of this app goes to work (deleted with it)."""

    target = dest / APP_NAME
    if target.exists() or target.is_symlink():
        if target.is_symlink() or not is_ours(target):
            raise BuildRefused(
                f"{target} exists and is not a Portfolio Lab built by this script; "
                "it was left untouched. Move it away or choose another --dest."
            )
        os.replace(target, work / "replaced.app")
    os.replace(bundle, target)
    return target


def build(
    checkout: Path,
    dest: Path,
    *,
    runner: Runner = run,
    brand: Path | None = None,
    private: Path = PRIVATE_DIR,
    register: bool = True,
) -> Path:
    """
    Build the app for checkout and install it into dest; returns the installed bundle.
    register asks Launch Services to read the new bundle, so Finder and the Dock show its
    icon at once (tests turn it off).
    """

    checkout = checkout.expanduser().resolve()
    dest = dest.expanduser()
    check_checkout(checkout)
    check_dest(dest, checkout)
    dest.mkdir(parents=True, exist_ok=True)
    brand = brand if brand is not None else checkout / BRAND
    with tempfile.TemporaryDirectory(prefix=".portfolio-lab-build-", dir=dest) as folder:
        work = Path(folder)
        source = work / "Portfolio Lab.applescript"
        source.write_text(applescript(checkout, private), encoding="utf-8")
        bundle = work / APP_NAME
        runner(["/usr/bin/osacompile", "-s", "-o", str(bundle), str(source)])
        configure(bundle, checkout, commit_of(checkout), render_icon(brand, work, runner))
        runner(["/usr/bin/codesign", "--force", "--sign", "-", str(bundle)])
        runner(["/usr/bin/codesign", "--verify", "--strict", str(bundle)])
        target = install(bundle, dest, work)
    if register and LSREGISTER.exists():  # best effort
        subprocess.run([str(LSREGISTER), "-f", str(target)], capture_output=True, check=False)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="folder to install into")
    parser.add_argument(
        "--checkout", type=Path, default=None, help="project folder the app runs the Lab from"
    )
    args = parser.parse_args(argv)
    if sys.platform != "darwin":
        print("The Portfolio Lab app is built on macOS only.", file=sys.stderr)
        return 2
    checkout = args.checkout or default_checkout()
    try:
        target = build(checkout, args.dest)
    except BuildRefused as reason:
        print(reason, file=sys.stderr)
        return 2
    print(f"Built {target}")
    print(f"It runs the Lab from {checkout.expanduser().resolve()}.")
    if protected(checkout):
        print(
            "macOS will ask Portfolio Lab for access to that folder, and again after each "
            f"rebuild. To avoid it, build from the runner checkout ({RUNNER}).",
            file=sys.stderr,
        )
    else:
        print("That folder is outside Desktop, Documents and Downloads, so macOS won't ask.")
    print(
        "Open it with Spotlight (Cmd-Space, type Portfolio Lab, press Return). To keep it in "
        "the Dock, Control-click its Dock icon while it runs and choose Options, then Keep in Dock."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
