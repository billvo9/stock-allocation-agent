"""
Building "Portfolio Lab.app" (scripts/build_mac_app.py): the applet's source,
the checkout and destination checks, the Info.plist and icon, replacing only
its own earlier build, and nothing written inside the checkout. The tests that
need macOS's own tools (osacompile, codesign, sips, iconutil, osascript) run
only on macOS.
"""

from __future__ import annotations

import importlib.util
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "stock_agent_build_mac_app", REPO / "scripts" / "build_mac_app.py"
)
builder = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = builder
_spec.loader.exec_module(builder)

on_macos = pytest.mark.skipif(
    sys.platform != "darwin" or shutil.which("osacompile") is None,
    reason="needs macOS's osacompile, codesign and osascript",
)
TINY_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="64" height="64">'
    '<rect width="64" height="64" fill="#9A4325"/></svg>\n'
)


def checkout(root: Path) -> Path:
    """A project folder with everything the app needs."""

    folder = root / "project"
    for part in (builder.VENV_PYTHON, builder.LAUNCHER, builder.LAB_SCRIPT):
        (folder / part).parent.mkdir(parents=True, exist_ok=True)
        (folder / part).write_text("", encoding="utf-8")
    (folder / builder.VENV_PYTHON).chmod(0o755)
    return folder


def fake_bundle(bundle: Path, **extra) -> None:
    """What osacompile leaves: an applet with its usage prompts and asset-catalog icon."""

    resources = bundle / "Contents" / "Resources"
    resources.mkdir(parents=True)
    (resources / "Assets.car").write_bytes(b"catalog")
    (resources / "applet.icns").write_bytes(b"applet icon")
    plist = {
        "CFBundleExecutable": "applet",
        "CFBundleIconFile": "applet",
        "CFBundleIconName": "applet",
        "CFBundleName": "Portfolio Lab",
        "NSCameraUsageDescription": "This script needs access to your camera to run.",
        "NSAppleEventsUsageDescription": "This script needs to control other applications.",
        **extra,
    }
    with (bundle / "Contents" / "Info.plist").open("wb") as handle:
        plistlib.dump(plist, handle)


class FakeTools:
    """osacompile, sips, iconutil and codesign as recorders; osacompile makes a bundle."""

    def __init__(self):
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str]) -> None:
        self.commands.append(command)
        tool = Path(command[0]).name
        if tool == "osacompile":
            fake_bundle(Path(command[command.index("-o") + 1]))
        elif tool in {"sips", "iconutil"}:
            Path(command[-1]).write_bytes(b"icns" if tool == "iconutil" else b"png")


def plist_of(bundle: Path) -> dict:
    with (bundle / "Contents" / "Info.plist").open("rb") as handle:
        return plistlib.load(handle)


def snapshot(folder: Path) -> dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(folder)): (path.lstat().st_size, path.lstat().st_mtime_ns)
        for path in sorted(folder.rglob("*"))
    }


# --- the applet's source --------------------------------------------------------------


def test_applescript_strings_escape_quotes_backslashes_and_newlines():
    assert builder.applescript_string('a "b" \\c\nd') == '"a \\"b\\" \\\\c\\nd"'


def test_the_applet_records_the_checkout_and_runs_the_launcher_isolated(tmp_path):
    folder = tmp_path / 'odd "name" \\ folder'
    source = builder.applescript(folder, tmp_path / "private")
    assert f"property checkoutFolder : {builder.applescript_string(str(folder))}" in source
    assert 'return python & " -I -B " & launcher & " " & action' in source
    assert 'set launcher to quoted form of (checkoutFolder & "/scripts/portfolio_lab_app.py")' in (
        source
    )
    # Opening again only shows the Lab; quitting stops it before the app exits.
    assert "on reopen\n\topenLab()" in source
    assert "on quit\n\tstopLab()\n\tcontinue quit" in source
    assert 'launcherCommand("stop")' in source
    assert set(builder.MISSING) == {"blocked", "missing", "venv", "launcher"}
    for before, after in builder.MISSING.values():
        assert builder.applescript_string(before) in source
        assert builder.applescript_string(after) in source


def test_the_builder_and_the_launcher_agree_on_paths():
    _launcher_spec = importlib.util.spec_from_file_location(
        "stock_agent_portfolio_lab_app_paths", REPO / builder.LAUNCHER
    )
    launcher = importlib.util.module_from_spec(_launcher_spec)
    sys.modules[_launcher_spec.name] = launcher
    _launcher_spec.loader.exec_module(launcher)

    assert builder.PRIVATE_DIR == launcher.PRIVATE_DIR
    places = launcher.Places(REPO, launcher.PRIVATE_DIR)
    assert places.python == REPO / builder.VENV_PYTHON
    assert places.lab_script == REPO / builder.LAB_SCRIPT


# --- checks ---------------------------------------------------------------------------


def test_a_destination_inside_the_checkout_or_the_repository_is_refused(tmp_path):
    folder = checkout(tmp_path)
    for dest in (folder, folder / "dist", REPO / "build"):
        with pytest.raises(builder.BuildRefused, match="outside the repository"):
            builder.check_dest(dest, folder)
    builder.check_dest(tmp_path / "Applications", folder)


@pytest.mark.parametrize(
    ("part", "reason"),
    [
        (builder.VENV_PYTHON, "has no .venv"),
        (builder.LAUNCHER, "has no scripts/portfolio_lab_app.py"),
        (builder.LAB_SCRIPT, "has no scripts/run_portfolio_lab.py"),
    ],
)
def test_a_checkout_missing_a_part_is_refused(tmp_path, part, reason):
    folder = checkout(tmp_path)
    (folder / part).unlink()
    with pytest.raises(builder.BuildRefused, match=reason):
        builder.check_checkout(folder)


def test_the_main_working_tree_is_found_from_a_linked_worktree(tmp_path):
    repo, linked = tmp_path / "repo", tmp_path / "linked"
    git = ["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid"]
    subprocess.run([*git, "init", "-q", str(repo)], check=True)
    subprocess.run([*git, "-C", str(repo), "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    subprocess.run([*git, "-C", str(repo), "worktree", "add", "-q", str(linked)], check=True)
    assert builder.main_worktree(linked) == repo.resolve()
    assert builder.main_worktree(repo) == repo.resolve()


# --- Info.plist, icon and install -----------------------------------------------------


def test_configure_names_the_app_records_its_source_and_drops_unused_prompts(tmp_path):
    bundle, icon = tmp_path / "Portfolio Lab.app", tmp_path / "PortfolioLab.icns"
    fake_bundle(bundle)
    icon.write_bytes(b"our icon")

    builder.configure(bundle, Path("/projects/lab"), "abc1234", icon)

    plist = plist_of(bundle)
    assert plist["CFBundleIdentifier"] == builder.BUNDLE_ID
    assert plist["CFBundleName"] == plist["CFBundleDisplayName"] == "Portfolio Lab"
    assert plist["StockAgentCheckout"] == "/projects/lab"
    assert plist["StockAgentCommit"] == "abc1234"
    prompts = {key for key in plist if key.endswith("UsageDescription")}
    assert prompts == {"NSDesktopFolderUsageDescription", "NSDocumentsFolderUsageDescription"}
    assert plist["NSDesktopFolderUsageDescription"] == builder.FOLDER_USAGE
    assert "CFBundleIconName" not in plist and plist["CFBundleIconFile"] == "applet"
    resources = bundle / "Contents" / "Resources"
    assert not (resources / "Assets.car").exists()
    assert (resources / "applet.icns").read_bytes() == b"our icon"


def test_without_brand_artwork_the_applet_keeps_its_own_icon(tmp_path):
    bundle, tools = tmp_path / "Portfolio Lab.app", FakeTools()
    fake_bundle(bundle)
    icon = builder.render_icon(tmp_path / "no-brand", tmp_path, tools)
    builder.configure(bundle, Path("/projects/lab"), "abc1234", icon)

    assert icon is None and tools.commands == []
    assert plist_of(bundle)["CFBundleIconName"] == "applet"
    assert (bundle / "Contents" / "Resources" / "Assets.car").exists()


def test_the_icon_uses_the_small_artwork_up_to_32_pixels(tmp_path):
    brand, tools = tmp_path / "brand", FakeTools()
    brand.mkdir()
    (brand / builder.ICON_SVG).write_text(TINY_SVG, encoding="utf-8")
    (brand / builder.SMALL_ICON_SVG).write_text(TINY_SVG, encoding="utf-8")

    icns = builder.render_icon(brand, tmp_path, tools)

    renders = [command for command in tools.commands if Path(command[0]).name == "sips"]
    assert len(renders) == 10
    for command in renders:
        pixels, source = int(command[5]), Path(command[7]).name
        expected = builder.SMALL_ICON_SVG if pixels <= 32 else builder.ICON_SVG
        assert source == expected
    assert Path(tools.commands[-1][0]).name == "iconutil"
    assert icns == tmp_path / "PortfolioLab.icns"


def test_install_replaces_an_earlier_build_but_never_another_app(tmp_path):
    dest, work = tmp_path / "Applications", tmp_path / "work"
    dest.mkdir()
    work.mkdir()
    other = dest / builder.APP_NAME
    fake_bundle(other, CFBundleIdentifier="com.example.other")

    new = work / "new.app"
    fake_bundle(new, CFBundleIdentifier=builder.BUNDLE_ID)
    with pytest.raises(builder.BuildRefused, match="left untouched"):
        builder.install(new, dest, work)
    assert plist_of(other)["CFBundleIdentifier"] == "com.example.other"

    with (other / "Contents" / "Info.plist").open("wb") as handle:  # now an earlier build
        plistlib.dump({"CFBundleIdentifier": builder.BUNDLE_ID}, handle)
    assert builder.install(new, dest, work) == dest / builder.APP_NAME
    assert (work / "replaced.app").exists()
    assert not new.exists()


def test_a_build_writes_only_the_app_into_the_destination(tmp_path):
    folder, dest, tools = checkout(tmp_path), tmp_path / "Applications", FakeTools()
    before = snapshot(folder)

    target = builder.build(folder, dest, runner=tools, register=False)

    assert target == dest / builder.APP_NAME
    assert sorted(path.name for path in dest.iterdir()) == [builder.APP_NAME]  # no build folder
    assert snapshot(folder) == before
    assert plist_of(target)["StockAgentCheckout"] == str(folder.resolve())
    tools_used = [Path(command[0]).name for command in tools.commands]
    assert tools_used == ["osacompile", "codesign", "codesign"]  # no brand artwork here


# --- with macOS's own tools -----------------------------------------------------------


def call_handler(script: Path, checkout_folder: Path, call: str) -> str:
    """Run one handler of the compiled applet for another project folder, without any UI."""

    result = subprocess.run(
        [
            "osascript",
            "-e",
            f"set lab to load script POSIX file {builder.applescript_string(str(script))}",
            "-e",
            f"set checkoutFolder of lab to {builder.applescript_string(str(checkout_folder))}",
            "-e",
            f"tell lab to {call}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.rstrip("\n")


@on_macos
def test_the_real_applet_compiles_signs_and_names_each_missing_part(tmp_path):
    folder, dest = checkout(tmp_path), tmp_path / "Applications"
    target = builder.build(folder, dest, register=False)
    script = target / "Contents" / "Resources" / "Scripts" / "main.scpt"

    def missing_part(checkout_folder: Path) -> str:
        return call_handler(script, checkout_folder, "missingPart()")

    assert missing_part(folder.resolve()) == ""
    gone = tmp_path / "moved away"
    assert missing_part(gone).startswith("Portfolio Lab can't find its project folder")
    assert str(gone) in missing_part(gone)
    (folder / builder.LAUNCHER).unlink()
    assert "doesn't have the Lab's launcher" in missing_part(folder.resolve())
    (folder / builder.VENV_PYTHON).unlink()
    assert "Python environment (.venv) is missing" in missing_part(folder.resolve())

    plist = plist_of(target)
    assert plist["OSAAppletStayOpen"] is True  # stays open, so quitting can stop the Lab
    assert plist["CFBundleIdentifier"] == builder.BUNDLE_ID
    subprocess.run(["codesign", "--verify", "--strict", str(target)], check=True)


@on_macos
def test_the_real_applet_quotes_any_folder_name_and_explains_blocked_access(tmp_path):
    folder = checkout(tmp_path / 'it\'s a "quoted" $(folder) name')
    target = builder.build(folder, tmp_path / "Applications", register=False)
    script = target / "Contents" / "Resources" / "Scripts" / "main.scpt"

    command = call_handler(script, folder.resolve(), 'launcherCommand("open")')
    words = subprocess.run(
        ["/bin/sh", "-c", f"for word in {command}; do printf '%s\\n' \"$word\"; done"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    assert words == [
        str(folder.resolve() / builder.VENV_PYTHON),
        "-I",
        "-B",
        str(folder.resolve() / builder.LAUNCHER),
        "open",
    ]

    denied = call_handler(
        script, folder.resolve(), 'failureText("[Errno 1] Operation not permitted")'
    )
    assert denied.startswith("macOS isn't letting Portfolio Lab open its project folder")
    other = call_handler(script, folder.resolve(), 'failureText("The Lab stopped while starting.")')
    assert other == "The Lab stopped while starting."


@on_macos
def test_the_real_icon_is_rendered_with_sips_and_iconutil(tmp_path):
    brand = tmp_path / "brand"
    brand.mkdir()
    (brand / builder.ICON_SVG).write_text(TINY_SVG, encoding="utf-8")
    icns = builder.render_icon(brand, tmp_path)
    assert icns.read_bytes()[:4] == b"icns"
