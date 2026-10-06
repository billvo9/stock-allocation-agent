#!/usr/bin/env bash
# SessionStart hook: inject a short, factual Git snapshot.
#
# Read-only. Local git only (no fetch, no network). Always exits 0.
# stdout is added to Claude's context, so keep it brief.

set -u

cd "${CLAUDE_PROJECT_DIR:-.}" 2>/dev/null || exit 0
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

branch=$(git branch --show-current 2>/dev/null)
head=$(git log -1 --format='%h %s' 2>/dev/null | cut -c1-100)

upstream=$(git rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>/dev/null)
if [ -n "$upstream" ]; then
    counts=$(git rev-list --left-right --count 'HEAD...@{u}' 2>/dev/null)
    ahead=${counts%%[[:space:]]*}
    behind=${counts##*[[:space:]]}
    upstream_line="$upstream (ahead $ahead, behind $behind; as of last fetch)"
else
    upstream_line="none"
fi

status=$(git status --short 2>/dev/null)
status_count=$(printf '%s' "$status" | grep -c . || true)

echo "## Git snapshot at session start (local refs; not fetched)"
echo "- branch: ${branch:-<detached>}"
echo "- HEAD: $head"
echo "- upstream: $upstream_line"
if [ "$status_count" -eq 0 ]; then
    echo "- working tree: clean"
else
    echo "- working tree: $status_count changed path(s)"
    printf '%s\n' "$status" | head -n 20 | sed 's/^/    /'
    if [ "$status_count" -gt 20 ]; then
        echo "    ... $((status_count - 20)) more"
    fi
fi
echo "- recent commits:"
git log --oneline -5 2>/dev/null | cut -c1-100 | sed 's/^/    /'

if [ -x .venv/bin/python ]; then
    venv_version=$(.venv/bin/python -c 'import sys; print(*sys.version_info[:3], sep=".")' 2>/dev/null)
    echo "- project venv: .venv (Python ${venv_version:-unknown}); gates: python3 scripts/verify.py"
else
    echo "- project venv: MISSING (.venv/bin/python); scripts/verify.py will refuse to run"
fi

reviewed=$(grep -m1 '^Last reviewed:' docs/PROJECT_STATE.md 2>/dev/null)
if [ -n "$reviewed" ]; then
    echo "- docs/PROJECT_STATE.md $reviewed"
fi
echo "Git is authoritative over docs/PROJECT_STATE.md and memory."

exit 0
