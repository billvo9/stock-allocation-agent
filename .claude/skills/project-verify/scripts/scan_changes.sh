#!/usr/bin/env bash
# Read-only scope / artifact / secret scan for /project-verify.
#
# Usage: scan_changes.sh [base-ref]   (default: origin/main)
#
# Scans every path changed since merge-base(HEAD, base): committed, staged,
# unstaged, and untracked (non-ignored). Reports findings; never modifies
# anything and always exits 0. Secret matches print file:line and the
# pattern name only - never the matched text.

set -u

base="${1:-origin/main}"

root=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "scan: not inside a git repository"
    exit 0
}
cd "$root" || exit 0

if mb=$(git merge-base HEAD "$base" 2>/dev/null); then
    echo "scan base: $base (merge-base $(git rev-parse --short "$mb"))"
else
    mb=HEAD
    echo "scan base: $base not found locally; scanning working-tree changes vs HEAD only"
fi

changed=$(
    {
        git diff --name-only "$mb"
        git diff --name-only --cached
        git ls-files --others --exclude-standard
    } 2>/dev/null | sort -u
)

count=$(printf '%s' "$changed" | grep -c . || true)
echo "changed paths: $count"
printf '%s\n' "$changed" | grep . | sed 's/^/  /'

findings=0
report() {
    findings=$((findings + 1))
    echo "  [$1] $2"
}

echo
echo "== generated data / artifacts / caches / local config =="
artifact_re='(\.(parquet|csv|pkl|pickle|feather|h5|hdf5|zip|tar|gz|pt|pth|onnx|joblib|npy|npz|db|sqlite|duckdb|log)$)|(^|/)(data/raw|data/processed|reports|models|__pycache__|\.pytest_cache|\.ruff_cache|\.ipynb_checkpoints|[^/]*\.egg-info)/|(^|/)\.DS_Store$|(^|/)\.env($|\.)|(^|/)\.claude/settings\.local\.json$'
while IFS= read -r path; do
    [ -n "$path" ] || continue
    if printf '%s' "$path" | grep -Eq "$artifact_re"; then
        report artifact "$path"
    fi
done <<< "$changed"

echo
echo "== large files (> 1 MB) =="
while IFS= read -r path; do
    [ -f "$path" ] || continue
    size=$(wc -c < "$path" | tr -d ' ')
    if [ "$size" -gt 1048576 ]; then
        report large "$path ($size bytes)"
    fi
done <<< "$changed"

echo
echo "== tracked files that match .gitignore =="
git ls-files -ci --exclude-standard 2>/dev/null | while IFS= read -r path; do
    echo "  [tracked-but-ignored] $path"
done

echo
echo "== possible secrets / identities in changed text files =="
secret_patterns=(
    "aws-access-key|AKIA[0-9A-Z]{16}"
    "private-key|-----BEGIN [A-Z ]*PRIVATE KEY-----"
    "assigned-credential|(api[_-]?key|secret|password|passwd|token)[\"']?[[:space:]]*[:=][[:space:]]*[\"'][^\"']{8,}[\"']"
    "edgar-identity|EDGAR_IDENTITY[[:space:]]*="
    "github-token|gh[pousr]_[A-Za-z0-9]{30,}"
    "email-address|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
while IFS= read -r path; do
    [ -f "$path" ] || continue
    for entry in "${secret_patterns[@]}"; do
        name=${entry%%|*}
        regex=${entry#*|}
        # -I skips binary files; print only file:line, never content.
        grep -InEo "$regex" "$path" 2>/dev/null | cut -d: -f1 | sort -un | while IFS= read -r line; do
            echo "  [$name] $path:$line"
        done
    done
done <<< "$changed"

echo
echo "scan complete (artifact/large findings: $findings). Review every listed item; this scan is heuristic."
exit 0
