#!/usr/bin/env bash
# Subrosa machine-state backup — everything a rebuild needs that is NOT in git.
# See docs/SETUP.md. Output contains secrets: keep it off GitHub and off shared drives.
#
#   scripts/backup_state.sh [output-dir]      # default: ~/subrosa-backup-<date>
set -euo pipefail

OUT="${1:-$HOME/subrosa-backup-$(date +%Y%m%d-%H%M)}"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$OUT"

# Consistent DB snapshot (service may be running; WAL holds uncheckpointed writes).
mkdir -p "$STAGE/subrosa"
sqlite3 "$HOME/.subrosa/subrosa.db" ".backup '$STAGE/subrosa/subrosa.db'"

# ~/.subrosa minus the live DB files (snapshotted above) and old backups.
tar -C "$HOME" -cf - \
    --exclude='.subrosa/subrosa.db' --exclude='.subrosa/subrosa.db-*' \
    --exclude='.subrosa/subrosa_backup_*.db' \
    .subrosa | tar -C "$STAGE/subrosa" -xf -

# MCP token stores + caches, Claude Code memory/config, misc home state.
paths=(
    .slack-mcp-tokens.json
    .jira-mcp-tokens.json
    .jira-mcp-compass
    .chorus-mcp
    .claude.json
    .claude/settings.json
    .claude/settings.local.json
    .claude/projects/-home-ati/memory
    .config/subrosa
    .aws/config
)
existing=()
for p in "${paths[@]}"; do [[ -e "$HOME/$p" ]] && existing+=("$p"); done
tar -C "$HOME" -cf "$STAGE/home.tar" "${existing[@]}"

# Uncommitted / unpushed git work in the repos Subrosa depends on.
mkdir -p "$STAGE/git"
for repo in "$HOME/Dev/subrosa" "$HOME"/Dev/mcp/*; do
    [[ -d "$repo/.git" ]] || continue
    name="$(basename "$repo")"
    git -C "$repo" diff HEAD > "$STAGE/git/$name.uncommitted.patch"
    [[ -s "$STAGE/git/$name.uncommitted.patch" ]] || rm "$STAGE/git/$name.uncommitted.patch"
    if [[ -n "$(git -C "$repo" log --branches --not --remotes --oneline)" ]]; then
        git -C "$repo" bundle create "$STAGE/git/$name.unpushed.bundle" --branches --not --remotes
    fi
done

ARCHIVE="$OUT/subrosa-state.tar.gz"
tar -C "$STAGE" -czf "$ARCHIVE" .
chmod 600 "$ARCHIVE"
echo "Wrote $ARCHIVE ($(du -h "$ARCHIVE" | cut -f1))"
tar -tzf "$ARCHIVE" | sed 's/^/  /' | grep -v '/$' | head -50
