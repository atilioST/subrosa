# Subrosa — rebuilding on a new machine

Everything needed to get Subrosa running from a freshly formatted Linux box.
The repo holds the code; the secrets and state live outside it, so a backup has to
be taken **before** formatting (step 0).

## Assumptions baked into the code

- Linux user is **`ati`**, home `/home/ati`. Paths are hardcoded in
  `systemd/*.service`, `scripts/auth_watchdog.sh`, and `CLI_PATH` in `agent.py`,
  `memory.py`, `procedures.py`, `distiller.py`, `brain_server.py`. A different
  username means editing all of those.
- Repo at `~/Dev/subrosa`, MCP servers at `~/Dev/mcp/<name>`.
- Claude Code CLI at `~/.local/bin/claude` (the native installer puts it there).
- MCP servers are not in `~/.subrosa/mcp.json`. The agent uses
  `setting_sources=["user"]`, so it picks up the servers registered in Claude Code's
  user config (`~/.claude.json`).

## 0. Before formatting the old machine

```bash
cd ~/Dev/subrosa
scripts/backup_state.sh            # -> ~/subrosa-backup-<date>/subrosa-state.tar.gz
```

Copy the archive off the machine to a USB drive or a password manager attachment.
**Don't put it in git or a shared drive.** It contains the Telegram bot token,
Langfuse keys, the brain token, the Jira OAuth client secret, and the Slack/Jira
tokens.

The archive contains:

| Path in archive | What it is | Needed? |
|---|---|---|
| `subrosa/.subrosa/.env` | Secrets (Telegram, Langfuse, brain token, Jira OAuth client) | **Required** |
| `subrosa/.subrosa/config.toml` | App config (channels, models, schedule, people) | **Required** |
| `subrosa/subrosa.db` | SQLite DB snapshot: memories, events, brain knowledge, history | **Required**, or Subrosa starts with no memory |
| `subrosa/.subrosa/briefing.md`, `procedures/`, `uploads/` | User context doc, learned procedures, uploads | Recommended |
| `home.tar` → `.slack-mcp-tokens.json`, `.jira-mcp-tokens.json` | Slack xoxp token and Jira OAuth tokens. The hourly scan reads these directly | Recommended (otherwise you redo OAuth) |
| `home.tar` → `.jira-mcp-compass/`, `.chorus-mcp/` | MCP caches/context | Nice to have |
| `home.tar` → `.claude.json` | Claude Code config, including the MCP server registrations and `CHORUS_API_KEY` | Reference. Re-register instead of copying it wholesale (see step 4) |
| `home.tar` → `.claude/projects/-home-ati/memory/` | Claude Code's memory notes about this setup | Nice to have |
| `git/*.patch`, `git/*.bundle` | Uncommitted diffs and unpushed commits in subrosa + MCP repos | **Required if present** |

Also confirm nothing is stranded in git:

```bash
for d in ~/Dev/subrosa ~/Dev/mcp/*; do echo "== $d"; git -C $d status -s; git -C $d log --branches --not --remotes --oneline; done
```

> **slack-mcp runs from branch `feat/channel-read-state`** (pushed to
> `sitetracker/slack-mcp`, not merged to main). It adds `last_read`/`unread_count`
> on channel info, which the hourly scan's unread detection depends on.
> `local/attachment-fix` holds two older experimental commits; it isn't used.

## 1. System packages (Arch)

```bash
sudo pacman -S --needed git python sqlite curl base-devel
curl -fsSL https://claude.ai/install.sh | bash     # Claude Code -> ~/.local/bin/claude
curl -LsSf https://astral.sh/uv/install.sh | sh    # optional, faster venvs
```

Python 3.11+ is required (this machine ran 3.14).

## 2. Clone repos

```bash
mkdir -p ~/Dev/mcp && cd ~/Dev
git clone https://github.com/atilioST/subrosa.git
cd mcp
git clone https://github.com/sitetracker/slack-mcp.git
git clone https://github.com/sitetracker/jira-mcp-compass.git
git clone https://github.com/sitetracker/chorus-mcp.git
git clone https://github.com/sitetracker/internal-salesforce-mcp-server.git   # optional
```

Check out the branch Subrosa needs:

```bash
git -C ~/Dev/mcp/slack-mcp checkout feat/channel-read-state
```

Apply any `git/<repo>.uncommitted.patch` with `git apply`.

## 3. Restore state

```bash
mkdir -p ~/restore && tar -C ~/restore -xzf subrosa-state.tar.gz
cp -a ~/restore/subrosa/.subrosa ~/
cp ~/restore/subrosa/subrosa.db ~/.subrosa/subrosa.db
tar -C ~ -xf ~/restore/home.tar \
    .slack-mcp-tokens.json .jira-mcp-tokens.json .jira-mcp-compass .chorus-mcp \
    .claude/projects/-home-ati/memory
chmod 600 ~/.subrosa/.env ~/.slack-mcp-tokens.json ~/.jira-mcp-tokens.json
```

## 4. Claude Code login + MCP servers

```bash
claude            # then /login with the subscription account
```

Build each MCP venv. They break silently when system Python is upgraded, so
rebuild them after any Python upgrade too:

```bash
cd ~/Dev/mcp/slack-mcp        && python -m venv .venv && .venv/bin/pip install -e .
cd ~/Dev/mcp/jira-mcp-compass && python -m venv .venv && .venv/bin/pip install -e . 'mcp==1.26.0'
cd ~/Dev/mcp/chorus-mcp       && python -m venv .venv && .venv/bin/pip install -e .
cd ~/Dev/mcp/internal-salesforce-mcp-server && python -m venv venv && venv/bin/pip install -e .
```

(`jira-mcp-compass` needs `mcp==1.26.0`: mcp 2.x removed `Server.list_tools`.)

Register them at user scope. These match what ran on the old machine. Take
`CHORUS_API_KEY` from `.claude.json` in the backup.

```bash
claude mcp add -s user slack -- ~/Dev/mcp/slack-mcp/.venv/bin/python -m slack_mcp
claude mcp add -s user jira  -- ~/Dev/mcp/jira-mcp-compass/.venv/bin/python -m jira_mcp.server
claude mcp add -s user chorus -e CHORUS_API_KEY=... -- ~/Dev/mcp/chorus-mcp/.venv/bin/python -m chorus_mcp
claude mcp add -s user salesforce -- ~/Dev/mcp/internal-salesforce-mcp-server/venv/bin/python \
    ~/Dev/mcp/internal-salesforce-mcp-server/src/salesforce_mcp/server.py
claude mcp add -s user --transport http subrosa-brain http://127.0.0.1:7771/mcp
```

If the token files weren't restored, run `claude`, then `/mcp` (or call
`slack_authorize` / `jira_authorize`) to redo OAuth. That recreates
`~/.slack-mcp-tokens.json` and `~/.jira-mcp-tokens.json`.

## 5. Subrosa venv

```bash
cd ~/Dev/subrosa
python -m venv .venv
.venv/bin/pip install -e . pytest pytest-asyncio
.venv/bin/python -m pytest -q          # sanity check
```

The first start downloads the sentence-transformers embedding model (~100 MB).

## 6. systemd (user units)

```bash
mkdir -p ~/.config/systemd/user
cp ~/Dev/subrosa/systemd/subrosa.service \
   ~/Dev/subrosa/systemd/subrosa-watchdog.service \
   ~/Dev/subrosa/systemd/subrosa-watchdog.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now subrosa.service subrosa-watchdog.timer
sudo loginctl enable-linger ati      # keep running when logged out / after reboot without login
```

(The old machine had linger **off**, so Subrosa only ran while you were logged in.)

## 7. Verify

```bash
systemctl --user status subrosa
journalctl --user -u subrosa -f              # watch startup; no CLIError / auth errors
systemctl --user start subrosa-watchdog      # one probe; should NOT send a Telegram alert
claude mcp list                              # all servers ✓ connected
```

Then send `/status` to the bot on Telegram, and later check that the next hourly
scan runs cleanly in the journal. Turn the old machine's service off
(`systemctl --user disable --now subrosa`) before the new one starts. Two bots
polling the same token will conflict.

## Troubleshooting

- `Command failed with exit code 1` / `CLIError: authentication_failed`: the
  Claude CLI is logged out or the OAuth session expired. Run `claude`, then `/login`.
- `[claude-code:unrecognized_model]`: the CLI is older than the model in
  `config.toml`. Update with `claude update`.
- MCP server shows "Connection closed": its venv is broken. Rebuild it (step 4).
