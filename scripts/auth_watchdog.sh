#!/usr/bin/env bash
# Subrosa auth watchdog — probes the Claude CLI and alerts on Telegram when
# agent invocations would fail (expired subscription / logged-out CLI / broken install).
#
# Runs via subrosa-watchdog.timer. State in ~/.subrosa/watchdog_state:
#   ok                      — last probe succeeded
#   fail <epoch-of-last-alert>  — broken; re-alerts every REALERT_SECONDS
set -u

CLI="/home/ati/.local/bin/claude"
ENV_FILE="$HOME/.subrosa/.env"
STATE_FILE="${WATCHDOG_STATE_FILE:-$HOME/.subrosa/watchdog_state}"
CHAT_ID="${WATCHDOG_CHAT_ID:-8563048012}"  # same chat as Subrosa (config.toml allowed_chat_ids)
REALERT_SECONDS=$((6 * 3600)) # re-alert every 6h while still broken
PROBE_TIMEOUT=120

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

# Returns non-zero (and logs to the journal) if Telegram did not accept it,
# so a failed alert is visible instead of silently lost.
send_telegram() {
    local resp
    resp=$(curl -sS -m 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$CHAT_ID" \
        --data-urlencode text="$1" 2>&1)
    if [[ "$resp" != *'"ok":true'* ]]; then
        echo "watchdog: Telegram send FAILED: ${resp:0:300}" >&2
        return 1
    fi
    echo "watchdog: alert sent"
}

# Cheap probe: one haiku turn, same invocation style as the agent SDK uses.
# The CLI prints API errors ("Failed to authenticate: …") on STDOUT, so keep
# both streams for the alert details. </dev/null skips the 3s stdin wait.
probe_out=$(timeout "$PROBE_TIMEOUT" "$CLI" -p "reply with the word ok" \
    --model haiku --max-turns 1 </dev/null 2>&1)
probe_rc=$?

prev_state=$(cut -d' ' -f1 "$STATE_FILE" 2>/dev/null || echo "ok")
prev_alert=$(cut -d' ' -f2 "$STATE_FILE" 2>/dev/null || echo 0)
now=$(date +%s)

if [ "$probe_rc" -eq 0 ]; then
    echo "watchdog: probe ok"
    if [ "$prev_state" = "fail" ]; then
        send_telegram "✅ Subrosa watchdog: Claude CLI is working again. Agent runs should recover on their own."
    fi
    echo "ok" > "$STATE_FILE"
    exit 0
fi

# Probe failed
detail=$(printf '%s' "$probe_out" | grep -v "no stdin data received" | tail -c 400)
echo "watchdog: probe FAILED (exit ${probe_rc}): ${detail:-no output}" >&2
if [[ "$probe_out" == *"Failed to authenticate"* ]]; then
    headline="Claude login has expired — Subrosa can't run anything until you re-login."
else
    headline="Claude CLI probe failed (exit ${probe_rc}) — agent runs are likely failing. Check login / subscription."
fi
if [ "$prev_state" != "fail" ] || [ $((now - prev_alert)) -ge "$REALERT_SECONDS" ]; then
    if send_telegram "🚨 Subrosa watchdog: ${headline}

Run on SubrosaBox: claude /login

Details: ${detail:-no output}"; then
        echo "fail $now" > "$STATE_FILE"
    else
        # alert didn't go out — keep the old timestamp so the next run retries
        echo "fail $prev_alert" > "$STATE_FILE"
    fi
else
    # still broken, within re-alert window — keep original alert timestamp
    echo "fail $prev_alert" > "$STATE_FILE"
fi
exit 0
