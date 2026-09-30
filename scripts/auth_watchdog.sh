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
STATE_FILE="$HOME/.subrosa/watchdog_state"
CHAT_ID="8563048012"          # same chat as Subrosa (config.toml allowed_chat_ids)
REALERT_SECONDS=$((6 * 3600)) # re-alert every 6h while still broken
PROBE_TIMEOUT=120

# shellcheck disable=SC1090
set -a; source "$ENV_FILE"; set +a

send_telegram() {
    curl -sS -m 20 "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
        -d chat_id="$CHAT_ID" \
        --data-urlencode text="$1" >/dev/null
}

# Cheap probe: one haiku turn, same invocation style as the agent SDK uses.
stderr_out=$(timeout "$PROBE_TIMEOUT" "$CLI" -p "reply with the word ok" \
    --model haiku --max-turns 1 2>&1 >/dev/null)
probe_rc=$?

prev_state=$(cut -d' ' -f1 "$STATE_FILE" 2>/dev/null || echo "ok")
prev_alert=$(cut -d' ' -f2 "$STATE_FILE" 2>/dev/null || echo 0)
now=$(date +%s)

if [ "$probe_rc" -eq 0 ]; then
    if [ "$prev_state" = "fail" ]; then
        send_telegram "✅ Subrosa watchdog: Claude CLI is working again. Agent runs should recover on their own."
    fi
    echo "ok" > "$STATE_FILE"
    exit 0
fi

# Probe failed
detail=$(printf '%s' "$stderr_out" | tail -c 400)
if [ "$prev_state" != "fail" ] || [ $((now - prev_alert)) -ge "$REALERT_SECONDS" ]; then
    send_telegram "🚨 Subrosa watchdog: Claude CLI probe failed (exit ${probe_rc}). Agent invocations are likely failing — you may need to re-login or check the subscription.

Run: claude /login

Details: ${detail:-no output}"
    echo "fail $now" > "$STATE_FILE"
else
    # still broken, within re-alert window — keep original alert timestamp
    echo "fail $prev_alert" > "$STATE_FILE"
fi
exit 0
