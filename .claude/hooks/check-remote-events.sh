#!/bin/bash
# Check for completion events from FADA monitor server via Cloudflare Tunnel.
# Used by both SessionStart and Stop hooks.
#
# Curls the monitor API, acks consumed events.
# Fast: HTTPS via Cloudflare, no SSH needed.
#
# Design: at-least-once delivery. Events are fetched and written to the result
# file before ack. If ack fails, the same events will be re-delivered on the
# next poll, which is safe because the result file is overwritten each time.
#
# Exit 0 = no events (no continuation)
# Exit 2 = events found (triggers Claude Code continuation)

set -uo pipefail

HOOK_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULT="$HOOK_DIR/event_result.json"
MONITOR_URL="https://fada-monitor.elyasamri.com"
TOKEN_FILE="$HOOK_DIR/.monitor_token"

# Auth token (optional -- exit silently if not configured)
if [ ! -f "$TOKEN_FILE" ]; then
  exit 0
fi
TOKEN=$(cat "$TOKEN_FILE" | tr -d '[:space:]')
AUTH_HEADER="Authorization: Bearer $TOKEN"

# Check for pending events
RESPONSE=$(curl -sf --connect-timeout 3 --max-time 5 \
  -H "$AUTH_HEADER" \
  "$MONITOR_URL/events" 2>/dev/null) || exit 0

# Use python to safely parse JSON response: extract count, IDs, and build ack payload.
# Never interpolate untrusted HTTP response data into shell strings.
PARSED=$(py -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    count = int(data.get('count', 0))
    if count == 0:
        sys.exit(0)
    events = data.get('events', [])
    ids = [e['id'] for e in events if 'id' in e]
    ack_payload = json.dumps({'ids': ids})
    print(f'{count}')
    print(ack_payload)
except (json.JSONDecodeError, KeyError, TypeError, ValueError):
    sys.exit(0)
" <<< "$RESPONSE") || exit 0

COUNT=$(echo "$PARSED" | head -1)
ACK_PAYLOAD=$(echo "$PARSED" | tail -1)

[ -z "$COUNT" ] || [ "$COUNT" = "0" ] && exit 0

# Write full response to result file (before ack -- at-least-once delivery)
echo "$RESPONSE" > "$RESULT"

# Ack the events. Log failures to stderr instead of silently swallowing.
if [ -n "$ACK_PAYLOAD" ]; then
  if ! curl -sf --connect-timeout 3 --max-time 5 \
    -X POST "$MONITOR_URL/events/ack" \
    -H "Content-Type: application/json" \
    -H "$AUTH_HEADER" \
    -d "$ACK_PAYLOAD" >/dev/null 2>&1; then
    echo "[check-remote-events] WARNING: Failed to ack events. They will be re-delivered next poll." >&2
  fi
fi

echo "Remote task event: $COUNT event(s) detected. Details in .claude/hooks/event_result.json"
exit 2
