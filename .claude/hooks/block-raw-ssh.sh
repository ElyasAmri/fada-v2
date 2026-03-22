#!/bin/bash
# Block raw SSH commands -- enforce use of rccg.sh wrapper
# PreToolUse hook for Bash tool calls

INPUT=$(cat)

# Extract command without jq or grep -P (not available on Git Bash)
COMMAND=$(echo "$INPUT" | sed -n 's/.*"command"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' | head -1)

if echo "$COMMAND" | grep -qE 'ssh[[:space:]]+-i.*rccg_key|ssh[[:space:]]+ubuntu@|scp[[:space:]]+-i.*rccg_key'; then
  echo "BLOCKED: Raw SSH command detected." >&2
  echo "Use ./r.sh instead:" >&2
  echo "" >&2
  echo "  Simple:  ./r.sh ssh fada-1 hostname" >&2
  echo "" >&2
  echo "  Complex (pipes/redirects): use heredoc, no quoting needed:" >&2
  echo "    ./r.sh ssh fada-1 <<'CMD'" >&2
  echo "    nvidia-smi | grep MiB && echo done" >&2
  echo "    CMD" >&2
  echo "" >&2
  echo "  Other:   ./r.sh status" >&2
  echo "           ./r.sh logs fada-1" >&2
  exit 2
fi

exit 0
