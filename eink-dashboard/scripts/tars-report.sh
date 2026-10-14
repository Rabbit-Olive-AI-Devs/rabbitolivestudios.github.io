#!/bin/bash
# Reports this Mac's vital signs, and CodexBar's AI usage snapshot when CodexBar
# is installed, to the desk panel's TARS page (DECISIONS #69).
#
#   tars-report.sh <machine-id> "<display name>"      e.g.  tars-report.sh mini "Mac mini"
#
# Run every 30 minutes by launchd (see scripts/tars-launchd.plist.example).
#
# The key is read from the login Keychain, never from a file:
#   security add-generic-password -a "$USER" -s eink-tars -w      (it prompts for the key)
# TARS_KEY and TARS_URL in the environment override it, for testing.
set -u
PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

ID="${1:?machine id, e.g. mini}"
NAME="${2:?display name, e.g. \"Mac mini\"}"
URL="${TARS_URL:-https://eink-dashboard.thiago-oliveira77.workers.dev/desk/tars}"
KEY="${TARS_KEY:-$(security find-generic-password -s eink-tars -w 2>/dev/null)}"
if [ -z "$KEY" ]; then
  echo "tars-report: no key. Add one with: security add-generic-password -a \"\$USER\" -s eink-tars -w" >&2
  exit 1
fi

# Disk: the data volume, in 1 KiB blocks -> GiB.
read -r DISK_TOTAL DISK_FREE <<<"$(df -k /System/Volumes/Data | awk 'NR==2 {printf "%d %d", $2/1048576, $4/1048576}')"
# Memory: macOS reports the share that is free; the page shows the share in use.
MEM_FREE="$(memory_pressure 2>/dev/null | awk '/free percentage/ {gsub("%","",$NF); print $NF}')"
MEM_USED="null"; [ -n "$MEM_FREE" ] && MEM_USED=$((100 - MEM_FREE))
BOOT="$(sysctl -n kern.boottime | sed -E 's/^\{ sec = ([0-9]+).*/\1/')"
UPTIME=$(( $(date +%s) - BOOT ))

# CodexBar, if present. Emails are redacted at the source and dropped again by the Worker.
# The fuller `usage` output adds what the snapshot leaves out (extra tokens beyond a
# spent week); it has no redaction option, so anything that looks like an email is
# blanked here before it leaves the machine.
SNAPSHOT="null"
USAGE="null"
if command -v codexbar >/dev/null 2>&1; then
  BLANK_EMAILS='s/"[^"]*@[^"]*"/"redacted"/g'
  OUT="$(codexbar dashboard --identity redacted --timeout 60 2>/dev/null | sed -E "$BLANK_EMAILS")"
  case "$OUT" in "{"*) SNAPSHOT="$OUT" ;; esac
  OUT="$(codexbar usage --format json 2>/dev/null | sed -E "$BLANK_EMAILS")"
  # skipped when unexpectedly large, so it can never push the report over the size limit
  case "$OUT" in "["*) [ "${#OUT}" -lt 30000 ] && USAGE="$OUT" ;; esac
fi

# The name is the only free text; strip anything that could break the JSON.
SAFE_NAME="$(printf '%s' "$NAME" | tr -cd 'A-Za-z0-9 .-')"

printf '{"machine":{"id":"%s","name":"%s","diskTotalGb":%s,"diskFreeGb":%s,"memUsedPct":%s,"uptimeSec":%s},"codexbar":%s,"usage":%s}' \
  "$ID" "$SAFE_NAME" "${DISK_TOTAL:-0}" "${DISK_FREE:-0}" "$MEM_USED" "$UPTIME" "$SNAPSHOT" "$USAGE" |
  curl -sS -m 30 -X POST -H "Authorization: Bearer $KEY" -H "Content-Type: application/json" --data-binary @- "$URL"
echo
