#!/bin/bash
# Set the series access key (SERIES_KEY, DECISIONS #71) to one you choose.
# Run it yourself in a terminal on the Mac mini (over SSH is fine), NOT through Claude's "!"
# prompt: it reads the key with a hidden prompt, which needs a real terminal.
#
#   cd ~/Projetos/rabbitolivestudios.github.io/eink-dashboard && ./scripts/series-set-key.sh
#
# The key never appears on screen. It is saved in ~/.config/eink/series-key (readable only by
# you) and uploaded to Cloudflare as the Worker secret SERIES_KEY. Then the script checks the
# live proof page: refused without the key, open with it.
set -euo pipefail

KEYFILE="$HOME/.config/eink/series-key"
URL="https://eink-dashboard.thiago-oliveira77.workers.dev/series/proof?season=0&chapter=1"
export CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c

read -rsp "New series key (24+ characters; letters, digits, . _ ~ -): " key; echo
read -rsp "Type it again: " again; echo
[ "$key" = "$again" ] || { echo "The two entries differ. Nothing changed."; exit 1; }
[ "${#key}" -ge 24 ] || { echo "Too short: use at least 24 characters. Nothing changed."; exit 1; }
case "$key" in
  *[!A-Za-z0-9._~-]*) echo "Use only letters, digits and . _ ~ - (the key goes in a web address). Nothing changed."; exit 1 ;;
esac

umask 077
mkdir -p "$(dirname "$KEYFILE")"
printf %s "$key" > "$KEYFILE"
chmod 600 "$KEYFILE"
echo "Saved to $KEYFILE (only your user can read it)."

cd "$(dirname "$0")/.."
printf %s "$key" | npx wrangler secret put SERIES_KEY >/dev/null
echo "Uploaded to Cloudflare as SERIES_KEY. Checking the live site in 20 seconds..."
sleep 20

without=$(curl -s -o /dev/null -w '%{http_code}' "$URL")
with=$(curl -s -o /dev/null -w '%{http_code}' "$URL&key=$key")
echo "Proof page without the key: $without (expect 404)"
echo "Proof page with the key:    $with (expect 200)"
if [ "$without" = "404" ] && [ "$with" = "200" ]; then
  echo "Done. To show the key later: cat $KEYFILE"
else
  echo "Not as expected yet: Cloudflare can take a minute. Run this to re-check:"
  echo "  curl -s -o /dev/null -w '%{http_code}\n' \"$URL&key=\$(cat $KEYFILE)\""
fi
