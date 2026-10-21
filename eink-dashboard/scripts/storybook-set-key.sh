#!/bin/bash
# Set the story book key (STORYBOOK_KEY, DECISIONS #97) to one you choose, and print the link
# to the story book on the web. Run it yourself in a terminal on the Mac mini (over SSH is fine),
# NOT through Claude's "!" prompt: it reads the key with a hidden prompt, which needs a real terminal.
#
#   cd ~/Projetos/rabbitolivestudios.github.io/eink-dashboard && ./scripts/storybook-set-key.sh
#
# The key is saved in ~/.config/eink/storybook-key (readable only by you) and uploaded to
# Cloudflare as the Worker secret STORYBOOK_KEY. Then the script checks the live page: refused
# without the key, open with it.
set -euo pipefail

KEYFILE="$HOME/.config/eink/storybook-key"
BASE="https://eink-dashboard.thiago-oliveira77.workers.dev/storybook"
export CLOUDFLARE_ACCOUNT_ID=f22a506dedde3bb3837157cd47d5fe5c

read -rsp "New story book key (8+ characters; letters, digits, . _ ~ -): " key; echo
read -rsp "Type it again: " again; echo
[ "$key" = "$again" ] || { echo "The two entries differ. Nothing changed."; exit 1; }
[ "${#key}" -ge 8 ] || { echo "Too short: use at least 8 characters. Nothing changed."; exit 1; }
case "$key" in
  *[!A-Za-z0-9._~-]*) echo "Use only letters, digits and . _ ~ - (the key goes in a web address). Nothing changed."; exit 1 ;;
esac

umask 077
mkdir -p "$(dirname "$KEYFILE")"
printf %s "$key" > "$KEYFILE"
chmod 600 "$KEYFILE"
echo "Saved to $KEYFILE (only your user can read it)."

cd "$(dirname "$0")/.."
printf %s "$key" | npx wrangler secret put STORYBOOK_KEY >/dev/null
echo "Uploaded to Cloudflare as STORYBOOK_KEY. Checking the live site in 20 seconds..."
sleep 20

without=$(curl -s -o /dev/null -w '%{http_code}' "$BASE")
with=$(curl -s -o /dev/null -w '%{http_code}' "$BASE?key=$key")
echo "Story book without the key: $without (expect 404)"
echo "Story book with the key:    $with (expect 200)"
if [ "$without" = "404" ] && [ "$with" = "200" ]; then
  echo
  echo "Done. Your link (open it on your phone and add it to the Home Screen):"
  echo "  $BASE?key=$key"
  echo "To show it again later: echo \"$BASE?key=\$(cat $KEYFILE)\""
else
  echo "Not as expected yet: Cloudflare can take a minute. Run this to re-check:"
  echo "  curl -s -o /dev/null -w '%{http_code}\n' \"$BASE?key=\$(cat $KEYFILE)\""
fi
