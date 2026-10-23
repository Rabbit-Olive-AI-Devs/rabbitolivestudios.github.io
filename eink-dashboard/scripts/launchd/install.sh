#!/bin/bash
# Install the e-ink LaunchAgents on THIS Mac (DECISIONS #99). The plists in this folder are written
# for the Mac mini; this rewrites their paths for the Mac it runs on (home folder, this repo, node,
# and the Dropbox root from ~/.config/eink/paths.json), copies them to ~/Library/LaunchAgents and,
# with --load, starts them.
#
#   scripts/launchd/install.sh                  write all of them (not loaded), print what changed
#   scripts/launchd/install.sh --load           write and start all of them
#   scripts/launchd/install.sh --load serial-run storybook   only these
#
# com.eink.series-weekly is skipped unless named: it wrote Claude's chapters and is retired (#98).
# Never run the jobs on two Macs at once: two mailbox watchers would both act on every message.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
NODE="$(command -v node)"
DROPBOX="$(node -e 'import(process.argv[1]).then((m) => console.log(m.einkPaths().dropbox))' "$REPO/scripts/eink-paths.mjs")"
DEST="$HOME/Library/LaunchAgents"
OLD_REPO="/Users/tboliveira/Projetos/rabbitolivestudios.github.io/eink-dashboard"
OLD_DROPBOX="/Volumes/SamsungSSD/.CloudStorage/Data/Dropbox"

load=0
names=()
for a in "$@"; do
  if [ "$a" = "--load" ]; then load=1; else names+=("${a#com.eink.}"); fi
done
if [ ${#names[@]} -eq 0 ]; then
  for f in "$HERE"/com.eink.*.plist; do
    n="$(basename "$f" .plist)"; n="${n#com.eink.}"
    [ "$n" = "series-weekly" ] || names+=("$n")
  done
fi

[ -d "$DROPBOX/00_SHARED_INBOX/series" ] || { echo "No series mailbox under $DROPBOX: set \"dropbox\" in ~/.config/eink/paths.json first."; exit 1; }
mkdir -p "$DEST" "$HOME/Library/Logs/eink-series"
echo "repo $REPO | node $NODE | dropbox $DROPBOX"

for n in "${names[@]}"; do
  src="$HERE/com.eink.$n.plist"
  [ -f "$src" ] || { echo "no such job: $n"; exit 1; }
  out="$DEST/com.eink.$n.plist"
  sed -e "s#$OLD_REPO#$REPO#g" -e "s#$OLD_DROPBOX#$DROPBOX#g" -e "s#/opt/homebrew/bin/node#$NODE#g" -e "s#/Users/tboliveira#$HOME#g" "$src" > "$out"
  plutil -lint -s "$out"
  echo "wrote $out"
  if [ "$load" = 1 ]; then
    launchctl bootout "gui/$(id -u)/com.eink.$n" 2>/dev/null || true
    launchctl enable "gui/$(id -u)/com.eink.$n"
    launchctl bootstrap "gui/$(id -u)" "$out"
    echo "  loaded com.eink.$n"
  fi
done
