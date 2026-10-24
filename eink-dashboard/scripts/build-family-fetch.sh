#!/bin/bash
# Build ONLY into a new explicit staging path. Installation/consent are separate owner gates.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP=""
WITHOUT_PEOPLE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --output) [[ $# -ge 2 ]] || exit 2; APP="$2"; shift 2 ;;
    --without-people) WITHOUT_PEOPLE=1; shift ;;
    *) printf '%s\n' 'usage: build-family-fetch.sh --output /absolute/new/EinkFamilyFetch.app [--without-people]'; exit 2 ;;
  esac
done
[[ "$APP" = /*.app && ! -e "$APP" && ! -L "$APP" ]] || { printf '%s\n' 'output must be a new absolute .app path'; exit 2; }
case "$APP" in "$HOME/Applications/"*|*/../*|*/./*) printf '%s\n' 'use an explicit non-live staging path'; exit 2 ;; esac
[[ -d "$(dirname "$APP")" ]] || exit 2
PARENT="$(cd "$(dirname "$APP")" && pwd -P)"
[[ "$APP" = "$PARENT/$(basename "$APP")" ]] || { printf '%s\n' 'symlink or noncanonical output parent refused'; exit 2; }
WORK="$(mktemp -d "$(dirname "$APP")/.family-build.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
BUILD="$WORK/EinkFamilyFetch.app"
mkdir -p "$BUILD/Contents/MacOS" "$BUILD/Contents/Resources"
if [[ "$WITHOUT_PEOPLE" = 0 ]]; then
  # Owner's build-time copy only: never print, commit, or put this roster into source staging.
  ROSTER="$DIR/photos/portraits/people.json"
  [[ -f "$ROSTER" && ! -L "$ROSTER" ]] || { printf '%s\n' 'private roster unavailable; compile-only builds may use --without-people'; exit 2; }
  cp "$ROSTER" "$BUILD/Contents/Resources/people.json"
  chmod 600 "$BUILD/Contents/Resources/people.json"
fi
CACHE="${SWIFT_MODULECACHE_PATH:-${TMPDIR:?TMPDIR required}/family-swift-cache}"
mkdir -p "$CACHE"
swiftc -O -target arm64-apple-macos14.0 -module-cache-path "$CACHE" "$DIR/scripts/family-core.swift" "$DIR/scripts/family-fetch.swift" -lsqlite3 -o "$BUILD/Contents/MacOS/family-fetch"
cat > "$BUILD/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleIdentifier</key><string>com.eink.family-fetch</string>
  <key>CFBundleName</key><string>EinkFamilyFetch</string>
  <key>CFBundleExecutable</key><string>family-fetch</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>LSMinimumSystemVersion</key><string>14.0</string>
  <key>LSUIElement</key><true/>
  <key>NSPhotoLibraryUsageDescription</key><string>The e-ink Family Memory page downloads one family photo a day to repaint for the panels.</string>
</dict></plist>
PLIST
codesign --force --sign - "$BUILD"
mv "$BUILD" "$APP"
printf 'Built staged %s\n' "$APP"
if [[ "$WITHOUT_PEOPLE" = 1 ]]; then printf '%s\n' 'Compile-only: no people resource; selection cannot run until an owner build includes it.'; fi
