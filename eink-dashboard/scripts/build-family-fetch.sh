#!/bin/bash
# Family Memory (DECISIONS #92): build the PhotoKit downloader as a tiny background app,
# ~/Applications/EinkFamilyFetch.app. The first run asks for Photos access on the Mac's screen;
# allow it once and macOS remembers it for the app.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/EinkFamilyFetch.app"
mkdir -p "$APP/Contents/MacOS"
swiftc -O -target arm64-apple-macos14.0 "$DIR/scripts/family-fetch.swift" -o "$APP/Contents/MacOS/family-fetch"
cat > "$APP/Contents/Info.plist" <<PLIST
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
codesign --force --sign - "$APP"
/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister -f "$APP"
echo "Built $APP"
