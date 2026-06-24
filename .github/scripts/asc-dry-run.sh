#!/usr/bin/env bash
set -euo pipefail

APP_ID="$1"
VERSION="$2"
BUILD="$3"
ISSUER_ID="$4"
KEY_ID="$5"
KEY_PATH="$6"

export ASC_ISSUER_ID="$ISSUER_ID"
export ASC_KEY_ID="$KEY_ID"
export ASC_PRIVATE_KEY_PATH="$KEY_PATH"

# Read-only validation flow. Do not create, update, submit, or release anything.
# Exact asc-cli subcommands can vary by version; keep this script conservative.

echo "[ASC dry-run] Checking connectivity and access..."
asc whoami >/dev/null

echo "[ASC dry-run] Looking up app: $APP_ID"
asc apps list --filter-bundle-id "$APP_ID" --limit 1

echo "[ASC dry-run] Looking up version metadata: $VERSION"
asc appstore-versions list --filter-app "$APP_ID" --filter-version-string "$VERSION" --limit 5 || true

echo "[ASC dry-run] Looking up build: $BUILD"
asc builds list --filter-app "$APP_ID" --filter-version "$VERSION" --filter-build-number "$BUILD" --limit 5 || true

echo "[ASC dry-run] Completed read-only checks."
