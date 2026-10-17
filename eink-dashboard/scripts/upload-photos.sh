#!/bin/bash
# Upload the birthday reference photos to R2, prepared for FLUX.2: upright, cropped to the
# person's face, under 512px (DECISIONS #85). Uploading the raw photos would hand the model
# whole scenes shrunk until the face is a few dozen pixels wide, some of them sideways.
#
# Photos: photos/portraits/{key}_{0-3}.jpg (any case, .jpg or .jpeg). Group photos pick the
# right face in photos/portraits/picks.json; check the contact sheet before trusting a run:
#   python3 -E -P scripts/prepare-portraits.py --src photos/portraits --out /tmp/portraits \
#     --picks photos/portraits/picks.json

set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$(mktemp -d)"
python3 -E -P "$DIR/scripts/prepare-portraits.py" --src "$DIR/photos/portraits" --out "$OUT" \
  --picks "$DIR/photos/portraits/picks.json" --upload
echo "Contact sheet: $OUT/sheet.jpg"
