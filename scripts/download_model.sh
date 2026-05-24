#!/bin/bash
set -e

BUCKET="nyapsys-models"
FILENAME="Nyapsys-2B-MoE.Q4_K_M.gguf"
DEST="$HOME/volumes/models/$FILENAME"
GCS_PATH="gs://$BUCKET/$FILENAME"

echo "=== Nyapsys model download ==="
echo "Source: $GCS_PATH"
echo "Dest:   $DEST"

mkdir -p "$HOME/volumes/models"

if ! gcloud auth print-access-token > /dev/null 2>&1; then
  echo "Not authenticated. Run: gcloud auth login"
  exit 1
fi

echo "Downloading..."
gsutil -m cp "$GCS_PATH" "$DEST"

FILE_SIZE=$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")
if [ "$FILE_SIZE" -lt 1000000000 ]; then
  echo "ERROR: File too small ($FILE_SIZE bytes). Download may be incomplete."
  exit 1
fi

echo "Downloaded: $(ls -lh "$DEST" | awk '{print $5}')"
echo "=== Done. Model ready at $DEST ==="