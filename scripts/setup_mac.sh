#!/bin/bash
set -e
echo "=== Nyapsys Mac setup ==="

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

brew install llama.cpp python@3.11 git wget
pip3.11 install -r "$PROJECT_DIR/backend/requirements.txt"

mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite

[ -d ~/nyapsys ] || git clone https://github.com/YOUR_USERNAME/nyapsys ~/nyapsys

bash "$PROJECT_DIR/scripts/download_model.sh"

brew install cloudflared
bash "$PROJECT_DIR/scripts/setup_tunnel.sh"

cp "$PROJECT_DIR/scripts/com.nyapsys.backend.plist" ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyapsys.backend.plist

echo "=== Setup complete ==="
