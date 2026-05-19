#!/bin/bash
set -e
echo "=== Nyapsys Mac setup ==="

brew install llama.cpp python@3.11 git wget
pip3.11 install -r backend/requirements.txt

mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite
[ -d ~/nyapsys ] || git clone https://github.com/YOUR_USER/nyapsys ~/nyapsys

bash ~/nyapsys/scripts/download_model.sh

brew install cloudflared
bash ~/nyapsys/scripts/setup_tunnel.sh

cp ~/nyapsys/scripts/com.nyapsys.backend.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyapsys.backend.plist

echo "=== Setup complete ==="