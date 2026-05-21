#!/bin/bash

# ============================================================
#  Nyapsys Launcher
#  Run this script to boot up the full Nyapsys stack
# ============================================================

set -e

NYAPSYS_DIR="$HOME/nyapsys"
LOG_DIR="$NYAPSYS_DIR/logs"
ENV_FILE="$NYAPSYS_DIR/.env"
MODEL_PATH="$HOME/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf"
LLAMA_PORT=8080
CHROMA_PORT=8001
FASTAPI_PORT=8000

# ── colours ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

banner() {
  echo ""
  echo -e "${BOLD}${CYAN}"
  echo "  ███╗   ██╗██╗   ██╗ █████╗ ██████╗ ███████╗██╗   ██╗███████╗"
  echo "  ████╗  ██║╚██╗ ██╔╝██╔══██╗██╔══██╗██╔════╝╚██╗ ██╔╝██╔════╝"
  echo "  ██╔██╗ ██║ ╚████╔╝ ███████║██████╔╝███████╗ ╚████╔╝ ███████╗"
  echo "  ██║╚██╗██║  ╚██╔╝  ██╔══██║██╔═══╝ ╚════██║  ╚██╔╝  ╚════██║"
  echo "  ██║ ╚████║   ██║   ██║  ██║██║     ███████║   ██║   ███████║"
  echo "  ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝     ╚══════╝   ╚═╝   ╚══════╝"
  echo -e "${RESET}"
  echo -e "  ${CYAN}Self-hosted AI — running on your Mac${RESET}"
  echo ""
}

step()    { echo -e "\n${BOLD}${BLUE}▶ $1${RESET}"; }
ok()      { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail()    { echo -e "  ${RED}✗${RESET} $1"; }
die()     { fail "$1"; echo ""; exit 1; }

port_in_use() { lsof -i ":$1" -sTCP:LISTEN -t > /dev/null 2>&1; }
kill_port()   { lsof -ti ":$1" | xargs kill -9 2>/dev/null || true; }

wait_for_http() {
  local url=$1 label=$2 attempts=0 max=30
  printf "  Waiting for $label"
  while ! curl -sf "$url" > /dev/null 2>&1; do
    sleep 2; attempts=$((attempts+1))
    printf "."
    if [ $attempts -ge $max ]; then
      echo ""
      die "$label did not start in time. Check $LOG_DIR/$label.log"
    fi
  done
  echo ""
  ok "$label is ready"
}

# ── COMMAND: stop ─────────────────────────────────────────
cmd_stop() {
  step "Stopping Nyapsys services"
  launchctl stop com.nyapsys.backend 2>/dev/null && ok "launchd service stopped" || true
  kill_port $LLAMA_PORT;   ok "llama-server stopped"
  kill_port $CHROMA_PORT;  ok "ChromaDB stopped"
  kill_port $FASTAPI_PORT; ok "FastAPI stopped"
  rm -f /tmp/nyapsys-*.pid
  echo ""
  echo -e "  ${CYAN}Nyapsys stopped.${RESET}"
  echo ""
}

# ── COMMAND: status ───────────────────────────────────────
cmd_status() {
  echo ""
  echo -e "  ${BOLD}Service Status${RESET}"
  echo ""

  check_service() {
    local label=$1 url=$2
    if curl -sf "$url" > /dev/null 2>&1; then
      echo -e "  ${GREEN}●${RESET} $label ${GREEN}running${RESET}"
    else
      echo -e "  ${RED}○${RESET} $label ${RED}stopped${RESET}"
    fi
  }

  check_service "llama-server (model)" "http://127.0.0.1:$LLAMA_PORT/health"
  check_service "ChromaDB            " "http://127.0.0.1:$CHROMA_PORT/api/v1/heartbeat"
  check_service "FastAPI backend     " "http://127.0.0.1:$FASTAPI_PORT/health"

  echo ""
  if [ -f "$MODEL_PATH" ]; then
    SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
    ok "Model found ($SIZE) — $MODEL_PATH"
  else
    warn "Model not found at $MODEL_PATH"
    warn "Run training on GCP and then: bash scripts/download_model.sh"
  fi
  echo ""
}

# ── COMMAND: logs ─────────────────────────────────────────
cmd_logs() {
  echo ""
  echo -e "  ${BOLD}Tailing backend logs — Ctrl+C to exit${RESET}"
  echo ""
  tail -f "$LOG_DIR/backend.log" 2>/dev/null || die "No log file found at $LOG_DIR/backend.log"
}

# ── COMMAND: setup (first-time install) ───────────────────
cmd_setup() {
  step "Checking system dependencies"

  command -v brew  > /dev/null || die "Homebrew not found. Install it first: https://brew.sh"
  ok "Homebrew found"

  for pkg in llama.cpp python@3.11 git wget cloudflared; do
    if brew list "$pkg" > /dev/null 2>&1; then
      ok "$pkg already installed"
    else
      echo "  Installing $pkg..."
      brew install "$pkg" 2>&1 | tail -1
      ok "$pkg installed"
    fi
  done

  step "Creating volume directories"
  mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite "$LOG_DIR"
  ok "Directories ready"

  step "Checking .env file"
  if [ ! -f "$ENV_FILE" ]; then
    if [ -f "$NYAPSYS_DIR/.env.example" ]; then
      cp "$NYAPSYS_DIR/.env.example" "$ENV_FILE"
      GENERATED_KEY=$(openssl rand -hex 32)
      sed -i '' "s/replace-with-output-of-openssl-rand-hex-32/$GENERATED_KEY/" "$ENV_FILE"
      ok ".env created with auto-generated API key"
      warn "Edit $ENV_FILE to set your domain and GCP settings before deploying"
    else
      die ".env.example not found in $NYAPSYS_DIR"
    fi
  else
    ok ".env already exists"
  fi

  step "Installing Python dependencies"
  python3.11 -m pip install -r "$NYAPSYS_DIR/backend/requirements.txt" -q
  ok "Python packages installed"

  step "Installing frontend dependencies"
  cd "$NYAPSYS_DIR/frontend" && npm install --silent
  ok "Node packages installed"

  step "Registering launchd service"
  PLIST_SRC="$NYAPSYS_DIR/scripts/com.nyapsys.backend.plist"
  PLIST_DEST="$HOME/Library/LaunchAgents/com.nyapsys.backend.plist"

  if [ -f "$PLIST_SRC" ]; then
    sed "s/YOUR_USERNAME/$(whoami)/g" "$PLIST_SRC" > "$PLIST_DEST"
    launchctl load "$PLIST_DEST" 2>/dev/null || true
    ok "launchd service registered (auto-starts on login)"
  else
    warn "Plist not found at $PLIST_SRC — skipping launchd setup"
  fi

  echo ""
  echo -e "  ${GREEN}${BOLD}Setup complete!${RESET}"
  echo ""
  echo -e "  ${CYAN}Next steps:${RESET}"
  if [ ! -f "$MODEL_PATH" ]; then
    echo -e "  ${YELLOW}1.${RESET} Train your model on GCP (see training/ directory)"
    echo -e "  ${YELLOW}2.${RESET} Download the GGUF: ${BOLD}bash scripts/download_model.sh${RESET}"
    echo -e "  ${YELLOW}3.${RESET} Start Nyapsys: ${BOLD}bash nyapsys-launcher.sh start${RESET}"
  else
    echo -e "  ${YELLOW}1.${RESET} Start Nyapsys: ${BOLD}bash nyapsys-launcher.sh start${RESET}"
  fi
  echo ""
}

# ── COMMAND: start ────────────────────────────────────────
cmd_start() {
  banner

  # ── preflight ─────────────────────────────────────────
  step "Preflight checks"

  [ -f "$ENV_FILE" ]     || die ".env not found at $ENV_FILE — run: bash nyapsys-launcher.sh setup"
  ok ".env found"

  [ -f "$MODEL_PATH" ]   || {
    warn "Model not found at $MODEL_PATH"
    warn "If training is done, run: bash $NYAPSYS_DIR/scripts/download_model.sh"
    die "Cannot start without the model file"
  }
  MODEL_SIZE=$(ls -lh "$MODEL_PATH" | awk '{print $5}')
  ok "Model found ($MODEL_SIZE)"

  command -v llama-server > /dev/null || die "llama-server not found — run: bash nyapsys-launcher.sh setup"
  ok "llama.cpp found"

  command -v python3.11  > /dev/null || die "python3.11 not found — run: bash nyapsys-launcher.sh setup"
  ok "Python 3.11 found"

  source "$ENV_FILE"

  # ── clear any stale processes ─────────────────────────
  step "Clearing any stale processes"
  for p in $LLAMA_PORT $CHROMA_PORT $FASTAPI_PORT; do
    if port_in_use $p; then
      warn "Port $p in use — killing"
      kill_port $p
      sleep 1
    fi
  done
  ok "All ports clear"

  mkdir -p "$LOG_DIR"

  # ── llama-server ──────────────────────────────────────
  step "Starting llama-server (model inference)"
  llama-server \
    --model "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port $LLAMA_PORT \
    --ctx-size 2048 \
    --n-predict 2048 \
    --threads 4 \
    --parallel 1 \
    --flash-attn \
    --mlock \
    --cont-batching \
    > "$LOG_DIR/llama.log" 2>&1 &

  LLAMA_PID=$!
  echo $LLAMA_PID > /tmp/nyapsys-llama.pid
  wait_for_http "http://127.0.0.1:$LLAMA_PORT/health" "llama-server"
  sleep 2

  # ── chromadb ──────────────────────────────────────────
  step "Starting ChromaDB (vector store)"
  python3.11 -m chromadb.cli.cli run \
    --path ~/volumes/chromadb \
    --host 127.0.0.1 \
    --port $CHROMA_PORT \
    > "$LOG_DIR/chroma.log" 2>&1 &

  CHROMA_PID=$!
  echo $CHROMA_PID > /tmp/nyapsys-chroma.pid
  wait_for_http "http://127.0.0.1:$CHROMA_PORT/api/v1/heartbeat" "ChromaDB"
  sleep 2

  # ── fastapi ───────────────────────────────────────────
  step "Starting FastAPI backend"
  cd "$NYAPSYS_DIR/backend"
  python3.11 -m uvicorn app.main:app \
    --host 127.0.0.1 \
    --port $FASTAPI_PORT \
    --workers 1 \
    > "$LOG_DIR/fastapi.log" 2>&1 &

  FASTAPI_PID=$!
  echo $FASTAPI_PID > /tmp/nyapsys-fastapi.pid
  wait_for_http "http://127.0.0.1:$FASTAPI_PORT/health" "FastAPI"

  # ── frontend ──────────────────────────────────────────
  step "Starting frontend (dev server)"
  cd "$NYAPSYS_DIR/frontend"
  if [ ! -d "node_modules" ]; then
    warn "node_modules not found — running npm install first"
    npm install --silent
  fi
  npm run dev > "$LOG_DIR/frontend.log" 2>&1 &
  FRONTEND_PID=$!
  echo $FRONTEND_PID > /tmp/nyapsys-frontend.pid
  sleep 4

  # ── cloudflare tunnel (if configured) ─────────────────
  if [ -f "$HOME/.cloudflared/config.yml" ]; then
    step "Starting Cloudflare Tunnel"
    cloudflared tunnel run nyapsys > "$LOG_DIR/tunnel.log" 2>&1 &
    echo $! > /tmp/nyapsys-tunnel.pid
    sleep 2
    ok "Cloudflare Tunnel started"
  fi

  # ── done ──────────────────────────────────────────────
  echo ""
  echo -e "  ${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo -e "  ${GREEN}${BOLD}  Nyapsys is running!${RESET}"
  echo -e "  ${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
  echo ""
  echo -e "  ${CYAN}Frontend${RESET}   http://localhost:3000"
  echo -e "  ${CYAN}API${RESET}        http://localhost:$FASTAPI_PORT"
  echo -e "  ${CYAN}Health${RESET}     http://localhost:$FASTAPI_PORT/health"
  echo ""
  echo -e "  ${YELLOW}Logs:${RESET}  tail -f $LOG_DIR/fastapi.log"
  echo -e "  ${YELLOW}Stop:${RESET}  bash nyapsys-launcher.sh stop"
  echo ""

  # Open browser
  sleep 2
  open "http://localhost:3000" 2>/dev/null || true
}

# ── COMMAND: restart ──────────────────────────────────────
cmd_restart() {
  cmd_stop
  sleep 2
  cmd_start
}

# ── ENTRY POINT ───────────────────────────────────────────
case "${1:-start}" in
  start)   cmd_start   ;;
  stop)    cmd_stop    ;;
  restart) cmd_restart ;;
  status)  cmd_status  ;;
  logs)    cmd_logs    ;;
  setup)   cmd_setup   ;;
  *)
    echo ""
    echo -e "  ${BOLD}Nyapsys Launcher${RESET}"
    echo ""
    echo -e "  Usage: ${CYAN}bash nyapsys-launcher.sh [command]${RESET}"
    echo ""
    echo -e "  ${BOLD}Commands:${RESET}"
    echo -e "  ${CYAN}start${RESET}    Boot the full stack (default)"
    echo -e "  ${CYAN}stop${RESET}     Shut everything down"
    echo -e "  ${CYAN}restart${RESET}  Stop then start"
    echo -e "  ${CYAN}status${RESET}   Check what's running"
    echo -e "  ${CYAN}logs${RESET}     Tail the backend log"
    echo -e "  ${CYAN}setup${RESET}    First-time install (deps, .env, launchd)"
    echo ""
    ;;
esac
