#!/bin/bash

# ============================================================
#  Nyapsys Full Training Pipeline
#  Run this on your Mac — it handles everything end-to-end
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TRAINING_DIR="$PROJECT_DIR/training"

# ── colours ──────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step()    { echo -e "\n${BOLD}${BLUE}▶ $1${RESET}"; }
ok()      { echo -e "  ${GREEN}✓${RESET} $1"; }
warn()    { echo -e "  ${YELLOW}⚠${RESET}  $1"; }
fail()    { echo -e "  ${RED}✗${RESET} $1"; }
die()     { fail "$1"; echo ""; exit 1; }

# ── Config ───────────────────────────────────────────────
GCP_PROJECT="${GCP_PROJECT:-nyapsys}"
GCP_ZONE="${GCP_ZONE:-asia-southeast1-a}"
INSTANCE_NAME="${INSTANCE_NAME:-nyapsys-training}"
GCS_BUCKET="${GCS_BUCKET:-nyapsys-training}"
MODEL_BUCKET="${MODEL_BUCKET:-nyapsys-models}"
TOKENIZED_DATA="gs://$GCS_BUCKET/tokenized"
CHECKPOINT_DIR="gs://$GCS_BUCKET/checkpoints"
INSTRUCT_DIR="gs://$GCS_BUCKET/instruct-output"

banner() {
  echo ""
  echo -e "${BOLD}${CYAN}"
  echo "  ████████╗██╗  ██╗███████╗    ███████╗██╗   ██╗███████╗████████╗"
  echo "  ╚══██╔══╝██║  ██║██╔════╝    ██╔════╝╚██╗ ██╔╝██╔════╝╚══██╔══╝"
  echo "     ██║   ███████║█████╗      ███████╗ ╚████╔╝ ███████╗   ██║   "
  echo "     ██║   ██╔══██║██╔══╝      ╚════██║  ╚██╔╝  ╚════██║   ██║   "
  echo "     ██║   ██║  ██║███████╗    ███████║   ██║   ███████║   ██║   "
  echo "     ╚═╝   ╚═╝  ╚═╝╚══════╝    ╚══════╝   ╚═╝   ╚══════╝   ╚═╝   "
  echo -e "${RESET}"
  echo -e "  ${CYAN}Full training pipeline — from data to GGUF${RESET}"
  echo ""
}

# ── PHASE 0: Prerequisites ───────────────────────────────
phase_prerequisites() {
  banner
  step "Phase 0: Checking prerequisites"

  command -v gcloud > /dev/null || die "gcloud not found. Run: brew install --cask google-cloud-sdk"
  ok "gcloud found"

  gcloud auth print-access-token > /dev/null 2>&1 || die "Not authenticated. Run: gcloud auth login"
  ok "gcloud authenticated"

  gcloud config get-value project 2>/dev/null | grep -q "." || {
    warn "No GCP project set. Setting to: $GCP_PROJECT"
    gcloud config set project "$GCP_PROJECT"
  }
  ok "GCP project: $(gcloud config get-value project)"

  # Check GPU quota
  QUOTA=$(gcloud compute regions describe $GCP_ZONE --format="value(quotas[].limit)" --filter="quotas.metric=NVIDIA_L4_GPUS" 2>/dev/null || echo "0")
  if [ "$QUOTA" = "0" ] || [ -z "$QUOTA" ]; then
    warn "No L4 GPU quota found."
    warn "Request it at: https://console.cloud.google.com/iam-admin/quotas"
    warn "Search: NVIDIA_L4_GPUS → Request increase to 1"
    read -p "Have you requested GPU quota? (y/n): " answer
    if [ "$answer" != "y" ]; then
      die "GPU quota required before training. Request it and try again."
    fi
  else
    ok "L4 GPU quota: $QUOTA"
  fi

  # Check billing
  BILLING=$(gcloud projects describe $(gcloud config get-value project) --format="value(lifecycleState)" 2>/dev/null)
  if [ "$BILLING" != "ACTIVE" ]; then
    die "Project is not ACTIVE. Check GCP Console."
  fi
  ok "Project status: ACTIVE"

  # Check Python deps
  step "Checking Python dependencies"
  python3.11 -c "import datasets, tokenizers, datasketch" 2>/dev/null || {
    step "Installing Python dependencies"
    pip3.11 install -r "$TRAINING_DIR/requirements.txt" -q
    ok "Python packages installed"
  }
  ok "Python dependencies ready"
}

# ── PHASE 1: Data Preparation ────────────────────────────
phase_data_prep() {
  step "Phase 1: Preparing training data"

  cd "$TRAINING_DIR"
  python3.11 prepare_dataset.py

  TOTAL=$(wc -l < data/tokenized/train.jsonl 2>/dev/null || echo "0")
  if [ "$TOTAL" -lt 100 ]; then
    die "Only $TOTAL samples in train.jsonl — data preparation failed"
  fi
  ok "Train samples: $TOTAL"

  step "Uploading data to GCS"
  gsutil -m cp -r data/tokenized/ "$TOKENIZED_DATA/"
  ok "Data uploaded to $TOKENIZED_DATA"
}

# ── PHASE 2: Create GCP Instance ─────────────────────────
phase_create_instance() {
  step "Phase 2: Creating GCP spot instance"

  # Check if instance already exists
  EXISTING=$(gcloud compute instances list --filter="name=$INSTANCE_NAME" --format="value(name)" 2>/dev/null || true)
  if [ -n "$EXISTING" ]; then
    warn "Instance $INSTANCE_NAME already exists"
    read -p "Delete and recreate? (y/n): " answer
    if [ "$answer" = "y" ]; then
      gcloud compute instances delete "$INSTANCE_NAME" --zone="$GCP_ZONE" --quiet
      ok "Deleted existing instance"
    else
      ok "Using existing instance"
      return
    fi
  fi

  gcloud compute instances create "$INSTANCE_NAME" \
    --zone="$GCP_ZONE" \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-l4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --maintenance-policy=TERMINATE \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP

  ok "Instance created: $INSTANCE_NAME"

  step "Waiting for instance to be ready..."
  sleep 30

  # Verify GPU
  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="nvidia-smi --query-gpu=name,memory.total --format=csv,noheader" || {
    warn "GPU not detected yet — waiting 60s and retrying..."
    sleep 60
    gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="nvidia-smi" || die "GPU still not available"
  }
  ok "GPU verified"
}

# ── PHASE 3: Setup Instance ──────────────────────────────
phase_setup_instance() {
  step "Phase 3: Setting up training instance"

  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
    set -e
    echo 'Installing dependencies...'
    git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
    cd /opt/llama.cpp && make -j\$(nproc) > /dev/null 2>&1

    git clone https://github.com/sameweeeeeee/nyapsys /opt/nyapsys
    cd /opt/nyapsys
    pip install -r training/requirements.txt -q

    gsutil -m cp -r $TOKENIZED_DATA/ training/data/tokenized/

    echo 'Setup complete'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  "
  ok "Instance setup complete"
}

# ── PHASE 4: Smoke Test ──────────────────────────────────
phase_smoke_test() {
  step "Phase 4: Running smoke test (1% data, ~30 min)"

  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
    set -e
    cd /opt/nyapsys/training
    python3 train.py \
      --config 2b_moe \
      --data_path ../training/data/tokenized/ \
      --output_dir $CHECKPOINT_DIR/ \
      --max_steps 500 \
      --smoke_test
  "
  ok "Smoke test passed"
}

# ── PHASE 5: Full Pretraining ────────────────────────────
phase_pretrain() {
  step "Phase 5: Starting full pretraining (~70-90 hours)"
  warn "This will run for 70-90 hours. You can safely disconnect."
  warn "Check progress with: gcloud compute ssh $INSTANCE_NAME --zone=$GCP_ZONE --command='tail -f /tmp/train.log'"

  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
    set -e
    cd /opt/nyapsys/training

    nohup python3 train.py \
      --config 2b_moe \
      --data_path gs://$GCS_BUCKET/tokenized/ \
      --output_dir $CHECKPOINT_DIR/ \
      --resume_from_checkpoint latest \
      > /tmp/train.log 2>&1 &

    echo \$! > /tmp/train.pid
    echo \"Training started (PID \$(cat /tmp/train.pid))\"
  "

  ok "Pretraining started in background"
  step "Monitoring first 5 minutes..."

  # Wait and show initial progress
  for i in $(seq 1 15); do
    sleep 20
    gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="tail -5 /tmp/train.log 2>/dev/null || echo 'Waiting for logs...'" || true
    echo "---"
  done

  ok "Pretraining is running. Check logs anytime with:"
  echo "  gcloud compute ssh $INSTANCE_NAME --zone=$GCP_ZONE --command='tail -f /tmp/train.log'"
}

# ── PHASE 6: Wait for Pretraining ────────────────────────
phase_wait_pretrain() {
  step "Phase 6: Waiting for pretraining to complete"
  echo ""
  echo "  Polling every 10 minutes. Press Ctrl+C to skip ahead."
  echo "  (The script will continue automatically when done)"
  echo ""

  while true; do
    RESULT=$(gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="test -f /opt/nyapsys/training/hf_model/config.json && echo 'done' || echo 'running'" 2>/dev/null || echo "error")

    if [ "$RESULT" = "done" ]; then
      ok "Pretraining complete!"
      break
    fi

    if [ "$RESULT" = "error" ]; then
      warn "Instance may have been preempted. Checking..."
      STATUS=$(gcloud compute instances describe "$INSTANCE_NAME" --zone="$GCP_ZONE" --format="value(status)" 2>/dev/null || echo "unknown")
      if [ "$STATUS" = "TERMINATED" ] || [ "$STATUS" = "STOPPED" ]; then
        warn "Instance was preempted. Restarting and resuming..."
        gcloud compute instances start "$INSTANCE_NAME" --zone="$GCP_ZONE"
        sleep 30
        gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
          cd /opt/nyapsys/training
          nohup python3 train.py \
            --config 2b_moe \
            --data_path gs://$GCS_BUCKET/tokenized/ \
            --output_dir $CHECKPOINT_DIR/ \
            --resume_from_checkpoint latest \
            > /tmp/train.log 2>&1 &
          echo Resumed training
        "
      fi
    fi

    # Show latest log snippet
    gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="tail -3 /tmp/train.log 2>/dev/null" || true
    echo "  [$(date '+%H:%M')] Still training..."
    sleep 600
  done
}

# ── PHASE 7: Instruction Tuning ──────────────────────────
phase_instruction_tune() {
  step "Phase 7: Instruction tuning with LoRA (~12-15 hours)"

  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
    set -e
    cd /opt/nyapsys/training

    # Prepare finetune data if not already done
    if [ ! -f training/data/finetune/train.jsonl ]; then
      echo 'Finetune data not found — using pretrain data as fallback'
      mkdir -p training/data/finetune
      head -10000 training/data/tokenized/train.jsonl > training/data/finetune/train.jsonl
    fi

    python3 instruction_tune.py \
      --base_model gs://$GCS_BUCKET/checkpoints/hf_model \
      --data_path training/data/finetune/ \
      --output_dir $INSTRUCT_DIR/ \
      --lora_r 32 \
      --lora_alpha 64 \
      --epochs 3 \
      --learning_rate 2e-4
  "
  ok "Instruction tuning complete"
}

# ── PHASE 8: Export GGUF ─────────────────────────────────
phase_export() {
  step "Phase 8: Exporting GGUF model"

  gcloud compute ssh "$INSTANCE_NAME" --zone="$GCP_ZONE" --command="
    set -e
    cd /opt/nyapsys/training

    python3 merge_and_export.py \
      --model_path gs://$GCS_BUCKET/instruct-output/final \
      --output_path ./Nyapsys-2B-MoE.Q4_K_M.gguf

    ls -lh ./Nyapsys-2B-MoE.Q4_K_M.gguf

    python3 upload_to_gcs.py \
      --file ./Nyapsys-2B-MoE.Q4_K_M.gguf \
      --bucket $MODEL_BUCKET
  "
  ok "GGUF exported and uploaded to gs://$MODEL_BUCKET/"
}

# ── PHASE 9: Cleanup ─────────────────────────────────────
phase_cleanup() {
  step "Phase 9: Cleaning up GCP resources"

  read -p "Delete training instance? (y/n): " answer
  if [ "$answer" = "y" ]; then
    gcloud compute instances delete "$INSTANCE_NAME" --zone="$GCP_ZONE" --quiet
    ok "Instance deleted — billing stopped"
  else
    warn "Instance still running. Delete it manually to stop billing:"
    echo "  gcloud compute instances delete $INSTANCE_NAME --zone=$GCP_ZONE --quiet"
  fi
}

# ── PHASE 10: Download Model ─────────────────────────────
phase_download() {
  step "Phase 10: Downloading model to Mac"

  bash "$SCRIPT_DIR/download_model.sh"

  step "Running smoke test on local model"
  echo "  (Start llama-server in another terminal first, or skip)"
  echo ""
  echo "  llama-server --model ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf --port 8080 --ctx-size 512"
  echo ""
  echo "  Then test:"
  echo "  curl http://127.0.0.1:8080/v1/chat/completions -d '{\"model\":\"nyapsys\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in one sentence.\"}],\"max_tokens\":32}'"
}

# ── ENTRY POINT ───────────────────────────────────────────
case "${1:-full}" in
  full)
    phase_prerequisites
    phase_data_prep
    phase_create_instance
    phase_setup_instance
    phase_smoke_test
    phase_pretrain
    phase_wait_pretrain
    phase_instruction_tune
    phase_export
    phase_cleanup
    phase_download
    echo ""
    echo -e "  ${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "  ${GREEN}${BOLD}  Training pipeline complete!${RESET}"
    echo -e "  ${GREEN}${BOLD}  Model ready at ~/volumes/models/${RESET}"
    echo -e "  ${GREEN}${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo ""
    ;;
  prereq)  phase_prerequisites ;;
  data)    phase_data_prep ;;
  instance) phase_create_instance ;;
  setup)   phase_setup_instance ;;
  smoke)   phase_smoke_test ;;
  pretrain) phase_pretrain ;;
  wait)    phase_wait_pretrain ;;
  finetune) phase_instruction_tune ;;
  export)  phase_export ;;
  cleanup) phase_cleanup ;;
  download) phase_download ;;
  *)
    echo ""
    echo -e "  ${BOLD}Nyapsys Training Pipeline${RESET}"
    echo ""
    echo -e "  Usage: ${CYAN}bash run_training_pipeline.sh [command]${RESET}"
    echo ""
    echo -e "  ${BOLD}Commands:${RESET}"
    echo -e "  ${CYAN}full${RESET}       Run entire pipeline (default)"
    echo -e "  ${CYAN}prereq${RESET}     Check prerequisites only"
    echo -e "  ${CYAN}data${RESET}       Prepare and upload data"
    echo -e "  ${CYAN}instance${RESET}   Create GCP instance"
    echo -e "  ${CYAN}setup${RESET}      Setup instance"
    echo -e "  ${CYAN}smoke${RESET}      Run smoke test"
    echo -e "  ${CYAN}pretrain${RESET}   Start pretraining"
    echo -e "  ${CYAN}wait${RESET}       Wait for pretraining to finish"
    echo -e "  ${CYAN}finetune${RESET}   Run instruction tuning"
    echo -e "  ${CYAN}export${RESET}     Export GGUF"
    echo -e "  ${CYAN}cleanup${RESET}    Delete instance"
    echo -e "  ${CYAN}download${RESET}   Download model to Mac"
    echo ""
    ;;
esac
