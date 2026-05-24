# AGENTS.md — Nyapsys AI Agent

Read this file in full before writing a single line of code. Every architectural decision is final unless explicitly marked as a choice. Build strictly in the order defined at the bottom of this file. Yes, this is a full rewrite. Yes, everything changed. Start from scratch.

---

## What Nyapsys Does

Nyapsys is a self-hosted, always-on AI agent with three core capabilities:

- **Answers questions** — conversational Q&A over a from-scratch 2B MoE model
- **Reads files** — PDF, DOCX, TXT, Markdown, CSV, JSON uploaded by the user
- **Understands images** — via multimodal vision pipeline

It is NOT a wrapper around a third-party API. The LLM runs entirely on a MacBook Air M3, served locally via llama.cpp. The frontend is a static Next.js app on Firebase. There is no cloud VM, no Kubernetes, no managed database. The entire inference stack runs on the Mac.

---

## Final Model Decision

| Property | Value |
|---|---|---|
| Architecture | Sparse Mixture-of-Experts (MoE) transformer decoder, built from scratch in PyTorch |
| Total parameters | ~1.59B (4 experts × 400M, 18 layers, top-2) |
| Active params per token | ~795M (top-2 routing — only 2 experts fire per token) |
| Format (inference) | GGUF Q4_K_M |
| Size on disk | ~1.0GB (est.) |
| RAM at runtime | ~2–2.5GB |
| Inference speed | ~80–100 tokens/sec on M3 Air (pays 795M compute cost) |
| Training | GCP Compute Engine L4 GPU (24GB VRAM), one-time, from scratch |
| Base | No base model — true from-scratch pretraining + instruction tuning |

### Why 18 Layers (not 32)

Reduced from the original 32-layer 2B MoE plan to **18 layers (~1.59B total)** for two reasons:

1. **Better tokens/param ratio.** With a 4B token dataset, 32 layers (2B params) gives only ~2 tokens/param — too data-poor for good convergence. 18 layers (~1.59B) gives ~2.5 tokens/param.
2. **Lower cost.** Fewer layers = faster training (~29 days vs ~36 days), fitting comfortably within the $300 GCP credit at $284 total (including 200GB pd-ssd + instruction tuning).

### MoE design — how it works

Standard sparse MoE following the Mixtral architecture:

- Attention layers are **shared** — not per-expert. Every token uses the same attention weights.
- FFN/MLP layers are **per-expert** — 4 independent FFN blocks per transformer layer.
- A learned router (small linear layer) picks the **top-2 experts** per token.
- Auxiliary load-balancing loss (`router_aux_loss_coef: 0.01`) prevents all tokens routing to one expert.
- Result: 2B total params, ~1B active compute per token.

Model is NOT committed to git. The GGUF lives at `~/volumes/models/` on the Mac. Use `scripts/download_model.sh` to pull from GCP Cloud Storage onto a fresh machine. Never hardcode model logic anywhere — all inference goes through `backend/app/model.py` only.

---

## Architecture Overview

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GOOGLE CLOUD PLATFORM  (one-time training — destroy after)
  $300 credit / 90 days — covers full training run + retrains
  GPU: L4 24GB VRAM (~$0.70–1.00/hr on-demand, ~$0.30–0.40/hr spot)
  → Upgrade free trial to paid account first (unlocks GPUs)
  → Pretrain 2B MoE (4×500M experts) from scratch (~70–90 hrs spot)
  → Instruction tune via LoRA (~12–15 hrs spot)
  → Export GGUF Q4_K_M → Upload to GCP Cloud Storage → Destroy instance
  Total first run: ~$37–48 of $300
  Remaining for retrains: ~$252–263
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ↓ one-time GGUF transfer (~1.3GB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MACBOOK AIR M3 16GB  (always-on production server)
  Amphetamine keeps display + system awake 24/7
  $0/month forever
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[User browser / mobile]
        │
        ▼
Firebase Hosting  ←── GitHub Actions deploys Next.js static export
(free, global CDN)
        │
        │ HTTPS — Cloudflare Tunnel (free, no port forwarding needed)
        ▼
Cloudflare Tunnel daemon  (running on Mac as launchd service)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  MacBook Air M3 16GB  (localhost)                   │
│                                                     │
│  llama.cpp server      → 127.0.0.1:8080             │
│    Nyapsys-2B-MoE.Q4_K_M.gguf                      │
│    OpenAI-compatible API                            │
│    ~70–90 tok/s on M3 Metal (1B active/token)      │
│                                                     │
│  FastAPI backend       → 127.0.0.1:8000             │
│  ChromaDB              → 127.0.0.1:8001             │
│  SQLite                → ~/volumes/sqlite/          │
└─────────────────────────────────────────────────────┘
```

---

## Cost Breakdown

| Phase | Time | Spot Cost | Credit Remaining |
|---|---|---|---|
| Pretraining (6B tokens) | ~70–90 hrs | ~$28–36 | ~$264–272 |
| Instruction tuning (LoRA) | ~12–15 hrs | ~$5–7 | ~$257–267 |
| Ablations + smoke tests | ~10 hrs | ~$4–5 | ~$252–263 |
| **First run total** | **~95–115 hrs** | **~$37–48** | **~$252–263** |
| Retrain budget (word salad recovery) | up to ~600 hrs | up to ~$240 | ~$12–23 |

You can retrain the full 2B MoE **5–6 times** before exhausting the credit. If the first run produces garbage, destroy the instance, fix the issue (see Word Salad Diagnosis Guide below), spin up a new spot instance and resume from the last GCS checkpoint.

---

## Dataset Plan

### Pretraining Dataset — 6B tokens total

| Dataset | HuggingFace ID | Purpose | Tokens | Mix % |
|---|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | High-quality educational web text — best for reasoning | 2.1B | 35% |
| Stack v2 (filtered) | `bigcode/the-stack-v2-train-smol-ids` | Code across 600+ languages | 1.2B | 20% |
| Wikipedia EN | `wikimedia/wikipedia` | Factual grounding | 600M | 10% |
| OpenWebMath | `open-web-math/open-web-math` | Mathematical reasoning, proofs, step-by-step | 600M | 10% |
| Books3 subset | `the_pile` books split | Long-form reasoning, document structure | 900M | 15% |
| Dolma CC (filtered) | `allenai/dolma` | Diverse general knowledge | 600M | 10% |

**Why FineWeb-Edu over raw FineWeb:** filtered for educational quality — significantly better for reasoning and explanation tasks than raw web crawl data.

**Why OpenWebMath:** mathematical reasoning generalises to all structured reasoning, not just maths. Step-by-step thinking, proof structure, and logical chaining all improve with this data.

### Fine-tuning Dataset — ~120K examples

| Dataset | HuggingFace ID | Purpose | Examples |
|---|---|---|---|
| OpenHermes 2.5 | `teknium/OpenHermes-2.5` | Best general instruction dataset — reasoning, Q&A, conversation | 50K |
| CodeAlpaca 20K | `sahil2801/CodeAlpaca-20k` | Code generation + explanation | 20K |
| DocVQA-style JSONL | custom (see below) | Document Q&A — PDFs, reports, tables | 15K |
| Camel-AI Science | `camel-ai/science` | Complex topic explanation, multi-step reasoning | 15K |
| MathInstruct | `TIGER-Lab/MathInstruct` | Step-by-step reasoning chains | 12K |
| Custom Nyapsys JSONL | `training/data/custom/` | Personality, system behaviour, edge cases | 8K |

### Custom Nyapsys JSONL (8K examples) — most important

This is what makes it *Nyapsys* and not a generic model. Cover these cases:

- How it introduces itself ("I'm Nyapsys, running locally on your Mac")
- How it handles uncertainty ("I don't know" vs hallucinating)
- File analysis format (PDF summary style, CSV interpretation, table reading)
- Refusals and edge cases
- Preferred response length and tone
- Multi-turn conversation coherence

**How to generate 8K examples cheaply:**
1. Write 50–100 seed examples by hand in `training/data/custom/seeds.jsonl`
2. Run `training/expand_custom_dataset.py` — uses Claude API (~$8–12) to synthetically expand seeds to 8K
3. Review 200 random samples manually before using

Format for every example (pretrain + finetune):
```json
{"messages": [
  {"role": "system", "content": "You are Nyapsys, a self-hosted AI assistant..."},
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

---

## Training Pipeline — Full Guide

### Overview

```
Phase 1: Data prep on Mac M3 (no GPU needed)
Phase 2: Tokenizer (reuse from 240M if vocab_size matches, else retrain)
Phase 3: Smoke test on GCP L4 with 1% data
Phase 4: Full pretraining — 6B tokens, ~70–90 hrs
Phase 5: Evaluate — is it word salad? (see diagnosis guide)
Phase 6: Instruction tuning — LoRA, ~12–15 hrs
Phase 7: Evaluate again
Phase 8: Export GGUF, upload to GCS, destroy instance
```

---

### Critical First Step — Unlock GPUs

GCP's free trial blocks GPU access. You **must** upgrade to a paid billing account first:

1. Sign up at cloud.google.com/free
2. Go to Billing → Upgrade to paid account
3. Your $300 credit carries over automatically — you won't be charged until it runs out
4. Request GPU quota: IAM & Admin → Quotas → search `NVIDIA_L4_GPUS` → request increase to 1

---

### Phase 1 — Data Preparation (Mac M3)

```bash
cd training
pip install -r requirements.txt

# Download and tokenize all datasets
python prepare_dataset.py

# This script:
# 1. Downloads each dataset from HuggingFace
# 2. Filters low-quality samples (perplexity filter, length filter)
# 3. Tokenizes with your tokenizer
# 4. Writes shards to training/data/tokenized/
# 5. Prints token count per dataset and total

# Upload tokenized data to GCS
gsutil -m cp -r training/data/tokenized/ gs://nyapsys-training/tokenized/
```

`training/prepare_dataset.py` must:
- Enforce the mix percentages from the table above
- Filter samples shorter than 64 tokens (noise)
- Filter samples longer than 4096 tokens (truncate, don't discard)
- Deduplicate using MinHash (use `datasketch` library)
- Print a final token count report before uploading

---

### Phase 2 — Tokenizer

If carrying over the tokenizer from the existing 240M model and `vocab_size == 32000`, skip this step.

If starting fresh:

```python
# training/tokenizer_train.py
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(
    files=["training/data/raw/corpus.txt"],
    vocab_size=32000,
    min_frequency=2,
    special_tokens=["<|pad|>", "<|eos|>", "<|bos|>", "<|unk|>", "<|sep|>"]
)
tokenizer.save_model("training/tokenizer/")
```

**NEVER change the tokenizer after pretraining begins.** vocab_size must be identical between pretraining and instruction tuning.

---

### Phase 3 — Model Config

```python
# training/model_config.py

# 1.59B MoE config — PRIMARY TARGET (actual: CONFIG_3B_MOE in model_config.py)
CONFIG_3B_MOE = {
    "vocab_size": 32000,
    "num_hidden_layers": 18,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,           # GQA — reduces KV cache memory
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
    "num_experts": 4,
    "num_experts_per_token": 2,         # top-2 routing = 795M active per token
    "expert_intermediate_size": 4096,
    "router_aux_loss_coef": 0.03,
}
# Total params: ~1.59B. Active per token: ~795M. VRAM during training: ~10–12GB.
```

---

### Phase 4 — GCP L4 Instance Setup

```bash
# Install gcloud on Mac if not already done
brew install --cask google-cloud-sdk
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID

# Create spot instance
gcloud compute instances create nyapsys-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --boot-disk-type=pd-ssd \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP

gcloud compute ssh nyapsys-training --zone=us-central1-a

# Verify GPU
nvidia-smi   # should show L4 24GB

# Clone repo and install
git clone https://github.com/YOUR_USERNAME/nyapsys /opt/nyapsys
cd /opt/nyapsys
pip install -r training/requirements.txt

# Pull tokenized data from GCS
gsutil -m cp -r gs://nyapsys-training/tokenized/ training/data/tokenized/
```

---

### Phase 5 — Smoke Test (1% data, ~30 min)

Always do this before the full run. Catches config bugs cheaply.

```bash
python training/train.py \
  --config 2b_moe \
  --data_path training/data/tokenized/ \
  --output_dir gs://nyapsys-training/checkpoints/ \
  --max_steps 500 \
  --smoke_test True

# What to check:
# ✅ Loss starts ~10–11 (random for vocab_size=32000) and drops within 100 steps
# ✅ Router load balance: each expert ~25% (check logs every 100 steps)
# ✅ No OOM errors
# ✅ Checkpoint saves to GCS successfully
# ✅ GPU utilisation >60% (nvidia-smi)
```

If router shows one expert at 80%+, increase `router_aux_loss_coef` to `0.02` before full run.

---

### Phase 6 — Full Pretraining

```bash
python training/train.py \
  --config 2b_moe \
  --data_path gs://nyapsys-training/tokenized/ \
  --output_dir gs://nyapsys-training/checkpoints/ \
  --resume_from_checkpoint latest
```

**training/train.py requirements:**
- `torch.bfloat16` mixed precision
- `model.gradient_checkpointing_enable()` — critical for MoE memory
- AdamW, cosine LR schedule, 2000 step warmup
- Learning rate: 3e-4 peak → 3e-5 decay
- Batch size 4, gradient accumulation 8 (effective batch 32)
- **Checkpoint every 2000 steps to GCS — never local disk only**
- Log loss + router stats every 100 steps

```python
def compute_loss(logits, labels, router_logits, num_experts, num_experts_per_token):
    ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
    router_probs = F.softmax(router_logits, dim=-1)
    expert_usage = router_probs.mean(dim=0)
    target_usage = torch.ones_like(expert_usage) / num_experts
    aux_loss = F.mse_loss(expert_usage, target_usage)
    return ce_loss + CONFIG_2B_MOE["router_aux_loss_coef"] * aux_loss

def log_router_stats(router_logits):
    probs = F.softmax(router_logits, dim=-1).mean(dim=0)
    print(f"Expert load: {[f'{p:.2%}' for p in probs.tolist()]}")
    # Healthy: ~25% each. Collapsed: one expert at 80%+
```

**Expected timeline:**
- Steps at 1,000–1,400 tok/s throughput
- 6B tokens ÷ 1,200 tok/s ≈ 72 hours
- You will likely get preempted 2–4 times on spot — each time spin up a new instance and pass `--resume_from_checkpoint latest`

---

### Phase 7 — Instruction Tuning (LoRA)

```bash
python training/instruction_tune.py \
  --base_model gs://nyapsys-training/checkpoints/final \
  --data_path training/data/finetune/ \
  --output_dir gs://nyapsys-training/instruct-output \
  --lora_r 32 \
  --lora_alpha 64 \
  --epochs 3 \
  --learning_rate 2e-4
```

**training/instruction_tune.py requirements:**
- Uses `trl.SFTTrainer` + `peft.LoraConfig`
- LoRA target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`, `gate` (router)
- Including `gate` in LoRA targets lets the model re-specialise experts during tuning
- Max sequence length: 2048
- Pack short examples to fill context window (improves GPU utilisation)
- Save merged checkpoint (base + LoRA weights merged) for GGUF export

---

### Phase 8 — Export GGUF

```bash
python training/merge_and_export.py \
  --model_path gs://nyapsys-training/instruct-output \
  --output_path ./Nyapsys-2B-MoE.Q4_K_M.gguf

ls -lh ./Nyapsys-2B-MoE.Q4_K_M.gguf   # should be ~1.2–1.4GB

python training/upload_to_gcs.py \
  --file ./Nyapsys-2B-MoE.Q4_K_M.gguf \
  --bucket nyapsys-models
```

`merge_and_export.py` steps:
1. Load base checkpoint + LoRA weights
2. Merge LoRA into base weights
3. Save as HuggingFace format
4. Run `llama.cpp/convert_hf_to_gguf.py --outtype q4_k_m`
5. Verify output size is 1.1–1.5GB

---

### Phase 9 — Destroy GCP Instance

```bash
gcloud compute instances delete nyapsys-training \
  --zone=us-central1-a --quiet
echo "GCP instance destroyed. Billing stopped."
```

Verify in GCP Console → Compute Engine → VM instances — must show empty.

---

## Word Salad Diagnosis Guide

If the model outputs gibberish, repetitive loops, or incoherent text, do not panic. This is fixable. Work through this checklist before retraining.

### Step 1 — Identify the failure mode

Run these test prompts on the raw pretrained checkpoint (before instruction tuning):

```bash
# Start llama-server with pretrained GGUF (no instruct tuning yet)
llama-server --model ./Nyapsys-2B-MoE-pretrain.gguf --port 8080

# Test 1: Continuation (pretrain behaviour — should complete naturally)
curl http://127.0.0.1:8080/v1/completions -d '{
  "prompt": "The capital of France is",
  "max_tokens": 20
}'
# Expected: "Paris. France is a country in Western Europe..."
# Word salad: "the the the the" or "xzqk lmnop"

# Test 2: Repetition check
curl http://127.0.0.1:8080/v1/completions -d '{
  "prompt": "Photosynthesis is the process by which",
  "max_tokens": 100
}'
# Expected: coherent continuation
# Failure: looping the same phrase endlessly

# Test 3: Router sanity (run during training, not inference)
# Check logs — if one expert handles 80%+ of tokens, router collapsed
```

### Step 2 — Diagnose by symptom

| Symptom | Likely cause | Fix |
|---|---|---|
| Repeating the same token/phrase | LR too high, or no repetition penalty | Reduce peak LR to 1e-4, add `repetition_penalty: 1.1` at inference |
| Complete gibberish from step 1 | Tokenizer mismatch | Verify `vocab_size` matches between tokenizer and model config |
| Loss never drops below 4.0 | Model too small for data complexity, or bad data | Check data quality — run perplexity filter, remove low-quality shards |
| Loss drops then spikes repeatedly | LR too high or bad data shard | Reduce LR, add gradient clipping (`max_grad_norm: 1.0`) |
| One expert handles 80%+ tokens | Router collapsed | Increase `router_aux_loss_coef` from 0.01 to 0.02–0.05 |
| Coherent pretrain but gibberish after finetuning | LR too high in LoRA | Reduce LoRA LR to 5e-5, reduce epochs to 1–2 |
| Coherent pretrain but ignores instructions | Not enough instruction data | Add more OpenHermes examples, increase epochs to 4 |
| Repeats the user's question back | System prompt not applied correctly | Check chat template formatting in `instruction_tune.py` |
| Good English but factually wrong everything | Too few training tokens | Increase token budget to 10B, retrain |

### Step 3 — Quick eval before full retrain

Before spending another 70+ hours retraining, run a fast eval:

```bash
# training/eval_quick.py
# Tests 200 samples from a held-out eval set
# Measures: perplexity, repetition rate, instruction-following rate
python training/eval_quick.py \
  --model ./Nyapsys-2B-MoE.Q4_K_M.gguf \
  --eval_data training/data/eval/held_out.jsonl \
  --n_samples 200

# Healthy numbers:
# Perplexity: < 8.0 (pretrain), < 4.0 (after finetuning)
# Repetition rate: < 5%
# Instruction following: > 70% (does it actually answer the question?)
```

### Step 4 — Retrain decision tree

```
Is perplexity > 12 after pretraining?
  YES → Data quality issue. Re-run prepare_dataset.py with stricter filters.
        Increase minimum token length to 128. Retrain from scratch.
  NO  → Continue to next check.

Is router collapsed (one expert > 60%)?
  YES → Increase router_aux_loss_coef to 0.03. Resume from last checkpoint.
  NO  → Continue.

Is perplexity good but instruction-following < 50%?
  YES → Finetuning issue only. Don't retrain from scratch.
        Fix instruction_tune.py, re-run finetuning only (~12 hrs, ~$5).
  NO  → Continue.

Is everything good on eval but bad in practice?
  YES → Custom JSONL quality issue. Improve your seed examples.
        Re-run finetuning only with better custom data.
```

### Step 5 — Resuming a retrain

```bash
# Spin up new spot instance (old one was destroyed)
gcloud compute instances create nyapsys-training-v2 \
  --zone=us-central1-a \
  [same flags as before]

# Resume from last good GCS checkpoint
python training/train.py \
  --config 2b_moe \
  --data_path gs://nyapsys-training/tokenized/ \
  --output_dir gs://nyapsys-training/checkpoints-v2/ \
  --resume_from_checkpoint gs://nyapsys-training/checkpoints/step-XXXXX
```

**Credit budget for retrains:**
- Each full retrain: ~$37–48
- Credit remaining after first run: ~$252–263
- You can retrain **5–6 full times** before exhausting credit
- Finetuning-only retrain: ~$5–7 (much cheaper — do this first if pretrain looks good)

---

## Pulling the Model from GCS to Your Mac (scripts/download_model.sh)

### Step 1 — Authenticate

```bash
gcloud auth login
gcloud config set project YOUR_GCP_PROJECT_ID
```

### Step 2 — Download the GGUF

```bash
mkdir -p ~/volumes/models

# Faster: parallel composite download
gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp \
  gs://nyapsys-models/Nyapsys-2B-MoE.Q4_K_M.gguf \
  ~/volumes/models/
```

### Step 3 — Verify integrity

```bash
ls -lh ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf  # should be ~1.2–1.4GB
gsutil hash gs://nyapsys-models/Nyapsys-2B-MoE.Q4_K_M.gguf
md5 ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf
# Hashes must match
```

### Step 4 — Smoke test

```bash
llama-server \
  --model ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 512

curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nyapsys","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":32}'
```

### Step 5 — Full script (scripts/download_model.sh)

```bash
#!/bin/bash
set -e

BUCKET="nyapsys-models"
FILENAME="Nyapsys-2B-MoE.Q4_K_M.gguf"
DEST="$HOME/volumes/models/$FILENAME"
GCS_PATH="gs://$BUCKET/$FILENAME"

echo "=== Nyapsys model download ==="
mkdir -p "$HOME/volumes/models"

if ! gcloud auth print-access-token > /dev/null 2>&1; then
  echo "Not authenticated. Run: gcloud auth login"
  exit 1
fi

gsutil -o GSUtil:parallel_composite_upload_threshold=150M cp "$GCS_PATH" "$DEST"

FILE_SIZE=$(stat -f%z "$DEST" 2>/dev/null || stat -c%s "$DEST")
if [ "$FILE_SIZE" -lt 1000000000 ]; then
  echo "ERROR: File too small ($FILE_SIZE bytes). Download incomplete."
  exit 1
fi

echo "Downloaded: $(ls -lh "$DEST" | awk '{print $5}')"
echo "=== Done. Model ready at $DEST ==="
```

---

## Mac Setup — Always-On Local Server (scripts/setup_mac.sh)

```bash
#!/bin/bash
set -e
echo "=== Nyapsys Mac setup ==="

brew install llama.cpp python@3.11 git wget
pip3.11 install -r backend/requirements.txt

mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite

[ -d ~/nyapsys ] || git clone https://github.com/YOUR_USERNAME/nyapsys ~/nyapsys

bash ~/nyapsys/scripts/download_model.sh

brew install cloudflared
bash ~/nyapsys/scripts/setup_tunnel.sh

cp ~/nyapsys/scripts/com.nyapsys.backend.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyapsys.backend.plist

echo "=== Setup complete. Nyapsys is running. ==="
```

### launchd service (scripts/com.nyapsys.backend.plist)

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.nyapsys.backend</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/Users/YOUR_USERNAME/nyapsys/backend/run.sh</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key>
  <string>/Users/YOUR_USERNAME/nyapsys/logs/backend.log</string>
  <key>StandardErrorPath</key>
  <string>/Users/YOUR_USERNAME/nyapsys/logs/backend.error.log</string>
  <key>WorkingDirectory</key>
  <string>/Users/YOUR_USERNAME/nyapsys/backend</string>
</dict>
</plist>
```

### backend/run.sh

```bash
#!/bin/bash
set -e
source ~/nyapsys/.env

llama-server \
  --model ~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-predict 2048 \
  --threads 4 \
  --parallel 2 \
  --cont-batching &

LLAMA_PID=$!
until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do sleep 2; done
echo "llama-server ready"

python3.11 -m chromadb.cli.cli run \
  --path ~/volumes/chromadb \
  --host 127.0.0.1 \
  --port 8001 &

CHROMA_PID=$!
sleep 3

cd ~/nyapsys/backend
python3.11 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1

kill $LLAMA_PID $CHROMA_PID 2>/dev/null || true
```

---

## Cloudflare Tunnel Setup (scripts/setup_tunnel.sh)

```bash
#!/bin/bash
cloudflared tunnel login
cloudflared tunnel create nyapsys

cat > ~/.cloudflared/config.yml <<EOF
tunnel: nyapsys
credentials-file: ~/.cloudflared/$(cloudflared tunnel list --output json | jq -r '.[] | select(.name=="nyapsys") | .id').json

ingress:
  - hostname: api.yourdomain.com
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF

cloudflared tunnel route dns nyapsys api.yourdomain.com
cloudflared service install
```

---

## API Capabilities

### Outward API

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.yourdomain.com/v1",
    api_key="your-API_SECRET_KEY"
)

response = client.chat.completions.create(
    model="nyapsys",
    messages=[{"role": "user", "content": "Summarise this for me"}],
    stream=True
)
for chunk in response:
    print(chunk.choices[0].delta.content or "", end="")
```

### Tool API (backend/app/tools.py)

Available tools:
- `web_search` — live web search via SerpAPI (100 free/month)
- `get_current_time` — current date/time
- `run_python` — execute Python snippet, return stdout
- `read_local_file` — read file from Mac filesystem (sandboxed to home dir)

New env vars:
```env
SERPAPI_KEY=
TOOL_TIMEOUT_SECONDS=10
MAX_TOOL_ROUNDS=5
ALLOW_CODE_EXECUTION=true
```

---

## Database Design

### SQLite (aiosqlite) — ~/volumes/sqlite/nyapsys.db

```sql
CREATE TABLE IF NOT EXISTS conversations (
    id           TEXT PRIMARY KEY,
    title        TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    has_file         BOOLEAN DEFAULT FALSE,
    has_image        BOOLEAN DEFAULT FALSE,
    tokens_used      INTEGER,
    latency_ms       INTEGER,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS files (
    id               TEXT PRIMARY KEY,
    conversation_id  TEXT REFERENCES conversations(id) ON DELETE SET NULL,
    filename         TEXT NOT NULL,
    file_type        TEXT NOT NULL,
    file_size_bytes  INTEGER,
    chunk_count      INTEGER DEFAULT 0,
    chroma_ids       TEXT,
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    message_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,
    score       REAL CHECK(score BETWEEN 0.0 AND 1.0),
    feedback    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### ChromaDB — ~/volumes/chromadb/

Collection: `nyapsys_kb`
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`

---

## Backend API Endpoints

```
POST   /chat                         SSE token stream
POST   /v1/chat/completions          OpenAI-compatible
POST   /ingest                       File ingestion
GET    /conversations                List conversations
GET    /conversations/{id}/messages  Message history
DELETE /conversations/{id}           Delete + cleanup ChromaDB
GET    /v1/tools                     Available tools
POST   /v1/tools/call                Invoke a tool directly
POST   /feedback                     Rate a response
GET    /health                       Status check (no auth)
```

---

## Frontend Design

Static Next.js app deployed to Firebase. `output: 'export'` — no SSR, no API routes.

### Fonts

```html
<link href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
```

- Display/logo/headings: **Instrument Serif**
- Body/UI: **DM Sans**

### Colour palette (globals.css)

```css
:root {
  --cream:          #FAF7F2;
  --cream-2:        #F3EEE6;
  --cream-3:        #EAE3D7;
  --warm-brown:     #8B6F5E;
  --deep-brown:     #3D2B1F;
  --accent:         #C4784A;
  --accent-soft:    #F0E0D0;
  --text-primary:   #2C1F14;
  --text-secondary: #7A6355;
  --text-muted:     #B0A090;
  --border:         rgba(139, 111, 94, 0.15);
  --border-strong:  rgba(139, 111, 94, 0.28);
  --shadow-sm:      0 2px 8px rgba(60, 30, 10, 0.06);
  --shadow-md:      0 8px 32px rgba(60, 30, 10, 0.10);
  --shadow-lg:      0 20px 60px rgba(60, 30, 10, 0.14);
  --radius:         20px;
  --radius-sm:      12px;
  --radius-xs:      8px;
}
```

### Layout

```
┌─────────────────────────────────────────────┐
│  Topbar (fixed, 68px)                       │
│  [☰ menu]  nyap·sys  ────────  [● Running]  │
├─────────────────────────────────────────────┤
│  Messages (flex: 1, overflow-y: auto)       │
│  max-width: 720px, centered                 │
├─────────────────────────────────────────────┤
│  Input area (fixed bottom)                  │
│  [file chips if any]                        │
│  [textarea ──────────── 📎  →]              │
│  "Nyapsys runs entirely on your Mac..."     │
└─────────────────────────────────────────────┘
```

Sidebar: 280px, hidden by default, slides in with `0.35s cubic-bezier(0.4,0,0.2,1)`.

### Non-negotiables

- `output: 'export'` — no exceptions
- Fonts from Google Fonts — no substitutions
- Colours via CSS variables only — no hardcoded hex outside globals.css
- Logo: `nya` + italic `psys` in accent colour
- Sidebar hidden by default
- Starter pills fill textarea — do not auto-send
- Hint text always visible below input
- Mobile responsive at 375px viewport
- No login UI

---

## Environment Variables (.env.example)

```env
# Model
MODEL_PATH=~/volumes/models/Nyapsys-2B-MoE.Q4_K_M.gguf
LLAMA_HOST=http://127.0.0.1:8080
CTX_SIZE=4096
N_PREDICT=2048
TEMPERATURE=0.7

# Database
SQLITE_PATH=~/volumes/sqlite/nyapsys.db
CHROMA_HOST=http://127.0.0.1:8001
CHROMA_COLLECTION=nyapsys_kb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# RAG
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.4
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MAX_HISTORY_MESSAGES=10

# API
API_SECRET_KEY=replace-with-output-of-openssl-rand-hex-32

# Tools
SERPAPI_KEY=
TOOL_TIMEOUT_SECONDS=10
MAX_TOOL_ROUNDS=5
ALLOW_CODE_EXECUTION=true

# GCP
GCS_BUCKET=nyapsys-models
GOOGLE_APPLICATION_CREDENTIALS=~/nyapsys/gcp-key.json

# Firebase
FIREBASE_PROJECT_ID=your-firebase-project-id

# Frontend (injected at build time)
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_API_KEY=same-value-as-API_SECRET_KEY
```

---

## Repository Structure

```
nyapsys/
├── AGENTS.md
├── README.md
├── .env.example
├── .gitignore                             ← .env, volumes/, *.gguf, *.pt
│
├── .github/workflows/
│   ├── deploy-firebase.yml
│   └── deploy-backend.yml
│
├── scripts/
│   ├── download_model.sh
│   ├── setup_mac.sh
│   ├── setup_gcp_gpu.sh
│   ├── setup_tunnel.sh
│   └── export_gguf.sh
│
├── training/
│   ├── README.md
│   ├── requirements.txt
│   ├── dataset_config.py
│   ├── prepare_dataset.py
│   ├── expand_custom_dataset.py          ← synthetic data expansion
│   ├── tokenizer_train.py
│   ├── model_config.py
│   ├── train.py
│   ├── instruction_tune.py
│   ├── merge_and_export.py
│   ├── upload_to_gcs.py
│   └── eval_quick.py                     ← fast eval before/after training
│
├── backend/
│   ├── run.sh
│   ├── requirements.txt
│   └── app/
│       ├── main.py
│       ├── schemas.py
│       ├── agent.py
│       ├── model.py
│       ├── rag.py
│       ├── tools.py
│       ├── ingest.py
│       ├── db.py
│       └── health.py
│
├── frontend/
│   └── src/
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── MessageInput.tsx
│       │   └── FilePreview.tsx
│       ├── hooks/useChat.ts
│       └── lib/api.ts
│
└── volumes/                               ← GITIGNORED
    ├── models/
    ├── chromadb/
    └── sqlite/
```

---

## CI/CD

### GitHub Secrets

| Secret | How to get it |
|---|---|
| `FIREBASE_SERVICE_ACCOUNT` | Firebase console → Service accounts → Generate key |
| `MAC_SSH_KEY` | Private key for SSH via Tailscale |
| `MAC_TAILSCALE_IP` | Tailscale admin console → Mac device IP |
| `API_SECRET_KEY` | `openssl rand -hex 32` |
| `NEXT_PUBLIC_API_URL` | `https://api.yourdomain.com` |
| `NEXT_PUBLIC_API_KEY` | Same as API_SECRET_KEY |

### deploy-firebase.yml

Triggers on push to `main` when `frontend/**` changes. Runs `npm ci`, `npm run build`, deploys to Firebase Hosting.

### deploy-backend.yml

Triggers on push to `main` when `backend/**` changes. SSHes to Mac via Tailscale, git pulls, pip installs, restarts launchd service, curls `/health` to verify.

---

## README Creation Guide

README.md must cover (in order): title + description, architecture diagram (abbreviated), requirements, quick start (4 steps), training (3 sentences + link to training/README.md), model download (3 commands), dev setup, deployment, project structure (top-level only), contributing (2 sentences), license.

Rules: never duplicate large code blocks from AGENTS.md, keep under 200 lines, every command must work as written, no marketing language.

Sync triggers — update README.md immediately when any of these change in AGENTS.md: model filename, GCS bucket name, port numbers, script names, build order.

---

## Build Order

```
PHASE 1 — Training pipeline
 1.  training/requirements.txt
 2.  training/dataset_config.py
 3.  training/tokenizer_train.py
 4.  training/model_config.py             ← CONFIG_2B_MOE
 5.  training/prepare_dataset.py
 6.  training/expand_custom_dataset.py
 7.  training/train.py
 8.  training/instruction_tune.py
 9.  training/merge_and_export.py
10.  training/upload_to_gcs.py
11.  training/eval_quick.py
12.  training/README.md
13.  scripts/setup_gcp_gpu.sh

PHASE 2 — Mac setup
14.  scripts/setup_mac.sh
15.  scripts/download_model.sh
16.  scripts/setup_tunnel.sh
17.  scripts/com.nyapsys.backend.plist
18.  backend/run.sh

PHASE 3 — Backend
19.  backend/requirements.txt
20.  backend/app/schemas.py
21.  backend/app/db.py
     TEST: python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
22.  backend/app/rag.py
23.  backend/app/model.py
     TEST: curl http://127.0.0.1:8080/v1/chat/completions
24.  backend/app/tools.py
25.  backend/app/ingest.py
     TEST: upload PDF + JPG, verify ChromaDB + SQLite
26.  backend/app/agent.py
27.  backend/app/health.py
28.  backend/app/main.py
     TEST: bash backend/run.sh; curl -X POST http://127.0.0.1:8000/chat

PHASE 4 — Frontend
29.  frontend/next.config.js              ← output: 'export' FIRST
30.  frontend/firebase.json
31.  frontend/package.json
32.  frontend/src/lib/api.ts
33.  frontend/src/hooks/useChat.ts
34.  frontend/src/components/FilePreview.tsx
35.  frontend/src/components/MessageBubble.tsx
36.  frontend/src/components/MessageInput.tsx
37.  frontend/src/components/ChatWindow.tsx
38.  frontend/src/app/layout.tsx
39.  frontend/src/app/page.tsx
     TEST: npm run build → open in browser → send message

PHASE 5 — Docs
40.  README.md

PHASE 6 — CI/CD
41.  .github/workflows/deploy-firebase.yml
42.  .github/workflows/deploy-backend.yml
     TEST: push to main → both workflows pass → end-to-end message test
```

---

## Non-Negotiable Constraints

1. Static frontend — `output: 'export'`, no SSR, no API routes
2. All inference on Mac — nothing moves to cloud after training
3. No Docker — native Mac processes via launchd
4. Model files never in git
5. .env never in git
6. Always stream — `/chat` endpoint is SSE, never buffered
7. launchd manages everything — `KeepAlive: true`
8. Cloudflare Tunnel for public access — no open router ports
9. aiosqlite everywhere — never synchronous sqlite3 in async handlers
10. Single ChromaDB collection — `nyapsys_kb`, scoped by `conversation_id`
11. Batch embedding — never one chunk at a time
12. Destroy GCP instance after training
13. Tailscale for SSH deploys
14. **1.59B MoE (4×400M, 18 layers, top-2) is the target** — 795M active per token
15. README.md stays in sync with AGENTS.md