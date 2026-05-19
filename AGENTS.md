# AGENTS.md — Nyapsys AI Agent
> Read this file **in full** before writing a single line of code.
> Every architectural decision is final unless explicitly marked as a choice.
> Build strictly in the order defined at the bottom of this file.
> Yes, this is a full rewrite. Yes, everything changed. Start from scratch.

---

## What Nyapsys Does

Nyapsys is a self-hosted, always-on AI agent with three core capabilities:

- **Answers questions** — conversational Q&A over a from-scratch 700M parameter model
- **Reads files** — PDF, DOCX, TXT, Markdown, CSV, JSON uploaded by the user
- **Understands images** — via multimodal vision pipeline

It is NOT a wrapper around a third-party API. The LLM runs entirely on a MacBook Air M3,
served locally via llama.cpp. The frontend is a static Next.js app on Firebase. There is
no cloud VM, no Kubernetes, no managed database. The entire inference stack runs on the Mac.

---

## Final Model Decision

| Property | Value |
|---|---|
| Architecture | Custom transformer decoder (GPT-style), built from scratch in PyTorch |
| Parameters | 700M |
| Format (inference) | GGUF Q4_K_M |
| Size on disk | ~400–450MB |
| RAM at runtime | ~600–800MB (trivial on M3 Air) |
| Inference speed | ~80–120 tokens/sec on M3 Air |
| Training | IBM Cloud L40S GPU, one-time, from scratch |
| Base | No base model — true from-scratch pretraining + instruction tuning |

### Why 700M from scratch
- Developer already built and trained a 240M model in PyTorch + HuggingFace Transformers
- 700M uses identical architecture — only config values change
- Q4_K_M GGUF fits in ~800MB RAM — runs on M3 Air with no performance impact
- Once trained, inference costs $0/month forever
- No Meta/Mistral licensing restrictions — this model is fully owned

### Model is NOT committed to git
The GGUF lives at `~/volumes/models/` on the Mac. Use `scripts/download_model.sh`
to pull from GCP Cloud Storage onto a fresh machine. Never hardcode model logic
anywhere — all inference goes through `backend/app/model.py` only.

---

## Architecture Overview

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  IBM CLOUD  (one-time training — destroy after)
  $1,000 credit / 120 days
  GPU: L40S 48GB VRAM (~$3–4/hr)
  → Pretrain 700M model from scratch (~60–100 hrs)
  → Instruction tune (~10–15 hrs)
  → Export GGUF Q4_K_M
  → Upload to GCP Cloud Storage
  → Destroy IBM instance immediately
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
                    ↓ one-time GGUF transfer (~450MB)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  MACBOOK AIR M3  (always-on production server)
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
│  MacBook Air M3  (localhost)                        │
│                                                     │
│  llama.cpp server      → 127.0.0.1:8080             │
│    Nyapsys-700M.Q4_K_M.gguf                        │
│    OpenAI-compatible API                            │
│    ~80–120 tok/s on M3 Metal                       │
│                                                     │
│  FastAPI backend       → 127.0.0.1:8000             │
│    Agent loop                                       │
│    File + image ingestion                           │
│    ChromaDB client                                  │
│    SQLite client                                    │
│    SSE streaming                                    │
│                                                     │
│  ChromaDB              → 127.0.0.1:8001             │
│    Vector store (RAG knowledge base)                │
│    Persisted to ~/volumes/chromadb/                 │
│                                                     │
│  SQLite                                             │
│    ~/volumes/sqlite/nyapsys.db                      │
│    Conversations, messages, files, evals            │
└─────────────────────────────────────────────────────┘

Storage (all local on Mac):
  ~/volumes/models/     ← GGUF weights
  ~/volumes/chromadb/   ← ChromaDB persistence
  ~/volumes/sqlite/     ← nyapsys.db
```

---

## Cost Breakdown

| Item | Cost |
|---|---|
| IBM Cloud training (one-time) | ~$210–460 — fully covered by $1,000 IBM credit |
| Firebase Hosting | $0 forever |
| Cloudflare Tunnel | $0 forever |
| GCP Cloud Storage (GGUF bucket) | ~$0.01/mo — negligible |
| Mac inference | $0 — already owned, ~10W draw |
| **Monthly ongoing** | **~$0** |

After training, Nyapsys costs nothing to run.

---

## Provider Split

| Phase | Provider | Purpose | Credit |
|---|---|---|---|
| Training (one-time) | IBM Cloud | L40S GPU, pretrain 700M from scratch, export GGUF | $1,000 / 120 days |
| Model storage | GCP Cloud Storage | Single bucket, GGUF backup + download | $300 credit / then ~$0.01/mo |
| Frontend | Firebase (Google) | Static Next.js hosting | Free forever |
| Tunnel | Cloudflare Tunnel | Expose local Mac server to internet, SSL, no open ports | Free forever |
| Inference | MacBook Air M3 | Everything — llama.cpp, FastAPI, ChromaDB, SQLite | $0 (already owned) |

---

## Repository Structure

```
nyapsys/
├── AGENTS.md                              ← this file
├── .env.example                           ← every env var documented
├── .gitignore                             ← .env, volumes/, *.gguf, *.pt, __pycache__
│
├── .github/
│   └── workflows/
│       ├── deploy-firebase.yml            ← frontend CI/CD (auto on push)
│       └── deploy-backend.yml            ← SSH deploy to Mac via Tailscale
│
├── scripts/
│   ├── download_model.sh                  ← pulls GGUF from GCP Cloud Storage
│   ├── setup_mac.sh                       ← one-time Mac setup (brew, llama.cpp, launchd)
│   ├── setup_ibm_gpu.sh                   ← IBM L40S instance prep for training
│   ├── setup_tunnel.sh                    ← installs + configures Cloudflare Tunnel as launchd
│   └── export_gguf.sh                     ← wraps llama.cpp convert script
│
├── training/
│   ├── README.md                          ← step-by-step training instructions
│   ├── requirements.txt                   ← torch, transformers, trl, peft, datasets, tokenizers
│   ├── dataset_config.py                 ← dataset mix ratios, preprocessing constants
│   ├── prepare_dataset.py                ← downloads HF datasets → tokenized JSONL
│   ├── tokenizer_train.py                ← trains BPE tokenizer on corpus
│   ├── model_config.py                   ← 700M architecture config (and 240M for reference)
│   ├── train.py                           ← from-scratch pretraining loop
│   ├── instruction_tune.py               ← LoRA instruction tuning after pretraining
│   ├── merge_and_export.py               ← merge LoRA → full model → GGUF Q4_K_M
│   └── upload_to_gcs.py                  ← uploads GGUF to GCP Cloud Storage
│
├── backend/
│   ├── run.sh                             ← starts llama-server + uvicorn (called by launchd)
│   ├── requirements.txt                   ← fastapi, uvicorn, chromadb, pymupdf, python-docx,
│   │                                         pillow, httpx, aiosqlite, sentence-transformers,
│   │                                         pandas, python-multipart
│   └── app/
│       ├── main.py                        ← FastAPI entry point, mounts all routers
│       ├── schemas.py                     ← ALL Pydantic models — write this first
│       ├── agent.py                       ← core agent loop
│       ├── model.py                       ← llama.cpp HTTP client
│       ├── rag.py                         ← ChromaDB init, embed, upsert, query
│       ├── ingest.py                      ← file + image ingestion pipeline
│       ├── db.py                          ← SQLite async CRUD via aiosqlite
│       └── health.py                      ← GET /health
│
├── frontend/
│   ├── package.json
│   ├── next.config.js                     ← output: 'export' — MUST have this
│   ├── .firebaserc
│   ├── firebase.json
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx
│       │   └── globals.css
│       ├── components/
│       │   ├── ChatWindow.tsx
│       │   ├── MessageBubble.tsx
│       │   ├── MessageInput.tsx
│       │   └── FilePreview.tsx
│       ├── hooks/
│       │   └── useChat.ts
│       └── lib/
│           └── api.ts
│
└── volumes/                               ← GITIGNORED — local Mac only
    ├── models/                            ← Nyapsys-700M.Q4_K_M.gguf
    ├── chromadb/                          ← ChromaDB persistence
    └── sqlite/                            ← nyapsys.db
```

---

## Training Pipeline — From Scratch on IBM Cloud (`training/`)

One-time process. Use IBM Cloud's $1,000 new-account credit (valid 120 days).
Destroy the IBM GPU instance the moment the GGUF is uploaded to GCP Cloud Storage.

### IBM GPU instance
- **Instance type:** `gx3-48x240x2l40s` — L40S 48GB VRAM
- **Cost:** ~$3–4/hr (covered by $1,000 IBM credit)
- **Estimated pretraining:** 60–100 hrs (~$180–400 total)
- **Estimated instruction tuning:** 10–15 hrs (~$30–60)
- **Total IBM spend:** ~$210–460 — well within $1,000 credit

### Why from scratch (not fine-tuning)
Developer already built and trained a 240M parameter model in PyTorch.
The 700M model uses the same architecture — just scaled config values.
Tokenizer, data pipeline, and training loop carry over with minimal changes.
This is a fully owned model with no third-party licensing.

### Timeline (fits 120-day IBM credit window)

```
Week 1–2   Mac M3      Data pipeline: download HF datasets, clean, tokenize
Week 2     Mac M3      Scale 240M config → 700M, update train.py
Week 3     IBM L40S    Smoke test: verify loop on 1% data, check loss curves
Week 4–9   IBM L40S    Full pretraining: 14B tokens, checkpoint every 2000 steps to GCS
Week 10    IBM L40S    Instruction tuning: LoRA on instruction datasets
Week 10    IBM L40S    Export GGUF, upload to GCS, destroy IBM instance
Week 11    Mac M3      Download GGUF, run locally, verify quality
```

### Step 1 — Prepare data on Mac M3 (`training/prepare_dataset.py`)

Dataset composition defined in `training/dataset_config.py`:

| Dataset | HuggingFace ID | Purpose | Mix % |
|---|---|---|---|
| FineWeb (sample 10BT) | `HuggingFaceFW/fineweb` | General knowledge pretraining | 40% |
| Wikipedia EN | `wikimedia/wikipedia` | Factual knowledge | 20% |
| Books3 subset | `the_pile` books split | Long-form reasoning | 15% |
| Alpaca Cleaned | `yahma/alpaca-cleaned` | Instruction following | 10% |
| SQuAD v2 | `rajpurkar/squad_v2` | Document Q&A | 8% |
| Custom JSONL | `training/data/custom/` | Nyapsys-specific examples | 7% |

**Target: ~14 billion tokens** for pretraining. Additional 50,000 instruction examples for tuning.

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py    # tokenizes → training/data/tokenized/

# Upload to GCS so IBM instance can pull it
gsutil -m cp -r training/data/tokenized/ gs://nyapsys-training/tokenized/
```

### Step 2 — Train tokenizer (`training/tokenizer_train.py`)

If carrying over the tokenizer from the existing 240M model, skip this step.
If starting truly fresh, train a BPE tokenizer on the corpus first:

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

**vocab_size MUST stay identical between pretraining and instruction tuning.**
Never change the tokenizer after pretraining begins.

### Step 3 — Scale 240M architecture → 700M (`training/model_config.py`)

Only these values change from the existing 240M model:

```python
# training/model_config.py

# Your 240M config (adjust to match your actual values)
CONFIG_240M = {
    "vocab_size": 32000,             # KEEP IDENTICAL IN ALL CONFIGS
    "num_hidden_layers": 12,
    "hidden_size": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 8,
    "intermediate_size": 4096,
    "max_position_embeddings": 2048,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}

# 700M config — same architecture, scaled numbers only
CONFIG_700M = {
    "vocab_size": 32000,             # MUST be identical to 240M
    "num_hidden_layers": 32,
    "hidden_size": 2048,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,        # GQA — reduces KV cache memory
    "intermediate_size": 5632,
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-5,
    "rope_theta": 10000.0,
}
```

### Step 4 — `training/train.py` implementation requirements

- Use `transformers.Trainer` or clean PyTorch loop (your existing loop carries over)
- `torch.bfloat16` mixed precision — halves memory, native on L40S
- `model.gradient_checkpointing_enable()` — reduces activation memory ~60%
- AdamW optimizer, cosine LR schedule, 2000 step warmup
- Learning rate: `3e-4` peak → `3e-5` decay
- Batch size 8, gradient accumulation 4 steps (effective batch = 32)
- Checkpoint every 2000 steps directly to GCS: `gsutil cp ckpt/ gs://nyapsys-training/checkpoints/`
- Log loss + tokens/sec every 100 steps to stdout
- **NEVER checkpoint only to local IBM disk** — instance can be interrupted

### Step 5 — Create IBM Cloud L40S instance

```bash
# Install IBM Cloud CLI
curl -fsSL https://clis.cloud.ibm.com/install/linux | sh
ibmcloud login --apikey YOUR_IBM_API_KEY
ibmcloud is instance-create nyapsys-training \
  --profile gx3-48x240x2l40s \
  --image ibm-ubuntu-22-04-minimal \
  --zone us-south-1 \
  --ssh-key YOUR_KEY_NAME

# Get IP and SSH in
ibmcloud is instance nyapsys-training --output json | \
  jq '.primary_network_interface.primary_ip.address'

ssh root@IBM_INSTANCE_IP
bash /dev/stdin < <(curl -s \
  https://raw.githubusercontent.com/YOUR_USERNAME/nyapsys/main/scripts/setup_ibm_gpu.sh)
```

`scripts/setup_ibm_gpu.sh` installs: Google Cloud SDK, PyTorch (CUDA), transformers, trl,
peft, accelerate, google-cloud-storage, llama.cpp (with CUDA). Pulls training code from
GitHub. Pulls tokenized data from GCS. Verifies GPU with `nvidia-smi`.

### Step 6 — Run pretraining

```bash
cd /opt/nyapsys
python training/train.py \
  --config 700m \
  --data_path gs://nyapsys-training/tokenized/ \
  --output_dir gs://nyapsys-training/checkpoints/ \
  --resume_from_checkpoint latest
```

Expected time: **60–100 hrs** on L40S. Leave running unattended.

### Step 7 — Instruction tuning (`training/instruction_tune.py`)

After pretraining, short instruction-tuning pass using LoRA (10–15 hrs):

```bash
python training/instruction_tune.py \
  --base_model gs://nyapsys-training/checkpoints/final \
  --output_dir gs://nyapsys-training/instruct-output \
  --lora_r 16 --lora_alpha 32 --epochs 3
```

Uses `trl.SFTTrainer` with `peft.LoraConfig`.
Target modules: `q_proj`, `v_proj`, `k_proj`, `o_proj`.

### Step 8 — Export GGUF and upload to GCS

```bash
python training/merge_and_export.py \
  --model_path gs://nyapsys-training/instruct-output \
  --output_path ./Nyapsys-700M.Q4_K_M.gguf

# Verify size (~400–450MB) before uploading
ls -lh ./Nyapsys-700M.Q4_K_M.gguf

python training/upload_to_gcs.py \
  --file ./Nyapsys-700M.Q4_K_M.gguf \
  --bucket nyapsys-models
```

`merge_and_export.py` steps: load base + LoRA → merge → save HF format →
run `llama.cpp/convert_hf_to_gguf.py` with `--outtype q4_k_m`.

### Step 9 — Destroy IBM instance immediately

```bash
ibmcloud is instance-delete nyapsys-training --force
echo "IBM instance destroyed. Billing stopped."
```

Total IBM spend: ~$210–460. Well within $1,000 credit. Nothing out of pocket.

---

## Mac Setup — Always-On Local Server (`scripts/setup_mac.sh`)

Run once on the Mac. Amphetamine handles sleep prevention — this script handles everything else.

```bash
#!/bin/bash
set -e
echo "=== Nyapsys Mac setup ==="

# 1. Install homebrew dependencies
brew install llama.cpp python@3.11 git wget

# 2. Install Python dependencies
pip3.11 install -r backend/requirements.txt

# 3. Create volume directories
mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite

# 4. Clone repo if not already present
[ -d ~/nyapsys ] || git clone https://github.com/YOUR_USERNAME/nyapsys ~/nyapsys

# 5. Download model from GCS
bash ~/nyapsys/scripts/download_model.sh

# 6. Install Cloudflare Tunnel
brew install cloudflared
bash ~/nyapsys/scripts/setup_tunnel.sh  # configures tunnel + installs launchd service

# 7. Install backend launchd service (auto-start on login)
cp ~/nyapsys/scripts/com.nyapsys.backend.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.nyapsys.backend.plist

echo "=== Setup complete. Nyapsys is running. ==="
```

### launchd service (`scripts/com.nyapsys.backend.plist`)

This keeps the backend + llama-server running permanently, restarting automatically on crash:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.nyapsys.backend</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>/Users/YOUR_USERNAME/nyapsys/backend/run.sh</string>
  </array>

  <key>RunAtLoad</key>
  <true/>

  <key>KeepAlive</key>
  <true/>

  <key>StandardOutPath</key>
  <string>/Users/YOUR_USERNAME/nyapsys/logs/backend.log</string>

  <key>StandardErrorPath</key>
  <string>/Users/YOUR_USERNAME/nyapsys/logs/backend.error.log</string>

  <key>WorkingDirectory</key>
  <string>/Users/YOUR_USERNAME/nyapsys/backend</string>
</dict>
</plist>
```

### `backend/run.sh` — starts everything

```bash
#!/bin/bash
set -e
source ~/nyapsys/.env

# Start llama.cpp server in background
llama-server \
  --model ~/volumes/models/Nyapsys-700M.Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8080 \
  --ctx-size 4096 \
  --n-predict 2048 \
  --threads 4 \
  --parallel 2 \
  --cont-batching &

LLAMA_PID=$!

# Wait for llama-server to be ready
echo "Waiting for llama-server..."
until curl -sf http://127.0.0.1:8080/health > /dev/null 2>&1; do sleep 2; done
echo "llama-server ready"

# Start ChromaDB in background
python3.11 -m chromadb.cli.cli run \
  --path ~/volumes/chromadb \
  --host 127.0.0.1 \
  --port 8001 &

CHROMA_PID=$!
sleep 3

# Start FastAPI backend
cd ~/nyapsys/backend
python3.11 -m uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 \
  --workers 1

# If FastAPI exits, kill everything so launchd restarts the whole service
kill $LLAMA_PID $CHROMA_PID 2>/dev/null || true
```

### Cloudflare Tunnel setup (`scripts/setup_tunnel.sh`)

Exposes the local Mac server to the internet with HTTPS — no port forwarding, no open router ports.

```bash
#!/bin/bash
# Authenticate with Cloudflare (opens browser)
cloudflared tunnel login

# Create tunnel
cloudflared tunnel create nyapsys

# Create tunnel config
cat > ~/.cloudflared/config.yml <<EOF
tunnel: nyapsys
credentials-file: ~/.cloudflared/$(cloudflared tunnel list --output json | jq -r '.[] | select(.name=="nyapsys") | .id').json

ingress:
  - hostname: api.yourdomain.com
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF

# Route DNS
cloudflared tunnel route dns nyapsys api.yourdomain.com

# Install as launchd service (auto-starts with Mac)
cloudflared service install
```

After this, `https://api.yourdomain.com` → Cloudflare Tunnel → Mac port 8000. SSL handled by Cloudflare. No open ports on your router.

---

## Database Design — 40/60 Split

### 40% — Dataset / Logs (SQLite via aiosqlite)

File: `~/volumes/sqlite/nyapsys.db`
Schema created automatically on startup in `backend/app/db.py`.
Use `aiosqlite` for all queries — never use synchronous `sqlite3` in async FastAPI handlers.

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
    chroma_ids       TEXT,            -- JSON array of ChromaDB document IDs
    created_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    message_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,
    score       REAL CHECK(score BETWEEN 0.0 AND 1.0),
    feedback    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
  ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_files_conversation
  ON files(conversation_id);
```

### 60% — Knowledge Base (ChromaDB)

- **Collection name:** `nyapsys_kb`
- **Embedding model:** `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, fast on CPU, ~90MB
- **Persistence:** `~/volumes/chromadb/`

Each chunk stored with:

```python
{
    "id": "{file_id}_chunk_{index}",
    "document": "chunk text content here",
    "metadata": {
        "file_id": "uuid",
        "filename": "report.pdf",
        "file_type": "pdf",
        "chunk_index": 3,
        "total_chunks": 12,
        "conversation_id": "uuid",
        "created_at": "2025-01-01T00:00:00"
    }
}
```

`backend/app/rag.py` must implement:
- `init_collection()` — called on startup
- `embed_and_upsert(chunks, metadatas, ids)` — batch embed + store
- `query(text, conversation_id, top_k=5)` — semantic search scoped to conversation
- `delete_by_file_id(file_id)` — cleanup when file is removed

---

## File + Image Ingestion Pipeline (`backend/app/ingest.py`)

**Supported types:** PDF, DOCX, TXT, MD, CSV, JSON, JPG, JPEG, PNG, WEBP, GIF

```python
async def ingest_file(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    conversation_id: str
) -> IngestResult:
    """
    Full pipeline: extract → chunk → embed → store → log.
    Returns IngestResult(file_id, chunk_count, filename).
    """
```

### Text extraction by file type

| Type | Library | Notes |
|---|---|---|
| PDF | `pymupdf` (fitz) | `fitz.open(stream=file_bytes)` — multi-page |
| DOCX | `python-docx` | Extract paragraphs + table cells |
| TXT / MD | built-in | `file_bytes.decode('utf-8', errors='replace')` |
| CSV | `pandas` | `pd.read_csv()` → convert to text rows |
| JSON | built-in | `json.loads()` → `json.dumps(indent=2)` → chunk as text |
| Images | **special path — see below** | Never extract text from images |

### Image handling

Images go through a completely different path. Do NOT attempt text extraction from images.

```python
async def ingest_image(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    conversation_id: str
) -> ImageIngestResult:
    img = Image.open(io.BytesIO(file_bytes))

    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Resize to max 1120x1120
    img.thumbnail((1120, 1120), Image.LANCZOS)

    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    file_id = str(uuid.uuid4())
    await db.insert_file(file_id, conversation_id, filename, file_type,
                          len(file_bytes), chunk_count=0, chroma_ids=[])

    return ImageIngestResult(file_id=file_id, image_b64=image_b64,
                              media_type="image/jpeg")
```

### Chunking strategy (text files only)

```python
CHUNK_SIZE = 512        # tokens
CHUNK_OVERLAP = 64      # tokens
MIN_CHUNK_TOKENS = 50   # discard fragments smaller than this

def chunk_text(text: str) -> list[str]:
    """
    Token-based chunking. Respects sentence boundaries.
    Never splits mid-sentence if it fits in one chunk.
    """
```

After chunking: embed all chunks in a **single batch call**, then upsert to ChromaDB
in a single batch. Never embed one chunk at a time.

---

## Model Client (`backend/app/model.py`)

The ONLY file that communicates with llama-server.
Nothing else imports from here directly except `agent.py`.

```python
LLAMA_HOST = os.getenv("LLAMA_HOST", "http://127.0.0.1:8080")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
N_PREDICT = int(os.getenv("N_PREDICT", "2048"))
```

### Text generation (streaming)

```python
async def generate(
    messages: list[dict],
    max_tokens: int = 2048,
    temperature: float = 0.7,
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    POST /v1/chat/completions to llama-server.
    Streams token strings as they arrive.
    messages: [{"role": "system"|"user"|"assistant", "content": "text"}]
    Yields individual token strings.
    """
```

### Vision generation (image + text, streaming)

```python
async def generate_with_image(
    text: str,
    image_b64: str,
    media_type: str = "image/jpeg",
    max_tokens: int = 2048,
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    Sends image + text as multimodal content array:
    [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{b64}"}},
      {"type": "text", "text": "{text}"}
    ]
    Streams tokens back. Same interface as generate().
    """
```

Always use `stream=True`. Use `httpx.AsyncClient` with 120-second timeout.
Handle `httpx.ReadTimeout` gracefully — yield an error message instead of crashing.

---

## Agent Loop (`backend/app/agent.py`)

```python
async def run(
    user_message: str,
    conversation_id: str,
    file_bytes: bytes | None = None,
    filename: str | None = None,
    file_type: str | None = None,
) -> AsyncGenerator[str, None]:
    """
    Core agent. Yields token strings for SSE streaming.
    """
```

```
STEP 1 — INGEST
  If file_bytes is not None:
    If image (jpg, jpeg, png, webp, gif):
      → ingest_image() → ImageIngestResult with image_b64
    Else:
      → ingest_file() → chunks stored in ChromaDB

STEP 2 — RETRIEVE
  Embed user_message with sentence-transformers
  Query ChromaDB nyapsys_kb:
    top_k = RAG_TOP_K (default 5)
    filter by conversation_id
    score_threshold = RAG_SCORE_THRESHOLD (default 0.4)
  If no results: context = ""
  Else: context = "\n\n---\n\n".join(chunk.document for chunk in results)

STEP 3 — PLAN
  Load last MAX_HISTORY_MESSAGES (default 10) from SQLite
  Build messages list:

  messages = [
    {
      "role": "system",
      "content": (
        "You are Nyapsys, a helpful AI assistant. "
        "You answer questions accurately, read and analyse files, "
        "and understand images. Be concise but thorough. "
        "If you are unsure, say so."
      )
    }
  ]

  If context:
    messages.append({
      "role": "system",
      "content": f"Relevant context from uploaded files:\n{context}"
    })

  Append historical messages (oldest first)

STEP 4 — GENERATE
  If image ingested:
    → model.generate_with_image(text=user_message, image_b64=...)
  Else:
    messages.append({"role": "user", "content": user_message})
    → model.generate(messages=messages)

  Collect tokens while yielding:
    full_response = ""
    async for token in model_stream:
      full_response += token
      yield token

STEP 5 — STORE
  await db.insert_message(conversation_id, "user", user_message, ...)
  await db.insert_message(conversation_id, "assistant", full_response)
  await db.update_conversation(conversation_id)

STEP 6 — DONE
  yield "[DONE]"
```

---

## Backend API (`backend/app/main.py`)

```
POST   /chat
  Content-Type: multipart/form-data
  Fields: message (str), conversation_id (str), file (optional)
  Response: text/event-stream (SSE) — token strings, ends with [DONE]
  Auth: Bearer {API_SECRET_KEY}

POST   /ingest
  Content-Type: multipart/form-data
  Fields: file (required), conversation_id (str)
  Response: { "file_id": "uuid", "chunk_count": 12, "filename": "..." }
  Auth: Bearer {API_SECRET_KEY}

GET    /conversations
  Response: [{ "id", "title", "message_count", "updated_at" }]
  Auth: Bearer {API_SECRET_KEY}

GET    /conversations/{id}/messages
  Response: [{ "id", "role", "content", "has_file", "has_image", "created_at" }]
  Auth: Bearer {API_SECRET_KEY}

DELETE /conversations/{id}
  Response: 204 — also deletes ChromaDB chunks for all files in conversation
  Auth: Bearer {API_SECRET_KEY}

POST   /feedback
  Body: { "message_id": "uuid", "score": 0.8, "feedback": "thumbs_up" }
  Response: 201
  Auth: Bearer {API_SECRET_KEY}

GET    /health
  Response: { "status": "ok", "model": "loaded", "uptime_seconds": 3600 }
  Auth: none
```

All error responses: `{"error": "human readable message"}` + appropriate HTTP status.
All endpoints async. Never use synchronous blocking calls in any handler.

---

## Frontend (`frontend/`)

Static Next.js app deployed to Firebase. Must be fully static —
no server components, no API routes, no `getServerSideProps`.

### `next.config.js` — exact required config

```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
    NEXT_PUBLIC_API_KEY: process.env.NEXT_PUBLIC_API_KEY,
  },
}
module.exports = nextConfig
```

### Chat UI requirements

- Full-height layout: fixed header + scrollable message thread + fixed input bar
- User messages right-aligned, assistant messages left-aligned
- Streaming: display tokens as they arrive via `fetch()` + `response.body.getReader()`
- Markdown rendering: `react-markdown` + `remark-gfm`
- Code blocks with syntax highlighting
- File/image upload: paperclip icon, shows preview before send
- Sidebar: past conversations, collapsible on mobile
- Mobile responsive: works at 375px viewport
- No login UI — API key injected at build time

### `frontend/src/lib/api.ts`

```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL
const API_KEY = process.env.NEXT_PUBLIC_API_KEY

export async function* streamChat(
  message: string,
  conversationId: string,
  file?: File
): AsyncGenerator<string> {
  const formData = new FormData()
  formData.append('message', message)
  formData.append('conversation_id', conversationId)
  if (file) formData.append('file', file)

  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${API_KEY}` },
    body: formData,
  })

  const reader = response.body!.getReader()
  const decoder = new TextDecoder()

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    const chunk = decoder.decode(value)
    if (chunk === '[DONE]') break
    yield chunk
  }
}
```

### `firebase.json`

```json
{
  "hosting": {
    "public": "out",
    "ignore": ["firebase.json", "**/.*", "**/node_modules/**"],
    "rewrites": [{ "source": "**", "destination": "/index.html" }],
    "headers": [{
      "source": "**",
      "headers": [{ "key": "Cache-Control", "value": "public, max-age=3600" }]
    }]
  }
}
```

---

## GitHub Actions — CI/CD Workflows

### GitHub Secrets

| Secret | How to get it |
|---|---|
| `FIREBASE_SERVICE_ACCOUNT` | Firebase console → Project Settings → Service accounts → Generate key |
| `MAC_SSH_KEY` | Private key for SSH access to Mac via Tailscale |
| `MAC_TAILSCALE_IP` | Tailscale admin console → Mac device IP (100.x.x.x) |
| `API_SECRET_KEY` | `openssl rand -hex 32` |
| `NEXT_PUBLIC_API_URL` | `https://api.yourdomain.com` |
| `NEXT_PUBLIC_API_KEY` | Same value as `API_SECRET_KEY` |

### GitHub Variables

| Variable | Value |
|---|---|
| `FIREBASE_PROJECT_ID` | Your Firebase project ID |

### `.github/workflows/deploy-firebase.yml`

```yaml
name: Deploy Frontend to Firebase

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
          cache: 'npm'
          cache-dependency-path: frontend/package-lock.json
      - name: Install dependencies
        working-directory: frontend
        run: npm ci
      - name: Build static export
        working-directory: frontend
        run: npm run build
        env:
          NEXT_PUBLIC_API_URL: ${{ secrets.NEXT_PUBLIC_API_URL }}
          NEXT_PUBLIC_API_KEY: ${{ secrets.NEXT_PUBLIC_API_KEY }}
      - name: Deploy to Firebase Hosting
        uses: FirebaseExtended/action-hosting-deploy@v0
        with:
          repoToken: ${{ secrets.GITHUB_TOKEN }}
          firebaseServiceAccount: ${{ secrets.FIREBASE_SERVICE_ACCOUNT }}
          channelId: live
          projectId: ${{ vars.FIREBASE_PROJECT_ID }}
          entryPoint: ./frontend
```

### `.github/workflows/deploy-backend.yml`

```yaml
name: Deploy Backend to Mac

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy via SSH (Tailscale)
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.MAC_TAILSCALE_IP }}
          username: YOUR_MAC_USERNAME
          key: ${{ secrets.MAC_SSH_KEY }}
          command_timeout: 5m
          script: |
            set -e
            cd ~/nyapsys
            git fetch origin main
            git reset --hard origin/main
            pip3.11 install -r backend/requirements.txt -q
            launchctl stop com.nyapsys.backend
            sleep 3
            launchctl start com.nyapsys.backend
            sleep 5
            curl -sf http://127.0.0.1:8000/health || exit 1
            echo "Deploy successful"
```

---

## Environment Variables (`.env.example`)

```env
# ── Model ──────────────────────────────────────────────────────────────
MODEL_PATH=~/volumes/models/Nyapsys-700M.Q4_K_M.gguf
LLAMA_HOST=http://127.0.0.1:8080
CTX_SIZE=4096
N_PREDICT=2048
TEMPERATURE=0.7

# ── Database ────────────────────────────────────────────────────────────
SQLITE_PATH=~/volumes/sqlite/nyapsys.db
CHROMA_HOST=http://127.0.0.1:8001
CHROMA_COLLECTION=nyapsys_kb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── RAG settings ────────────────────────────────────────────────────────
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.4
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MAX_HISTORY_MESSAGES=10

# ── API ─────────────────────────────────────────────────────────────────
API_SECRET_KEY=replace-with-output-of-openssl-rand-hex-32

# ── Google Cloud (model storage only) ───────────────────────────────────
GCS_BUCKET=nyapsys-models
GOOGLE_APPLICATION_CREDENTIALS=~/nyapsys/gcp-key.json

# ── IBM Cloud (training only — delete after training complete) ───────────
IBM_API_KEY=your-ibm-api-key
IBM_INSTANCE_NAME=nyapsys-training
IBM_REGION=us-south

# ── Firebase ─────────────────────────────────────────────────────────────
FIREBASE_PROJECT_ID=your-firebase-project-id

# ── Frontend (injected at build time via GitHub Actions) ──────────────────
NEXT_PUBLIC_API_URL=https://api.yourdomain.com
NEXT_PUBLIC_API_KEY=same-value-as-API_SECRET_KEY
```

---

## Build Order for OpenCode

Build in this exact sequence. Every phase produces something testable before moving on.
Do not skip phases. Do not build the frontend before the backend API is running end-to-end.

```
════════════════════════════════════════════════════════════
PHASE 1 — Training pipeline (Mac M3 dev + IBM L40S GPU)
════════════════════════════════════════════════════════════
 1.  training/requirements.txt
 2.  training/dataset_config.py          — mix ratios, preprocessing constants
 3.  training/tokenizer_train.py         — BPE tokenizer (skip if reusing 240M tokenizer)
 4.  training/model_config.py            — 240M and 700M config dicts
 5.  training/prepare_dataset.py         — download HF datasets → tokenized JSONL
 6.  training/train.py                   — from-scratch pretraining loop
 7.  training/instruction_tune.py        — LoRA instruction tuning (SFTTrainer)
 8.  training/merge_and_export.py        — merge adapters → GGUF Q4_K_M
 9.  training/upload_to_gcs.py           — push GGUF to GCP Cloud Storage
10.  training/README.md                  — human-readable step-by-step guide
11.  scripts/setup_ibm_gpu.sh            — IBM L40S instance automation

════════════════════════════════════════════════════════════
PHASE 2 — Mac setup and services
════════════════════════════════════════════════════════════
12.  scripts/setup_mac.sh                — brew deps, directories, launchd install
13.  scripts/download_model.sh           — pull GGUF from GCS to ~/volumes/models/
14.  scripts/setup_tunnel.sh             — Cloudflare Tunnel install + DNS config
15.  scripts/com.nyapsys.backend.plist   — launchd service definition
16.  backend/run.sh                      — starts llama-server + chromadb + uvicorn

════════════════════════════════════════════════════════════
PHASE 3 — Backend (runs on Mac, always-on)
════════════════════════════════════════════════════════════
17.  backend/requirements.txt
18.  backend/app/schemas.py              — ALL Pydantic models — write this first
19.  backend/app/db.py                   — SQLite schema + async CRUD
     TEST: python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
20.  backend/app/rag.py                  — ChromaDB init + embed + upsert + query
     TEST: start chromadb; python -c "from app.rag import init_collection; ..."
21.  backend/app/model.py                — llama.cpp text + vision client
     TEST: start llama-server; curl http://127.0.0.1:8080/v1/chat/completions
22.  backend/app/ingest.py               — file + image ingestion pipeline
     TEST: upload a PDF and a JPG, verify ChromaDB chunks + SQLite records created
23.  backend/app/agent.py                — full agent loop wiring phases 19–22
     TEST: call agent.run() with a text message, verify SSE token stream
24.  backend/app/health.py               — /health endpoint
25.  backend/app/main.py                 — FastAPI app, all routes mounted
     TEST: bash backend/run.sh; curl -X POST http://127.0.0.1:8000/chat with message

════════════════════════════════════════════════════════════
PHASE 4 — Frontend (Firebase Hosting)
════════════════════════════════════════════════════════════
26.  frontend/next.config.js             — output: 'export' — MUST be first
27.  frontend/firebase.json
28.  frontend/.firebaserc
29.  frontend/package.json
30.  frontend/src/lib/api.ts             — typed API client + streamChat generator
31.  frontend/src/hooks/useChat.ts       — chat state + streaming + conversation mgmt
32.  frontend/src/components/FilePreview.tsx
33.  frontend/src/components/MessageBubble.tsx
34.  frontend/src/components/MessageInput.tsx
35.  frontend/src/components/ChatWindow.tsx
36.  frontend/src/app/layout.tsx
37.  frontend/src/app/page.tsx
     TEST: npm run build → verify out/ directory; open in browser, send a message

════════════════════════════════════════════════════════════
PHASE 5 — CI/CD (GitHub Actions)
════════════════════════════════════════════════════════════
38.  .github/workflows/deploy-firebase.yml
39.  .github/workflows/deploy-backend.yml
     TEST: push to main → verify both workflows pass in GitHub Actions tab
     TEST: end-to-end — send message from Firebase URL, verify response streams back
```

---

## Non-Negotiable Constraints

These are hard rules. OpenCode must not deviate from them for any reason.

1. **Static frontend** — `output: 'export'` in `next.config.js`. No SSR. No Next.js API routes.
2. **All inference on Mac** — llama-server, backend, ChromaDB, SQLite all run locally. Nothing moves to cloud.
3. **No Docker** — everything runs as native Mac processes managed by launchd. No Docker Desktop.
4. **Model files never in git** — GGUF lives in `~/volumes/models/`, downloaded via script.
5. **`.env` never in git** — use GitHub Secrets for CI/CD, manual copy for local Mac.
6. **Always stream** — `/chat` endpoint streams SSE tokens. Never buffer the full response.
7. **launchd manages everything** — `KeepAlive: true` ensures auto-restart on crash. Never use `nohup` or `screen`.
8. **Cloudflare Tunnel for public access** — never open router ports, never use ngrok.
9. **aiosqlite everywhere** — never use synchronous `sqlite3` in async FastAPI handlers.
10. **Single ChromaDB collection** — `nyapsys_kb` only. Scope all queries by `conversation_id` metadata filter.
11. **Batch embedding** — never embed one chunk at a time. Always batch before upserting.
12. **Destroy IBM instance after training** — it bills per second even when idle.
13. **Tailscale for SSH** — GitHub Actions deploys to Mac via Tailscale IP, not public IP.
14. **700M is the target** — do not change the model size without updating this file first.