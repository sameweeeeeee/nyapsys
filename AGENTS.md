# AGENTS.md — Nyapsys AI Agent

> This is the single authoritative brief for OpenCode and any AI coding assistant working on this repo.
> Read this file in full before writing a single line of code. Every architectural decision is final
> unless explicitly marked as a choice. Build strictly in the order defined at the bottom of this file.

---

## What Nyapsys Does

Nyapsys is a self-hosted, always-on AI agent with three core capabilities:

1. **Answers questions** — conversational Q&A over a fine-tuned Llama 3.2 11B Vision model
2. **Reads files** — PDF, DOCX, TXT, Markdown, CSV, JSON uploaded by the user
3. **Understands images** — natively via the Vision model's multimodal architecture (no OCR fallback)

It is NOT a wrapper around a third-party API. The LLM runs entirely on self-hosted infrastructure.
The frontend is a static Next.js chat UI hosted on Firebase. All compute runs on DigitalOcean.

---

## Final Model Decision

**Model: Llama 3.2 11B Vision Instruct**
**Format: GGUF Q4_K_M**
**Size on disk: ~5.96 GB**
**Inference RAM required: ~6–7 GB (fits in 8 GB droplet with headroom)**
**Source: https://huggingface.co/pbatra/Llama-3.2-11B-Vision-Instruct-GGUF**

This model handles text, images, and documents in a single architecture — no separate vision
pipeline, no OCR, no secondary model. Fine-tuned by Meta on 6 billion image-text pairs.
Supports 128k token context length. Exposes an OpenAI-compatible API via llama.cpp server.

### Why this model
- 5.96 GB Q4_K_M fits on the 8 GB compute droplet alongside the full stack
- Native multimodal — send image bytes directly, model reasons over them natively
- Strong instruction following out of the box; LoRA fine-tuning sharpens Nyapsys behaviour
- Largest model that fits on chosen hardware without a droplet upgrade

### Model is NOT committed to git
The GGUF file lives on the DO Block Volume at `/volumes/models/`.
Use `scripts/download_model.sh` to pull it onto a fresh droplet.
Never hardcode model logic anywhere — all inference goes through `backend/app/model.py` only.

---

## Architecture Overview

```
[User browser / mobile]
        │
        ▼
Firebase Hosting  ←── GitHub Actions deploys Next.js static export here on every push
(free, global CDN, always on)
        │
        │  HTTPS API calls
        ▼
Cloudflare DNS + SSL  (free tier — SSL termination, DDoS protection, caching)
        │
        ▼
┌────────────────────────────────────────────────────────────────┐
│  DigitalOcean Kubernetes (DOKS)                                │
│  Control plane: managed by DO — free                          │
│                                                                │
│  Worker node: s-1vcpu-2gb  ($12/mo)                           │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Pod: Nginx Ingress Controller                           │  │
│  │  Pod: middleware (FastAPI)                               │  │
│  │    • JWT / API key auth on every request                 │  │
│  │    • Rate limiting — 60 req/min per IP                   │  │
│  │    • Async proxy → compute droplet                       │  │
│  │    • HPA: auto-scales 1→3 replicas at 70% CPU           │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬─────────────────────────────────┘
                               │ DO Private VPC (never public)
                               ▼
┌────────────────────────────────────────────────────────────────┐
│  DO Compute Droplet: s-2vcpu-8gb  ($36/mo)                    │
│  Docker Compose — restart: always on every service            │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Container: llama-server                                  │  │
│  │   llama.cpp HTTP server                                  │  │
│  │   Serves Nyapsys-11B-Vision.Q4_K_M.gguf                 │  │
│  │   OpenAI-compatible API on 127.0.0.1:8080               │  │
│  │   Loads mmproj-model-f16.gguf for vision support        │  │
│  │   CPU inference — 2 threads, 2 parallel slots           │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Container: backend (FastAPI)  port 8000                  │  │
│  │   Nyapsys agent loop                                     │  │
│  │   File + image ingestion pipeline                        │  │
│  │   ChromaDB client — knowledge base queries              │  │
│  │   SQLite client — conversation + file logging           │  │
│  │   SSE streaming endpoint for chat                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Container: chromadb  port 127.0.0.1:8001                │  │
│  │   Vector store for RAG knowledge base                    │  │
│  │   Persisted to block volume                             │  │
│  └─────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬─────────────────────────────────┘
                               │ Mounted block volume
                               ▼
                  DO Block Volume: 50 GB  ($5/mo)
                  /volumes/models/      ← GGUF + mmproj weights
                  /volumes/chromadb/    ← vector store persistence
                  /volumes/sqlite/      ← nyapsys.db
```

### Monthly Cost (always-on, 24/7)
| Resource | Cost | Notes |
|---|---|---|
| Firebase Hosting | $0 | Free tier |
| Cloudflare DNS + SSL | $0 | Free tier |
| DOKS control plane | $0 | DO managed |
| K8s worker s-1vcpu-2gb | $12/mo | Middleware only |
| Compute droplet s-2vcpu-8gb | $36/mo | All inference + RAG |
| Block volume 50 GB | $5/mo | Model + databases |
| DO Spaces (during training only) | ~$5 one-time | Delete after |
| **Total hosting** | **~$53/mo** | |
| **With DO education credits** | **~$0** | Credits cover all of this |
| **One-time training (GPU droplet)** | **~$15–23** | Also covered by credits |

---

## Repository Structure

```
nyapsys/
├── AGENTS.md                              ← this file
├── .env.example                           ← every env var documented here
├── .gitignore                             ← must include: .env, volumes/, *.gguf, *.pt, __pycache__
│
├── .github/
│   └── workflows/
│       ├── deploy-firebase.yml            ← frontend CI/CD
│       ├── deploy-k8s.yml                 ← middleware CI/CD
│       └── deploy-droplet.yml            ← backend CI/CD
│
├── scripts/
│   ├── download_model.sh                  ← pulls GGUF + mmproj from Spaces to /volumes/models/
│   ├── setup_droplet.sh                   ← Docker install, block volume mount, UFW rules
│   ├── setup_gpu_droplet.sh              ← prepares DO GPU droplet for fine-tuning session
│   └── export_gguf.sh                     ← wraps llama.cpp convert script for GGUF export
│
├── training/
│   ├── README.md                          ← step-by-step fine-tuning instructions for humans
│   ├── requirements.txt                   ← unsloth, trl, transformers, peft, datasets, boto3
│   ├── dataset_config.py                 ← dataset mix ratios, preprocessing rules, tokenizer config
│   ├── prepare_dataset.py                ← downloads HF datasets, formats to Llama 3.2 chat template JSONL
│   ├── train.py                           ← LoRA fine-tune: Unsloth + TRL SFTTrainer
│   ├── merge_and_export.py               ← merges LoRA adapters → full model → GGUF Q4_K_M
│   └── upload_to_spaces.py               ← uploads GGUF to DO Spaces
│
├── frontend/
│   ├── Dockerfile                         ← for local dev only; Firebase handles production
│   ├── package.json
│   ├── next.config.js                     ← MUST have output: 'export' and trailingSlash: true
│   ├── .firebaserc                        ← Firebase project binding
│   ├── firebase.json                      ← Firebase hosting config (public: "out")
│   └── src/
│       ├── app/
│       │   ├── layout.tsx                 ← root layout, fonts, metadata
│       │   ├── page.tsx                   ← main chat page
│       │   └── globals.css
│       ├── components/
│       │   ├── ChatWindow.tsx             ← scrollable message thread
│       │   ├── MessageBubble.tsx          ← single message with markdown rendering
│       │   ├── MessageInput.tsx           ← text input + file/image upload button
│       │   └── FilePreview.tsx            ← thumbnail/filename preview before send
│       ├── hooks/
│       │   └── useChat.ts                 ← all chat state, SSE streaming, conversation management
│       └── lib/
│           └── api.ts                     ← typed API client for middleware
│
├── middleware/
│   ├── Dockerfile
│   ├── requirements.txt                   ← fastapi, uvicorn, python-jose, httpx, slowapi
│   ├── k8s/
│   │   ├── namespace.yaml                 ← nyapsys namespace
│   │   ├── deployment.yaml               ← 1 replica, resource limits, liveness probe
│   │   ├── service.yaml                   ← ClusterIP, port 80 → 8000
│   │   ├── ingress.yaml                   ← Nginx Ingress, host api.nyapsys.yourdomain.com
│   │   ├── secret.yaml                    ← documented but applied manually — never commit values
│   │   └── hpa.yaml                       ← min 1, max 3, scale at 70% CPU
│   └── app/
│       ├── main.py                        ← FastAPI app: CORS, auth, limiter, router mounted
│       ├── auth.py                        ← Bearer token validation middleware
│       ├── router.py                      ← async httpx proxy to backend droplet, SSE passthrough
│       ├── limiter.py                     ← slowapi rate limiter: 60/min per IP
│       └── health.py                      ← GET /health — no auth required
│
├── backend/
│   ├── docker-compose.yml                 ← llama-server, backend, chromadb services
│   ├── Dockerfile                         ← python:3.11-slim, installs requirements
│   ├── requirements.txt                   ← fastapi, uvicorn, chromadb-client, pymupdf,
│   │                                         python-docx, pillow, httpx, aiosqlite,
│   │                                         sentence-transformers, pandas, python-multipart
│   └── app/
│       ├── main.py                        ← FastAPI entry point, mounts all routers
│       ├── schemas.py                     ← ALL Pydantic models — write this first
│       ├── agent.py                       ← core agent loop (see Agent Loop section)
│       ├── model.py                       ← llama.cpp HTTP client: text + vision inputs
│       ├── rag.py                         ← ChromaDB: init collection, embed, upsert, query
│       ├── ingest.py                      ← file/image processing pipeline
│       ├── db.py                          ← SQLite async CRUD via aiosqlite
│       └── health.py                      ← GET /health — returns model status + uptime
│
└── volumes/                               ← GITIGNORED — lives on DO block volume
    ├── models/                            ← Nyapsys-11B-Vision.Q4_K_M.gguf + mmproj here
    ├── chromadb/                          ← ChromaDB persistence directory
    └── sqlite/                            ← nyapsys.db lives here
```

---

## Fine-Tuning Pipeline (training/)

One-time process. Run on a DO GPU droplet. Destroy the GPU droplet the moment training ends.

### Dataset composition

`training/dataset_config.py` defines this exact mix:

| Dataset | HuggingFace ID | Purpose | Mix % |
|---|---|---|---|
| Alpaca Cleaned | `yahma/alpaca-cleaned` | General instruction following | 30% |
| SQuAD v2 | `rajpurkar/squad_v2` | Reading comprehension / document Q&A | 25% |
| LLaVA Instruct | `HuggingFaceH4/llava-instruct-mix-vsft` | Vision + image Q&A — critical for 11B Vision | 25% |
| LMSYS Chat 1M | `lmsys/lmsys-chat-1m` | Real multi-turn conversations | 15% |
| Custom JSONL | `training/data/custom.jsonl` | Nyapsys persona + domain-specific examples | 5% |

Total target: ~50,000 examples after balancing. Quality over quantity for a fine-tune.

### Chat template format — every training example must use this exactly

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are Nyapsys, a helpful AI assistant. You answer questions accurately, read and
analyse files, and understand images. Be concise but thorough. If you are unsure, say so.
<|eot_id|><|start_header_id|>user<|end_header_id|>
{user message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
{response}<|eot_id|>
```

IMPORTANT: Llama 3.2 Vision does NOT support system role in vision (image) turns.
For image examples from LLaVA dataset, inject the system instruction into the user turn instead:
```
<|begin_of_text|><|start_header_id|>user<|end_header_id|>
You are Nyapsys. {image content} {question}<|eot_id|>...
```

`training/prepare_dataset.py` handles all dataset downloading, mixing, formatting, and
validation. It outputs `training/data/prepared/train.jsonl` and `training/data/prepared/eval.jsonl`.
Run this on your Mac M3 — it needs no GPU.

### Step 1 — Prepare data on Mac M3

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py
# Output: training/data/prepared/train.jsonl (~45k examples)
#         training/data/prepared/eval.jsonl (~5k examples)

# Upload to DO Spaces for the GPU droplet to access
python upload_to_spaces.py --source data/prepared/ --bucket nyapsys-training --prefix dataset/
```

### Step 2 — Create DO GPU droplet (RTX 6000 Ada)

RTX 6000 Ada: 48 GB VRAM, ~$1.50/hr, pre-installed CUDA + PyTorch.
This is the best value option — 48 GB means the 11B model + all LoRA adapter states
fit without gradient checkpointing tricks.

```bash
doctl compute droplet create nyapsys-gpu-training \
  --size gpu-rtx6000ada-1 \
  --image gpu-h100x1-base \
  --region nyc2 \
  --ssh-keys YOUR_SSH_KEY_ID \
  --wait

# Get IP
doctl compute droplet get nyapsys-gpu-training --format PublicIPv4

# SSH in
ssh root@GPU_DROPLET_IP
```

### Step 3 — Setup GPU droplet (`scripts/setup_gpu_droplet.sh`)

```bash
#!/bin/bash
set -e

# Install training dependencies
pip install unsloth trl transformers peft datasets boto3 bitsandbytes accelerate

# Install llama.cpp for GGUF export
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && make -j$(nproc)
pip install -r requirements.txt

# Pull dataset from DO Spaces
aws s3 sync s3://nyapsys-training/dataset/ ./data/prepared/ \
  --endpoint-url https://nyc3.digitaloceanspaces.com

# Pull training code from GitHub
git clone https://github.com/YOUR_USERNAME/nyapsys /opt/nyapsys

# Verify GPU
nvidia-smi
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo "GPU droplet ready for training"
```

### Step 4 — Run LoRA fine-tune (`training/train.py`)

```bash
cd /opt/nyapsys
python training/train.py
```

`training/train.py` implementation requirements:

```python
# Key configuration — hardcode these values, do not make them CLI args
MODEL_NAME = "meta-llama/Llama-3.2-11B-Vision-Instruct"
OUTPUT_DIR = "./output/nyapsys-lora"
DATASET_PATH = "./data/prepared/"
MAX_SEQ_LENGTH = 2048
NUM_EPOCHS = 3
BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4   # effective batch = 16
LEARNING_RATE = 2e-4
WARMUP_RATIO = 0.05
LR_SCHEDULER = "cosine"
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
```

Use `unsloth.FastVisionModel` for loading — it applies 2x speed + 60% VRAM reduction automatically.
Use `trl.SFTTrainer` — do NOT write a custom training loop.
Use `peft.LoraConfig` for adapter config.
Log loss every 10 steps to stdout.
Save checkpoints every 200 steps to `./output/checkpoints/` — insurance against crashes.
After training: save final LoRA adapter to `./output/nyapsys-lora/`.

Expected training time on RTX 6000 Ada: 10–15 hours unattended.
Expected final eval loss: < 1.2 (if higher, dataset may need cleaning).

### Step 5 — Merge and export GGUF (`training/merge_and_export.py`)

```bash
python training/merge_and_export.py
```

Implementation steps inside `merge_and_export.py`:
1. Load base model with `unsloth.FastVisionModel.from_pretrained(MODEL_NAME)`
2. Load LoRA adapter with `PeftModel.from_pretrained(base_model, OUTPUT_DIR)`
3. Merge: `model = model.merge_and_unload()`
4. Save merged HuggingFace model to `./output/nyapsys-merged/`
5. Run llama.cpp conversion:
   ```bash
   python /opt/llama.cpp/convert_hf_to_gguf.py ./output/nyapsys-merged/ \
     --outfile ./output/Nyapsys-11B-Vision.Q4_K_M.gguf \
     --outtype q4_k_m
   ```
6. Verify file size is ~5.96 GB before uploading
7. Upload to DO Spaces:
   ```bash
   python training/upload_to_spaces.py \
     --file ./output/Nyapsys-11B-Vision.Q4_K_M.gguf \
     --bucket nyapsys-models
   ```

### Step 6 — Destroy GPU droplet immediately

```bash
# On your Mac M3 — do this the moment the GGUF upload completes
doctl compute droplet delete nyapsys-gpu-training --force
echo "GPU droplet destroyed. Billing stopped."
```

Total GPU cost: ~$15–23 depending on training time. Covered by DO education credits.

---

## Backend Services — docker-compose.yml

Full `docker-compose.yml` for the compute droplet.
Every service has `restart: always`. Internal network `nyapsys-net` for service discovery.
Only backend port 8000 is exposed to the host, and UFW restricts it to the K8s worker VPC IP.

```yaml
version: "3.9"

networks:
  nyapsys-net:
    driver: bridge

services:

  llama-server:
    image: ghcr.io/ggerganov/llama.cpp:server
    restart: always
    networks: [nyapsys-net]
    ports:
      - "127.0.0.1:8080:8080"
    volumes:
      - /volumes/models:/models:ro
    command: >
      --model /models/Nyapsys-11B-Vision.Q4_K_M.gguf
      --mmproj /models/mmproj-model-f16.gguf
      --host 0.0.0.0
      --port 8080
      --ctx-size 8192
      --n-predict 2048
      --threads 2
      --parallel 2
      --cont-batching
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 5
      start_period: 60s

  chromadb:
    image: chromadb/chroma:latest
    restart: always
    networks: [nyapsys-net]
    ports:
      - "127.0.0.1:8001:8000"
    volumes:
      - /volumes/chromadb:/chroma/chroma
    environment:
      - IS_PERSISTENT=TRUE
      - ANONYMIZED_TELEMETRY=FALSE
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
      interval: 30s
      timeout: 5s
      retries: 3

  backend:
    build: .
    restart: always
    networks: [nyapsys-net]
    ports:
      - "8000:8000"
    volumes:
      - /volumes/sqlite:/volumes/sqlite
      - /volumes/chromadb:/volumes/chromadb
    env_file: .env
    environment:
      - LLAMA_HOST=http://llama-server:8080
      - CHROMA_HOST=http://chromadb:8000
    depends_on:
      llama-server:
        condition: service_healthy
      chromadb:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
```

Note on `--mmproj`: the multimodal projector file is required for the Vision model to process
images. Without it, llama-server will load but image inputs will fail. Always download both
files in `scripts/download_model.sh`.

Note on `--parallel 2`: allows 2 concurrent inference requests. Suitable for the 8 GB droplet.
Do not increase without testing — each parallel slot uses additional RAM for KV cache.

---

## Database Design — 40/60 Split

### 40% — Dataset / Logs (SQLite via aiosqlite)

File: `/volumes/sqlite/nyapsys.db`
Schema is created automatically on startup in `backend/app/db.py`.
Use `aiosqlite` for all queries — never use synchronous sqlite3 in async FastAPI handlers.

```sql
-- Conversation sessions
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    message_count INTEGER DEFAULT 0
);

-- Individual messages (user and assistant turns)
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

-- Files ingested into the knowledge base
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

-- User feedback / evals for quality monitoring
CREATE TABLE IF NOT EXISTS evals (
    id          TEXT PRIMARY KEY,
    message_id  TEXT REFERENCES messages(id) ON DELETE CASCADE,
    score       REAL CHECK(score BETWEEN 0.0 AND 1.0),
    feedback    TEXT,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_files_conversation ON files(conversation_id);
```

### 60% — Knowledge Base (ChromaDB)

Collection name: `nyapsys_kb`
Embedding model: `sentence-transformers/all-MiniLM-L6-v2`
  — 384-dimensional embeddings, runs fast on CPU, ~90 MB model size
Persistence: `/volumes/chromadb/` — survives container restarts

Each chunk stored with this metadata structure:
```python
{
    "id": "{file_id}_chunk_{index}",
    "document": "chunk text content here",
    "metadata": {
        "file_id": "uuid-string",
        "filename": "report.pdf",
        "file_type": "pdf",
        "chunk_index": 3,
        "total_chunks": 12,
        "conversation_id": "uuid-string",
        "created_at": "2025-01-01T00:00:00"
    }
}
```

`backend/app/rag.py` must implement:
- `init_collection()` — called on startup, creates collection if it doesn't exist
- `embed_and_upsert(chunks, metadatas, ids)` — batch embed + store
- `query(text, conversation_id, top_k=5)` — semantic search scoped to conversation
- `delete_by_file_id(file_id)` — cleanup when file is removed

---

## File + Image Ingestion Pipeline (backend/app/ingest.py)

Supported types: PDF, DOCX, TXT, MD, CSV, JSON, JPG, JPEG, PNG, WEBP, GIF

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
| PDF | `pymupdf` (fitz) | `fitz.open(stream=file_bytes)` — preserves layout, handles multi-page |
| DOCX | `python-docx` | Extract paragraphs + table cells as text |
| TXT / MD | built-in | `file_bytes.decode('utf-8', errors='replace')` |
| CSV | `pandas` | `pd.read_csv()` → convert to readable text rows |
| JSON | built-in | `json.loads()` → `json.dumps(indent=2)` → chunk as text |
| Images | **special handling — see below** | Never extract text from images |

### Image handling — Vision model, NOT OCR

Images go through a completely different path. Do NOT attempt text extraction.

```python
async def ingest_image(
    file_bytes: bytes,
    filename: str,
    file_type: str,
    conversation_id: str
) -> ImageIngestResult:
    # 1. Load with Pillow
    img = Image.open(io.BytesIO(file_bytes))

    # 2. Convert to RGB if needed (handles RGBA, P mode etc)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 3. Resize to max 1120x1120 — Vision model tile limit
    img.thumbnail((1120, 1120), Image.LANCZOS)

    # 4. Base64 encode for storage and model input
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    image_b64 = base64.b64encode(buffer.getvalue()).decode()

    # 5. Store reference in SQLite files table (no ChromaDB for images)
    file_id = str(uuid.uuid4())
    await db.insert_file(file_id, conversation_id, filename, file_type,
                          len(file_bytes), chunk_count=0, chroma_ids=[])

    return ImageIngestResult(file_id=file_id, image_b64=image_b64, media_type="image/jpeg")
```

The `image_b64` is passed directly to `model.generate_with_image()` in the agent loop.
The Vision model processes the actual pixels — it can read text in images, describe content,
analyse charts, read handwriting, and understand document layouts natively.

### Chunking strategy (text files only)

```python
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-11B-Vision-Instruct")
CHUNK_SIZE = 512        # tokens
CHUNK_OVERLAP = 64      # tokens
MIN_CHUNK_TOKENS = 50   # discard fragments smaller than this

def chunk_text(text: str) -> list[str]:
    """
    Splits text into overlapping token-based chunks.
    Respects sentence boundaries — never splits mid-sentence.
    """
```

Use `TOKENIZER.encode()` for accurate token counting.
Split on sentence boundaries first (`. `, `\n\n`), then merge into chunks of CHUNK_SIZE.
Never split a sentence across two chunks if it fits in one.

After chunking: embed all chunks in a single batch call to sentence-transformers,
then upsert to ChromaDB in a single batch. Never embed one chunk at a time.

---

## Model Client (backend/app/model.py)

The ONLY file that communicates with llama-server. Nothing else imports from here directly
except `agent.py`. All model config comes from environment variables.

```python
LLAMA_HOST = os.getenv("LLAMA_HOST", "http://llama-server:8080")
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
    Calls POST /v1/chat/completions on llama-server.
    Streams token strings as they arrive via SSE.
    messages: [{"role": "system"|"user"|"assistant", "content": "text"}]
    Yields individual token strings. Caller assembles full response.
    """
```

### Vision generation (streaming, image + text)

```python
async def generate_with_image(
    text: str,
    image_b64: str,
    media_type: str = "image/jpeg",
    max_tokens: int = 2048,
    stream: bool = True
) -> AsyncGenerator[str, None]:
    """
    Sends image + text to llama-server vision endpoint.
    Constructs the multimodal content array:
    [
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,{b64}"}},
      {"type": "text", "text": "{text}"}
    ]
    Streams token strings back. Same interface as generate().
    """
```

Always use `stream=True`. The frontend displays tokens as they arrive.
Never buffer the entire response — users expect to see output within 1–2 seconds of sending.

Use `httpx.AsyncClient` with a 120-second timeout (large images + long responses take time).
Handle `httpx.ReadTimeout` gracefully — yield an error message instead of crashing.

---

## Agent Loop (backend/app/agent.py)

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
    Handles both text-only and multimodal (image/file) inputs.
    """
```

### Execution steps

```
STEP 1 — INGEST
  If file_bytes is not None:
    Detect if file is an image (jpg, jpeg, png, webp, gif):
      → Call ingest_image() → get ImageIngestResult with image_b64
      → Store image_ingest_result for use in STEP 4
    Else (PDF, DOCX, TXT, etc):
      → Call ingest_file() → chunks stored in ChromaDB
      → No image_b64 needed

STEP 2 — RETRIEVE
  Embed user_message using sentence-transformers (same model as ingest)
  Query ChromaDB nyapsys_kb:
    - top_k = RAG_TOP_K (default 5)
    - filter by conversation_id
    - score_threshold = RAG_SCORE_THRESHOLD (default 0.4)
  If no results above threshold: context = "" (proceed without context)
  Else: context = "\n\n---\n\n".join(chunk.document for chunk in results)

STEP 3 — PLAN (build prompt)
  Load last MAX_HISTORY_MESSAGES (default 10) from SQLite for this conversation_id
  Construct messages list:

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

  If context is not empty:
    messages.append({
      "role": "system",
      "content": f"Relevant context from uploaded files:\n{context}"
    })

  For each historical message (oldest first):
    messages.append({"role": msg.role, "content": msg.content})

  (Do not append current user message here — added in STEP 4)

STEP 4 — GENERATE
  If image was ingested in STEP 1:
    Call model.generate_with_image(
      text=user_message,
      image_b64=image_ingest_result.image_b64,
      media_type=image_ingest_result.media_type
    )
    Note: for vision turns, system message is folded into user turn
  Else:
    messages.append({"role": "user", "content": user_message})
    Call model.generate(messages=messages)

  Collect tokens in a buffer while yielding them:
    full_response = ""
    async for token in model_stream:
      full_response += token
      yield token   ← SSE to client

STEP 5 — STORE
  After stream ends:
  await db.insert_message(conversation_id, "user", user_message,
                           has_file=(file_bytes is not None),
                           has_image=(image was ingested))
  await db.insert_message(conversation_id, "assistant", full_response)
  await db.update_conversation(conversation_id)

STEP 6 — DONE
  yield "[DONE]"   ← SSE termination signal for client
```

---

## Backend API (backend/app/main.py)

```
POST   /chat
  Content-Type: multipart/form-data
  Fields:
    message         str  required — user's text message
    conversation_id str  required — UUID, generated client-side
    file            file optional — any supported file type
  Response: text/event-stream (SSE)
    Data events: token strings
    Final event: [DONE]
  Auth: Bearer {API_SECRET_KEY}

POST   /ingest
  Content-Type: multipart/form-data
  Fields:
    file            file required
    conversation_id str  required
  Response: application/json
    { "file_id": "uuid", "chunk_count": 12, "filename": "report.pdf" }
  Auth: Bearer {API_SECRET_KEY}

GET    /conversations
  Response: [{ "id", "title", "message_count", "updated_at" }]
  Auth: Bearer {API_SECRET_KEY}

GET    /conversations/{id}/messages
  Response: [{ "id", "role", "content", "has_file", "has_image", "created_at" }]
  Auth: Bearer {API_SECRET_KEY}

DELETE /conversations/{id}
  Response: 204 No Content
  Side effect: deletes ChromaDB chunks for all files in this conversation
  Auth: Bearer {API_SECRET_KEY}

POST   /feedback
  Body: { "message_id": "uuid", "score": 0.8, "feedback": "thumbs_up" }
  Response: 201
  Auth: Bearer {API_SECRET_KEY}

GET    /health
  Response: { "status": "ok", "model": "loaded", "uptime_seconds": 3600 }
  Auth: none — used by Docker health checks and K8s liveness probes
```

All error responses: `{"error": "human readable message"}` + appropriate HTTP status.
Use FastAPI's `HTTPException` for all error cases.
All endpoints are async. Never use synchronous blocking calls in any handler.

---

## Middleware (middleware/app/)

Thin FastAPI proxy. Implements no business logic.

### auth.py
```python
API_SECRET_KEY = os.getenv("API_SECRET_KEY")

async def verify_token(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = authorization.removeprefix("Bearer ")
    if token != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")
```

Apply as a FastAPI dependency on all routes except `/health`.

### router.py
```python
BACKEND_URL = os.getenv("BACKEND_URL")  # http://DROPLET_PRIVATE_IP:8000

# Catch-all proxy
@router.api_route("/{path:path}", methods=["GET", "POST", "DELETE", "PATCH"])
async def proxy(request: Request, path: str):
    # For SSE streaming responses (chat endpoint):
    #   Use httpx streaming client, yield chunks back without buffering
    # For regular responses:
    #   Forward complete response
    # Always forward: headers, body, query params
    # Remove hop-by-hop headers (connection, transfer-encoding)
```

### K8s manifests

`deployment.yaml` resource limits: 256Mi memory, 250m CPU request; 512Mi memory, 500m CPU limit.
Liveness probe: GET /health every 30s.
Readiness probe: GET /health every 10s, failure threshold 3.

`ingress.yaml` must set annotation `nginx.ingress.kubernetes.io/proxy-read-timeout: "300"`
for long SSE streaming connections to not be terminated by the ingress timeout.

`hpa.yaml`: min replicas 1, max 3, target CPU utilisation 70%.

---

## Frontend (frontend/)

Static Next.js app deployed to Firebase. Must be fully static — no server components,
no API routes, no getServerSideProps.

### next.config.js — exact required config
```js
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL,
  },
}
module.exports = nextConfig
```

### Chat UI requirements
- Full-height chat layout: fixed header + scrollable message thread + fixed input bar
- Message thread: user messages right-aligned, assistant messages left-aligned
- Streaming: display tokens as they arrive — do not wait for [DONE] to show content
  - Use `fetch()` with `response.body.getReader()` to consume SSE stream
  - Append tokens to the current assistant message bubble in real time
- Markdown rendering: assistant responses rendered with `react-markdown` + `remark-gfm`
  - Code blocks with syntax highlighting (`rehype-highlight` or `react-syntax-highlighter`)
- File/image upload: paperclip icon button next to input
  - Clicking opens file picker (accepts all supported types)
  - Shows preview: image thumbnail (max 100x100) or filename badge
  - File is sent with the next message via multipart/form-data
- Conversation history: sidebar (collapsible on mobile) listing past conversations
  - Clicking a conversation loads its messages from GET /conversations/{id}/messages
- New conversation: button that generates a new UUID and clears the message thread
- Mobile responsive: works correctly on 375px viewport
- No login UI — API key is injected at build time via NEXT_PUBLIC_API_URL env

### api.ts — typed API client
```typescript
const API_URL = process.env.NEXT_PUBLIC_API_URL
const API_KEY = process.env.NEXT_PUBLIC_API_KEY  // add this to .env.example + next.config.js

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

### firebase.json
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

### GitHub Secrets (configure in repo Settings → Secrets)
| Secret name | How to get it |
|---|---|
| `FIREBASE_SERVICE_ACCOUNT` | Firebase console → Project Settings → Service accounts → Generate key |
| `DO_ACCESS_TOKEN` | DO dashboard → API → Personal access tokens |
| `DROPLET_SSH_KEY` | Private key of SSH key added to your DO account |
| `DROPLET_IP` | Public IP of compute droplet |
| `KUBECONFIG_DATA` | `doctl k8s cluster kubeconfig show nyapsys-cluster \| base64 -w0` |
| `API_SECRET_KEY` | Random string — `openssl rand -hex 32` |
| `NEXT_PUBLIC_API_URL` | e.g. `https://api.nyapsys.yourdomain.com` |
| `NEXT_PUBLIC_API_KEY` | Same value as API_SECRET_KEY |

### GitHub Variables (repo Settings → Variables)
| Variable | Value |
|---|---|
| `FIREBASE_PROJECT_ID` | Your Firebase project ID |

### .github/workflows/deploy-firebase.yml

```yaml
name: Deploy Frontend to Firebase

on:
  push:
    branches: [main]
    paths:
      - 'frontend/**'
      - '.github/workflows/deploy-firebase.yml'

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

### .github/workflows/deploy-k8s.yml

```yaml
name: Deploy Middleware to Kubernetes

on:
  push:
    branches: [main]
    paths:
      - 'middleware/**'
      - '.github/workflows/deploy-k8s.yml'

jobs:
  build-and-deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Login to DO Container Registry
        run: |
          echo "${{ secrets.DO_ACCESS_TOKEN }}" | \
            docker login registry.digitalocean.com -u token --password-stdin

      - name: Build and push image
        run: |
          IMAGE=registry.digitalocean.com/nyapsys/middleware:${{ github.sha }}
          docker build -t $IMAGE ./middleware
          docker push $IMAGE
          echo "IMAGE=$IMAGE" >> $GITHUB_ENV

      - name: Configure kubectl
        run: |
          mkdir -p ~/.kube
          echo "${{ secrets.KUBECONFIG_DATA }}" | base64 -d > ~/.kube/config

      - name: Update deployment image
        run: |
          kubectl set image deployment/middleware \
            middleware=${{ env.IMAGE }} \
            -n nyapsys

      - name: Apply any manifest changes
        run: kubectl apply -f middleware/k8s/ -n nyapsys

      - name: Wait for rollout
        run: |
          kubectl rollout status deployment/middleware \
            -n nyapsys --timeout=120s

      - name: Smoke test
        run: |
          sleep 10
          curl -f ${{ secrets.NEXT_PUBLIC_API_URL }}/health
```

### .github/workflows/deploy-droplet.yml

```yaml
name: Deploy Backend to Droplet

on:
  push:
    branches: [main]
    paths:
      - 'backend/**'
      - '.github/workflows/deploy-droplet.yml'

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - name: Deploy via SSH
        uses: appleboy/ssh-action@v1
        with:
          host: ${{ secrets.DROPLET_IP }}
          username: root
          key: ${{ secrets.DROPLET_SSH_KEY }}
          command_timeout: 10m
          script: |
            set -e
            cd /opt/nyapsys

            # Tag current image as rollback point
            docker tag nyapsys-backend:latest nyapsys-backend:stable 2>/dev/null || true

            # Pull latest code
            git fetch origin main
            git reset --hard origin/main

            # Rebuild backend container only
            # llama-server and chromadb keep running — no inference downtime
            docker compose build backend

            # Rolling restart of backend only
            docker compose up -d --no-deps --force-recreate backend

            # Health check with retry
            for i in {1..6}; do
              sleep 5
              if curl -sf http://localhost:8000/health > /dev/null; then
                echo "Health check passed on attempt $i"
                exit 0
              fi
              echo "Attempt $i failed, retrying..."
            done

            # All retries failed — rollback
            echo "Deploy failed — rolling back to stable"
            docker tag nyapsys-backend:stable nyapsys-backend:latest
            docker compose up -d --no-deps --force-recreate backend
            exit 1
```

---

## Environment Variables (.env.example)

```env
# ── Model ─────────────────────────────────────────────────────────────
MODEL_PATH=/volumes/models/Nyapsys-11B-Vision.Q4_K_M.gguf
MMPROJ_PATH=/volumes/models/mmproj-model-f16.gguf
LLAMA_HOST=http://llama-server:8080
CTX_SIZE=8192
N_PREDICT=2048
TEMPERATURE=0.7

# ── Database ───────────────────────────────────────────────────────────
SQLITE_PATH=/volumes/sqlite/nyapsys.db
CHROMA_HOST=http://chromadb:8000
CHROMA_COLLECTION=nyapsys_kb
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ── RAG settings ───────────────────────────────────────────────────────
RAG_TOP_K=5
RAG_SCORE_THRESHOLD=0.4
CHUNK_SIZE=512
CHUNK_OVERLAP=64
MAX_HISTORY_MESSAGES=10

# ── API ────────────────────────────────────────────────────────────────
API_SECRET_KEY=replace-with-output-of-openssl-rand-hex-32
BACKEND_URL=http://DROPLET_PRIVATE_VPC_IP:8000

# ── DigitalOcean ───────────────────────────────────────────────────────
DO_CLUSTER_NAME=nyapsys-cluster
DO_REGION=sgp1
DO_DROPLET_IP=your.droplet.public.ip
DO_SPACES_BUCKET=nyapsys-models
DO_SPACES_ENDPOINT=https://sgp1.digitaloceanspaces.com
DO_SPACES_KEY=your-spaces-access-key
DO_SPACES_SECRET=your-spaces-secret-key

# ── Firebase ───────────────────────────────────────────────────────────
FIREBASE_PROJECT_ID=your-firebase-project-id

# ── Frontend (injected at build time via GitHub Actions) ───────────────
NEXT_PUBLIC_API_URL=https://api.nyapsys.yourdomain.com
NEXT_PUBLIC_API_KEY=same-value-as-API_SECRET_KEY
```

---

## Infrastructure Setup Scripts

### scripts/setup_droplet.sh — run once on a fresh compute droplet

```bash
#!/bin/bash
set -e
echo "=== Nyapsys compute droplet setup ==="

# 1. Install Docker
curl -fsSL https://get.docker.com | sh
systemctl enable docker && systemctl start docker
apt-get install -y docker-compose-plugin git curl

# 2. Format and mount block volume
# Check actual device name first: lsblk
# Only format if this is first setup — comment out mkfs line after first run
VOLUME_DEVICE=/dev/sda
VOLUME_MOUNT=/volumes

if ! blkid $VOLUME_DEVICE > /dev/null 2>&1; then
  echo "Formatting block volume..."
  mkfs.ext4 $VOLUME_DEVICE
fi

mkdir -p $VOLUME_MOUNT
mount $VOLUME_DEVICE $VOLUME_MOUNT
grep -q "$VOLUME_DEVICE" /etc/fstab || \
  echo "$VOLUME_DEVICE $VOLUME_MOUNT ext4 defaults,nofail 0 2" >> /etc/fstab

# 3. Create volume directories
mkdir -p $VOLUME_MOUNT/models $VOLUME_MOUNT/chromadb $VOLUME_MOUNT/sqlite

# 4. Clone repo
git clone https://github.com/YOUR_USERNAME/nyapsys /opt/nyapsys
cd /opt/nyapsys

# 5. UFW firewall — restrict backend to K8s worker VPC IP only
ufw allow OpenSSH
ufw allow from K8S_WORKER_PRIVATE_VPC_IP to any port 8000
ufw --force enable

echo "=== Setup complete ==="
echo "Next: scp .env root@DROPLET_IP:/opt/nyapsys/.env"
echo "Then: bash scripts/download_model.sh"
echo "Then: docker compose up -d"
```

### scripts/download_model.sh — downloads GGUF and mmproj to block volume

```bash
#!/bin/bash
set -e
source /opt/nyapsys/.env

MODEL_DIR=/volumes/models

echo "=== Downloading model files ==="

# Main GGUF — from DO Spaces (uploaded after fine-tuning)
if [ ! -f "$MODEL_DIR/Nyapsys-11B-Vision.Q4_K_M.gguf" ]; then
  echo "Downloading Nyapsys GGUF from DO Spaces..."
  AWS_ACCESS_KEY_ID=$DO_SPACES_KEY \
  AWS_SECRET_ACCESS_KEY=$DO_SPACES_SECRET \
  aws s3 cp s3://$DO_SPACES_BUCKET/Nyapsys-11B-Vision.Q4_K_M.gguf \
    $MODEL_DIR/ \
    --endpoint-url $DO_SPACES_ENDPOINT
else
  echo "Main GGUF already present — skipping"
fi

# Multimodal projector — required for Vision capability
# Download from HuggingFace if not already present
if [ ! -f "$MODEL_DIR/mmproj-model-f16.gguf" ]; then
  echo "Downloading multimodal projector weights..."
  wget -q --show-progress \
    -O $MODEL_DIR/mmproj-model-f16.gguf \
    "https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf"
else
  echo "mmproj already present — skipping"
fi

echo "=== Model files ready ==="
ls -lh $MODEL_DIR/
```

---

## Build Order for OpenCode

Build in this exact sequence. Every phase produces something you can test before moving on.
Do not skip phases. Do not build the frontend before the backend API is running end-to-end.

```
════════════════════════════════════════════════════════════
PHASE 1 — Training pipeline (Mac M3 + DO GPU droplet)
════════════════════════════════════════════════════════════
 1.  training/requirements.txt
 2.  training/dataset_config.py          — mix ratios, preprocessing constants
 3.  training/prepare_dataset.py         — download HF datasets → JSONL
 4.  training/train.py                   — Unsloth + TRL LoRA fine-tune
 5.  training/merge_and_export.py        — merge adapters → GGUF Q4_K_M
 6.  training/upload_to_spaces.py        — push GGUF to DO Spaces
 7.  training/README.md                  — human-readable step-by-step
 8.  scripts/setup_gpu_droplet.sh        — GPU droplet automation

════════════════════════════════════════════════════════════
PHASE 2 — Backend (compute droplet, Docker Compose)
════════════════════════════════════════════════════════════
 9.  backend/docker-compose.yml          — all three services defined
10.  backend/Dockerfile                  — python:3.11-slim image
11.  backend/requirements.txt
12.  backend/app/schemas.py              — ALL Pydantic models (write first)
13.  backend/app/db.py                   — SQLite schema creation + async CRUD
     TEST: python -c "import asyncio; from app.db import init_db; asyncio.run(init_db())"
14.  backend/app/rag.py                  — ChromaDB init + embed + upsert + query
     TEST: docker compose up chromadb; python -c "from app.rag import init_collection..."
15.  backend/app/model.py                — llama.cpp text + vision client
     TEST: docker compose up llama-server; curl http://localhost:8080/v1/chat/completions
16.  backend/app/ingest.py               — file + image ingestion pipeline
     TEST: upload a PDF and a JPG, verify ChromaDB chunks + SQLite records
17.  backend/app/agent.py                — full agent loop wiring steps 13–16
     TEST: call agent.run() with a text message, verify SSE token stream
18.  backend/app/health.py               — /health endpoint
19.  backend/app/main.py                 — FastAPI app: all routes mounted
     TEST: docker compose up; curl -X POST http://localhost:8000/chat with message
20.  scripts/setup_droplet.sh            — droplet setup automation
21.  scripts/download_model.sh           — model download automation

════════════════════════════════════════════════════════════
PHASE 3 — Middleware (K8s worker node)
════════════════════════════════════════════════════════════
22.  middleware/requirements.txt
23.  middleware/app/health.py
24.  middleware/app/auth.py              — Bearer token validation
25.  middleware/app/limiter.py           — slowapi rate limiting
26.  middleware/app/router.py            — async httpx proxy + SSE passthrough
27.  middleware/app/main.py              — FastAPI app assembly
28.  middleware/Dockerfile
29.  middleware/k8s/namespace.yaml
30.  middleware/k8s/secret.yaml          — document values, note: apply manually
31.  middleware/k8s/deployment.yaml
32.  middleware/k8s/service.yaml
33.  middleware/k8s/ingress.yaml         — include proxy-read-timeout: "300" annotation
34.  middleware/k8s/hpa.yaml
     TEST: kubectl apply -f middleware/k8s/; curl https://api.yourdomain.com/health

════════════════════════════════════════════════════════════
PHASE 4 — Frontend (Firebase Hosting)
════════════════════════════════════════════════════════════
35.  frontend/next.config.js             — output: 'export' — MUST be first
36.  frontend/firebase.json
37.  frontend/.firebaserc
38.  frontend/package.json               — next, react, react-markdown, remark-gfm
39.  frontend/src/lib/api.ts             — typed API client + streamChat generator
40.  frontend/src/hooks/useChat.ts       — all chat state + streaming + conversation mgmt
41.  frontend/src/components/FilePreview.tsx
42.  frontend/src/components/MessageBubble.tsx
43.  frontend/src/components/MessageInput.tsx
44.  frontend/src/components/ChatWindow.tsx
45.  frontend/src/app/layout.tsx
46.  frontend/src/app/page.tsx           — assembles all components
     TEST: npm run build → verify out/ directory; npm run start → test in browser

════════════════════════════════════════════════════════════
PHASE 5 — CI/CD (GitHub Actions)
════════════════════════════════════════════════════════════
47.  .github/workflows/deploy-droplet.yml
48.  .github/workflows/deploy-k8s.yml
49.  .github/workflows/deploy-firebase.yml
     TEST: push to main → verify all three workflows pass in GitHub Actions tab
     TEST: end-to-end — send a message from Firebase URL, verify response streams back
```

---

## Non-Negotiable Constraints

These are hard rules. OpenCode must not deviate from them for any reason.

1. **Static frontend** — `output: 'export'` in next.config.js. No SSR. No Next.js API routes.
2. **All inference on compute droplet** — llama-server and backend never move to K8s.
3. **K8s worker is proxy only** — no ML, no database access, no business logic.
4. **Model files never in git** — GGUF + mmproj live on block volume, downloaded via script.
5. **`.env` never in git** — use GitHub Secrets for CI/CD, manual scp for droplet.
6. **restart: always on everything** — Docker Compose must recover from any crash automatically.
7. **Always stream** — `/chat` endpoint streams SSE tokens. Never buffer the full response.
8. **Vision model handles images natively** — never add pytesseract or any OCR for images.
9. **Destroy GPU droplet after training** — it bills per second even when idle.
10. **Port 8000 is VPC-only** — UFW blocks all external access. Only K8s worker private IP reaches backend.
11. **aiosqlite everywhere** — never use synchronous sqlite3 in async FastAPI handlers.
12. **Single ChromaDB collection** — `nyapsys_kb` only. Scope queries by conversation_id metadata filter.
13. **Batch embedding** — never embed one chunk at a time. Always batch before upserting to ChromaDB.
14. **mmproj is required** — llama-server must always load both the GGUF and the mmproj file.
    Without mmproj, Vision inputs will fail at runtime.
