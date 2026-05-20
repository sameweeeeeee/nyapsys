# Nyapsys

Self-hosted AI agent with file and image understanding. Built from scratch with a 1B parameter model, running on MacBook Air M3.

## Features

- **Conversational AI** - Chat with a fine-tuned 1B parameter GPT-style model
- **File Understanding** - Upload PDF, DOCX, TXT, CSV, JSON
- **Image Analysis** - Vision pipeline for images
- **RAG Knowledge Base** - ChromaDB vector store for uploaded files
- **Persistent Conversations** - SQLite database for chat history

## Architecture

```
User → Firebase Hosting → Cloudflare Tunnel → MacBook Air M3
                                                  ↓
                                            llama.cpp server
                                            FastAPI backend
                                            ChromaDB
                                            SQLite
```

## Prerequisites

- MacBook Air M3 (16GB recommended)
- [Firebase](https://firebase.google.com) account (free)
- [GCP](https://cloud.google.com/free) account ($300 credit, 90 days)
- GitHub account for CI/CD

---

## Quick Start

### 1. Clone the Repo

```bash
git clone https://github.com/sameweeeeeee/nyapsys.git
cd nyapsys
```

### 2. Create .env

```bash
cp .env.example .env
# Edit .env with your values
```

### 3. Setup Mac

```bash
brew install llama.cpp python@3.11 git wget
pip3.11 install -r backend/requirements.txt
mkdir -p ~/volumes/models ~/volumes/chromadb ~/volumes/sqlite
```

### 4. Download Model

```bash
# Edit scripts/download_model.sh with your GCS bucket
bash scripts/download_model.sh
```

### 5. Start Services

```bash
# Edit backend/run.sh with your paths
bash backend/run.sh
```

### 6. Setup Cloudflare Tunnel

```bash
brew install cloudflared
bash scripts/setup_tunnel.sh
```

### 7. Deploy Frontend

```bash
cd frontend
npm install
npm run build
firebase deploy
```

---

## Training (One-Time)

### 1. Upgrade GCP to Paid

GCP free trial blocks GPU access. Go to Billing → Upgrade to paid account.

### 2. Prepare Data (Mac)

```bash
cd training
pip install -r requirements.txt
python prepare_dataset.py
gsutil -m cp -r training/data/tokenized/ gs://nyapsys-training/tokenized/
```

### 3. Create GPU Instance

```bash
gcloud compute instances create nyapsys-training \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release \
  --provisioning-model=SPOT
```

### 4. Train

```bash
python training/train.py --config 1b --data_path gs://nyapsys-training/tokenized/
```

~120-160 hrs on L4 GPU (~$60-80 with spot pricing).

### 5. Instruction Tune

```bash
python training/instruction_tune.py \
  --base_model gs://nyapsys-training/checkpoints/final \
  --output_dir gs://nyapsys-training/instruct-output
```

### 6. Export GGUF

```bash
python training/merge_and_export.py \
  --model_path gs://nyapsys-training/instruct-output \
  --output_path ./Nyapsys-1B.Q4_K_M.gguf
```

### 7. Upload to GCS

```bash
python training/upload_to_gcs.py \
  --file ./Nyapsys-1B.Q4_K_M.gguf \
  --bucket nyapsys-models
```

### 8. Destroy Instance

```bash
gcloud compute instances delete nyapsys-training --zone=us-central1-a --quiet
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_SECRET_KEY` | `openssl rand -hex 32` |
| `GCS_BUCKET` | Your GCS bucket name |
| `NEXT_PUBLIC_API_URL` | Your domain |

---

## GitHub Actions CI/CD

Set secrets:

| Secret | Description |
|--------|-------------|
| `FIREBASE_SERVICE_ACCOUNT` | Firebase JSON key |
| `MAC_SSH_KEY` | SSH key for Mac via Tailscale |
| `MAC_TAILSCALE_IP` | Tailscale IP of your Mac |
| `NEXT_PUBLIC_API_URL` | Your API URL |
| `NEXT_PUBLIC_API_KEY` | Same as API_SECRET_KEY |

---

## Cost

| Item | Cost |
|------|------|
| GCP L4 GPU (~140 hrs spot) | ~$60-80 (covered by $300 credit) |
| GCS Storage | ~$0.01/mo |
| Firebase Hosting | $0 |
| Cloudflare Tunnel | $0 |
| Mac inference | $0 |
| **Monthly** | **~$0** |

---

## API Endpoints

| Endpoint | Method | Auth |
|----------|--------|------|
| `/chat` | POST | Bearer token |
| `/ingest` | POST | Bearer token |
| `/conversations` | GET | Bearer token |
| `/conversations/{id}/messages` | GET | Bearer token |
| `/conversations/{id}` | DELETE | Bearer token |
| `/health` | GET | None |

---

## License

MIT