# Nyapsys

Self-hosted AI agent with file and image understanding. Built with Llama 3.2 11B Vision, running on DigitalOcean.

## Features

- **Conversational AI** - Chat with a fine-tuned Llama 3.2 Vision model
- **File Understanding** - Upload PDF, DOCX, TXT, CSV, JSON
- **Image Analysis** - Native multimodal - no OCR needed
- **RAG Knowledge Base** - ChromaDB vector store for uploaded files
- **Persistent Conversations** - SQLite database for chat history

## Architecture

```
User → Firebase Hosting → Cloudflare → K8s Middleware → Compute Droplet
                                                         ↓
                                                   llama-server (GGUF)
                                                   backend (FastAPI)
                                                   chromadb
```

## Prerequisites

- [Firebase](https://firebase.google.com) account (free)
- [DigitalOcean](https://digitalocean.com) account
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

### 3. Set Up Compute Droplet

Create a droplet: **s-2vcpu-8gb** (Ubuntu 22.04)

SSH in and run:
```bash
curl -fsSL https://get.docker.com | sh
apt-get install -y docker-compose-plugin git curl

# Mount block volume (50GB recommended)
# Follow DO docs to format and mount /dev/sda to /volumes

git clone https://github.com/sameweeeeeee/nyapsys /opt/nyapsys
cd /opt/nyapsys

# Edit .env with your values
bash scripts/download_model.sh
docker compose up -d
```

### 4. Set Up K8s (Middleware)

Create DOKS cluster (1 node, s-1vcpu-2gb)

```bash
kubectl apply -f middleware/k8s/namespace.yaml
# Edit secret.yaml with your API_SECRET_KEY first
kubectl apply -f middleware/k8s/secret.yaml
kubectl apply -f middleware/k8s/
```

### 5. Deploy Frontend

```bash
cd frontend
npm install
npm run build
firebase deploy
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `API_SECRET_KEY` | Random string - run `openssl rand -hex 32` |
| `BACKEND_URL` | Private IP of your compute droplet |
| `DO_SPACES_*` | DigitalOcean Spaces for model storage |
| `NEXT_PUBLIC_API_URL` | Your domain (e.g., https://api.nyapsys.com) |

---

## GitHub Actions CI/CD

Set these secrets in your repo:

| Secret | Value |
|--------|-------|
| `FIREBASE_SERVICE_ACCOUNT` | Firebase JSON key |
| `DO_ACCESS_TOKEN` | DO API token |
| `DROPLET_SSH_KEY` | SSH private key |
| `DROPLET_IP` | Public IP of droplet |
| `KUBECONFIG_DATA` | `doctl k8s cluster kubeconfig show nyapsys-cluster | base64 -w0` |
| `NEXT_PUBLIC_API_URL` | Your API URL |
| `NEXT_PUBLIC_API_KEY` | Same as API_SECRET_KEY |

Three workflows auto-deploy on push:
- **deploy-firebase.yml** - Frontend to Firebase
- **deploy-droplet.yml** - Backend to droplet
- **deploy-k8s.yml** - Middleware to K8s

---

## Model

The model is NOT in this repo. Download after fine-tuning:

```bash
bash scripts/download_model.sh
```

Or use the base model from HuggingFace:
- GGUF: https://huggingface.co/pbatra/Llama-3.2-11B-Vision-Instruct-GGUF
- mmproj: https://huggingface.co/leafspark/Llama-3.2-11B-Vision-Instruct-GGUF/resolve/main/mmproj-model-f16.gguf

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Stream chat (multipart/form-data) |
| `/ingest` | POST | Upload file to knowledge base |
| `/conversations` | GET | List all conversations |
| `/conversations/{id}/messages` | GET | Get messages |
| `/conversations/{id}` | DELETE | Delete conversation |
| `/feedback` | POST | Submit feedback |
| `/health` | GET | Health check (no auth) |

All endpoints except `/health` require `Authorization: Bearer YOUR_API_SECRET_KEY`

---

## Cost

| Resource | Monthly Cost |
|----------|-------------|
| Compute droplet (s-2vcpu-8gb) | $36 |
| K8s worker (s-1vcpu-2gb) | $12 |
| Block volume (50GB) | $5 |
| Firebase Hosting | $0 |
| Cloudflare DNS | $0 |
| **Total** | **~$53/mo** |

---

## Troubleshooting

**Backend not starting:**
```bash
docker compose logs backend
```

**Health check failing:**
```bash
curl http://localhost:8000/health
```

**View all containers:**
```bash
docker compose ps
```

**Restart a service:**
```bash
docker compose restart backend
```

---

## License

MIT