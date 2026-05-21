# Nyapsys

Nyapsys is a self-hosted AI assistant that runs entirely on your MacBook Air M3. It features a from-scratch 2B Mixture-of-Experts (MoE) language model, file and image understanding, and a warm, premium chat interface. No cloud APIs, no subscriptions, no data leaves your machine.

## Architecture

```
User browser → Firebase Hosting (static Next.js) → Cloudflare Tunnel → MacBook Air M3
                                                                         ├── llama-server (127.0.0.1:8080)
                                                                         ├── FastAPI backend (127.0.0.1:8000)
                                                                         ├── ChromaDB (127.0.0.1:8001)
                                                                         └── SQLite (~/volumes/sqlite/)
```

## Requirements

- MacBook with Apple Silicon (M1/M2/M3), 16GB RAM minimum
- macOS 13+
- Python 3.11
- Node.js 20
- Homebrew
- GCP account (one-time, for model training only)
- Cloudflare account (free)
- Firebase project (free)
- A domain name

## Quick Start

1. **Clone the repo:** `git clone https://github.com/YOUR_USERNAME/nyapsys.git && cd nyapsys`
2. **Run setup:** `bash scripts/setup_mac.sh` (installs dependencies, configures services)
3. **Train the model:** Follow `training/README.md` on a GCP L4 GPU (or skip if you already have the GGUF)
4. **Open the frontend:** Visit your Firebase Hosting URL or run `cd frontend && npm run dev`

## Training

Nyapsys uses a from-scratch 2B MoE model (4 experts × 500M params, top-2 routing = 1B active per token) trained on an NVIDIA L4 GPU on GCP. The entire training cost (~$37–48) is covered by GCP's $300 free credit, leaving ~$252–263 for retrains. See `training/README.md` for the full walkthrough and `AGENTS.md` for architectural decisions.

## Downloading the Model (After Training)

```bash
gcloud auth login
gsutil cp gs://nyapsys-models/Nyapsys-2B-MoE.Q4_K_M.gguf ~/volumes/models/
bash backend/run.sh
```

See `AGENTS.md` for verification and troubleshooting steps.

## Development

```bash
# Backend
bash backend/run.sh

# Frontend (dev mode)
cd frontend && npm install && npm run dev

# API health check
curl http://localhost:8000/health
```

## Deployment

Push to `main` on GitHub — Actions automatically deploy the frontend to Firebase and the backend to your Mac via SSH over Tailscale.

## Project Structure

```
nyapsys/
├── training/          # Model training scripts (GCP GPU)
├── backend/           # FastAPI inference server + agent loop
├── frontend/          # Next.js static chat interface
├── scripts/           # Setup, download, and deployment scripts
├── volumes/           # Local storage (gitignored)
└── AGENTS.md          # Full architecture and build guide
```

## Contributing

Read `AGENTS.md` before making changes — every architectural decision is documented there.

## License

MIT