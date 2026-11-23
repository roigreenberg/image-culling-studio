# Security & Zero-Secrets Policy

**In no way may any secret, API key, token, `.env` file, personal filesystem path, private photo/RAW/XMP file, or local culling output ever be committed to this repository.**

---

## What Must Never Be Committed

1. **Secrets & Credentials:** API keys, Hugging Face tokens (`hf_...`), GitHub tokens (`ghp_...`, `gho_...`), AWS keys (`AKIA...`), OpenAI/LLM keys (`sk-...`), bearer tokens, private keys (`*.pem`, `*.key`, `*.p12`), or `.env` files.
2. **Personal Filesystem Paths:** Hardcoded local user directories (`C:\Users\...`, `/Users/...`, `/home/...`) in source code, configs, or documentation. Always pass paths via CLI flags (`--input-dir`, `--output-dir`) or an untracked local `.env` file.
3. **Private Media & Local Outputs:** Camera RAWs (`*.CR3`, `*.NEF`, `*.ARW`, `*.DNG`), JPEGs (`*.JPG`, `*.JPEG`), XMP sidecars (`*.xmp`), `./cull_output/`, `./studio_data/`, `my_picks.json`, `embeddings.npz`, or trained personal ranker files (`*.pkl`).
4. **Downloaded Model Weights & Caches:** `*.onnx`, `*.onnx.data`, `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `.venv/`, `__pycache__/`.

---

## Automated Pre-Commit Guard

This repository includes a local Git pre-commit hook at `./scripts/pre-commit` (installed at `.git/hooks/pre-commit`) that scans every staged file before each commit and **blocks the commit** if it detects:

- Blocked file extensions or sensitive filenames (`.env`, keys/certs, RAW/image/XMP files, model weights, `cull_output/`, `my_picks.json`, `embeddings.npz`).
- Secret patterns (`AKIA...`, `hf_...`, `ghp_...`, `gho_...`, `sk-...`, `Bearer `, `PRIVATE KEY`, or variable assignments matching `API_KEY`, `SECRET`, `TOKEN`, `PASSWORD`).
- Hardcoded local user paths (`C:\Users\`, `/Users/`, `/home/`).

### Installing the Hook on a Fresh Clone
```powershell
Copy-Item scripts\pre-commit .git\hooks\pre-commit -Force
```
On macOS/Linux:
```bash
cp scripts/pre-commit .git/hooks/pre-commit && chmod +x .git/hooks/pre-commit
```
