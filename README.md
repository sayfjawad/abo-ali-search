# Abo Ali Smart Search

**AI-enabled semantic search and RAG question-answering over a large archive of
transcribed Arabic videos — fully local, GPU-accelerated, with time-indexed,
playable results.**

Live deployment: <https://aboali.scrib-r.com>

---

## 1. What the application is

This application makes a ~2,500-episode video archive (2015–2026, ≈500 GB of
media plus speaker-diarized transcripts) *searchable by meaning*. Instead of
keyword matching, it embeds every transcript segment into a multilingual vector
space, so a question phrased in your own words finds the moments where the
speaker actually discussed that topic — even when the wording differs.

Two interaction modes are offered through a single Arabic (RTL) web page:

| Mode | What happens |
|---|---|
| **🔍 بحث فقط** (semantic search) | The query is embedded on the GPU and matched against ~160,000 transcript chunks. Results appear in a table: episode **date**, **in-video timestamp**, the matching **highlighted text**, and the **episode title** — sorted by relevance and filterable by date range. |
| **🤖 اسأل الذكاء الاصطناعي** (Ask AI / RAG) | The top-matching clips are handed to a **local LLM**, which writes a sourced Arabic answer describing *what was said and when*. Citations like `[3]` in the answer click through to the corresponding clip row. |

Every result row offers two ways to *watch the exact moment*:

- **▶ تشغيل** — plays the local media file in an in-page player, automatically
  seeked to ~2 seconds before the quoted passage (served with HTTP Range
  support, so seeking inside multi-GB files is instant).
- **YouTube ↗** — a deep link of the form
  `https://www.youtube.com/watch?v=<id>&t=<seconds>s` that opens the same
  moment online; suitable for sharing with end-users.

Everything — embedding, retrieval, answer generation, media streaming — runs on
the local machine. No cloud APIs are involved and no data leaves the host.

---

## 2. Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │  FastAPI app (uvicorn, port 8901)            │
 Browser (RTL SPA) ───▶│                                              │
   static/index.html   │  /api/search ──▶ BGE-M3 query embedding      │
                       │                  cosine top-k on GPU          │
                       │                  (~160k × 1024 fp16 matrix)   │
                       │                                              │
                       │  /api/ask ─────▶ retrieval (as above)        │
                       │                  → OpenAI-compatible local   │
                       │                    LLM (llama.cpp / Qwen3)   │
                       │                  → cited Arabic answer       │
                       │                                              │
                       │  /media/{file} ─▶ Range-enabled streaming    │
                       │                   of local mp4/m4a           │
                       └──────────────────────────────────────────────┘
                                        ▲
                     build_index.py (offline, one-time / on new videos)
                     parses transcripts → chunks → GPU embeddings
                     → index/index.sqlite + index/embeddings.npy
```

### Retrieval design

- **Chunking** — consecutive segments by the same speaker are merged up to
  ~700 characters (`MERGE_TARGET_CHARS` in `build_index.py`), preserving the
  original start/end timestamps. This gives each vector enough context while
  keeping timestamps precise. ~2,500 videos → ~160,000 chunks.
- **Embeddings** — [`BAAI/bge-m3`](https://huggingface.co/BAAI/bge-m3)
  (dense mode, 1024-d, fp16), among the strongest open multilingual retrieval
  models for Arabic. Loaded via `transformers` directly (CLS pooling +
  L2-normalization) — no heavyweight framework needed.
- **Vector search** — deliberately *not* a vector database. The whole matrix
  (160k × 1024 fp16 ≈ 320 MB) sits on the GPU and a query is a single matmul:
  a few milliseconds, exact (no ANN approximation), zero infrastructure. This
  scales comfortably into the millions of chunks before FAISS/qdrant would pay
  off.
- **Date filtering** — each chunk carries its episode's upload date; filters
  are applied by masking scores before top-k.

### RAG answer generation

`/api/ask` builds a prompt of numbered excerpts (episode title, date,
timestamp, text) plus the user's question, and calls **any OpenAI-compatible
chat endpoint**. By default it auto-discovers a llama.cpp container named
`scrib-r-backend-llama-1` (Qwen3-8B) via `docker inspect`; override with
environment variables (see Configuration). `<think>…</think>` blocks emitted
by reasoning models are stripped; citation markers `[n]` in the answer are
cross-linked to the source table.

---

## 3. Technologies

| Layer | Technology | Why |
|---|---|---|
| Embeddings | `BAAI/bge-m3` via 🤗 `transformers`, PyTorch, CUDA fp16 | Best-in-class open multilingual (incl. Arabic) dense retrieval; runs on a single V100 |
| Vector search | PyTorch matmul on GPU | Exact, ~ms latency at this scale, no extra service |
| Metadata store | SQLite | Single-file, zero-ops storage for video + chunk metadata |
| Backend | FastAPI + uvicorn | Async Python API, static file serving, Range-capable `FileResponse` |
| Answer LLM | Any OpenAI-compatible server (llama.cpp, LM Studio, Ollama, vLLM) | Provider-agnostic; default is a llama.cpp container with Qwen3-8B |
| Frontend | Vanilla HTML/CSS/JS, RTL, dark-mode aware, responsive (mobile card layout) | No build step, no framework — one file |
| Deployment | systemd service + nginx reverse proxy + Let's Encrypt (certbot) | Boot persistence, TLS, public hostname |

---

## 4. Expected data layout

The indexer reads a flat directory (default `/data/ABO_ALI`) with, per video:

```
<YYYYMMDD>_<youtube-id>.metadata.json   # title, url, upload_date, duration
<YYYYMMDD>_<youtube-id>.json            # transcript: segments[] with
                                        #   speaker_id, speaker, start, end, text
<YYYYMMDD>_<youtube-id>.mp4             # (or .m4a) local media, optional
```

`metadata.json` example:

```json
{
  "id": "MwBBTTMiS5s",
  "title": "…",
  "url": "https://www.youtube.com/watch?v=MwBBTTMiS5s",
  "upload_date": "20150110",
  "duration_seconds": 4131.0
}
```

Transcripts in this format are produced by a WhisperX + pyannote pipeline
(e.g. [scrib-r](https://github.com/sayfjawad/scrib-r)); any source works as
long as the JSON shape matches.

---

## 5. Setup

### Requirements

- Linux, Python 3.10+
- NVIDIA GPU with ≥ 8 GB VRAM (tested: Tesla V100 32 GB) + CUDA-enabled PyTorch
- ~500 MB disk for the index (at 160k chunks)
- Optional: a running OpenAI-compatible LLM server for the Ask-AI mode
- Optional: Docker (only for auto-discovery of the default llama.cpp container)

### Install

```bash
git clone git@github.com:sayfjawad/abo-ali-search.git
cd abo-ali-search
python3 -m venv --system-site-packages .venv   # reuse a system CUDA torch if present
.venv/bin/pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" transformers torch numpy httpx
```

> If your system Python already provides a CUDA build of `torch` (as on ML
> hosts), `--system-site-packages` avoids a second multi-GB install. On a clean
> machine, drop the flag and let pip install torch.

### Build the index

```bash
# adjust DATA_DIR in build_index.py if your archive lives elsewhere
CUDA_VISIBLE_DEVICES=0 .venv/bin/python build_index.py
```

First run downloads the BGE-M3 model (~2.3 GB) into `HF_HOME`. Indexing the
full 2,500-video corpus takes ~8 minutes on a V100. Re-run whenever new videos
are added (the index is rebuilt from scratch; it's cheap).

### Run

```bash
./run.sh                # http://localhost:8901
```

### Run as a service (systemd)

```bash
sudo cp deploy/aboali-search.service /etc/systemd/system/
# edit User/WorkingDirectory/paths if they differ
sudo systemctl daemon-reload
sudo systemctl enable --now aboali-search
```

### Public HTTPS via nginx + certbot (optional)

On the internet-facing host (may be a different machine reaching the app over
a VPN/overlay network):

```bash
sudo cp deploy/nginx-aboali.conf /etc/nginx/sites-available/aboali.example.com
# edit server_name and the proxy_pass target
sudo ln -s /etc/nginx/sites-available/aboali.example.com /etc/nginx/sites-enabled/
sudo nginx -t && sudo systemctl reload nginx
sudo certbot --nginx -d aboali.example.com --redirect
```

`proxy_buffering off` and long timeouts in the shipped config matter for
smooth video streaming through the proxy.

> ⚠️ The app itself has **no authentication**. If exposed publicly, anyone can
> search, stream the media files, and use the (GPU-consuming) Ask-AI endpoint.
> Gate it with nginx basic-auth, an allowlist, or an auth proxy if needed.

---

## 6. Configuration

All optional, via environment variables (see `run.sh` / the systemd unit):

| Variable | Default | Purpose |
|---|---|---|
| `HF_HOME` | `/data/huggingface` | Hugging Face model cache location |
| `CUDA_VISIBLE_DEVICES` | `0` | Which GPU embeds queries / holds the matrix |
| `LLM_BASE_URL` | auto-discover `scrib-r-backend-llama-1` container | OpenAI-compatible endpoint for Ask-AI, e.g. `http://localhost:1234/v1` |
| `LLM_MODEL_ID` | `qwen3-8b` | Model name passed to the endpoint |
| `LLM_API_KEY` | `none` | Bearer token if the endpoint requires one |

Paths that live in code: archive directory (`DATA_DIR` in `build_index.py`,
`MEDIA_DIR` in `app.py`) and chunk-merge size (`MERGE_TARGET_CHARS`).

---

## 7. HTTP API

| Endpoint | Method | Body / params | Returns |
|---|---|---|---|
| `/api/search` | POST | `{query, top_k?, date_from?, date_to?}` (dates `YYYYMMDD`) | `{results: [{score, video, title, date, start, end, start_fmt, text, youtube_url, media_url, …}]}` |
| `/api/ask` | POST | `{question, top_k?, date_from?, date_to?}` | `{answer, error, sources: [...]}` — sources carry `cited: true` where referenced |
| `/api/stats` | GET | — | `{videos, chunks}` |
| `/media/{file}` | GET | Range supported | mp4/m4a stream |
| `/` | GET | — | the single-page UI |

Example:

```bash
curl -s localhost:8901/api/search -H 'Content-Type: application/json' \
  -d '{"query":"ماذا قال عن الزلازل في تركيا","top_k":5,"date_from":"20160101"}'
```

---

## 8. Performance (reference hardware: Tesla V100 32 GB)

| Operation | Measured |
|---|---|
| Full index build (2,523 videos → 159,532 chunks) | ~8 min |
| Semantic search query (embed + exact top-k) | < 100 ms |
| RAG answer (Qwen3-8B, 12 excerpts) | ~4–5 s |
| GPU memory (serving) | ~1.8 GB |

---

## 9. Repository layout

```
app.py                  FastAPI backend (search, ask, media, stats)
build_index.py          offline indexer (parse → chunk → embed → store)
embedder.py             BGE-M3 embedding wrapper (transformers, fp16)
static/index.html       the entire frontend (RTL, responsive, dark-mode)
run.sh                  dev/foreground launcher
deploy/
  aboali-search.service systemd unit
  nginx-aboali.conf     reverse-proxy template (TLS added by certbot)
index/                  generated — not committed (sqlite + npy)
```
