#!/usr/bin/env bash
# Start the Abo Ali smart-search web app on http://localhost:8901
cd "$(dirname "$0")"
export HF_HOME=/data/huggingface
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# AI answers use a local OpenAI-compatible LLM. Default: auto-discovers the
# scrib-r llama.cpp container. Override: export LLM_BASE_URL / LLM_MODEL_ID.
exec .venv/bin/uvicorn app:app --host 0.0.0.0 --port 8901
