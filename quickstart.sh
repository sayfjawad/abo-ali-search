#!/usr/bin/env bash
# Build a working search index straight from the transcripts checked into
# this repo (archive-data/) -- no video download, no whisperx/diarization
# needed. Useful to try the app, or to rebuild the index from scratch if a
# live deployment's compiled index (index.sqlite + embeddings.npy,
# deliberately NOT checked in -- see archive-data/README.md) is ever lost:
# it regenerates from these transcripts in minutes.
#
# Usage: ./quickstart.sh
#
# Everything lands under ./local-data/, next to this script -- doesn't
# touch /data/ABO_ALI or any other path a production deployment uses.
set -eu
cd "$(dirname "$0")"

if [ ! -d "archive-data" ]; then
  echo "archive-data/ missing -- did you clone with it intact?" >&2
  exit 1
fi

export DATA_DIR="$PWD/local-data"
export INDEX_DIR="$PWD/local-index"
mkdir -p "$DATA_DIR" "$INDEX_DIR"
echo "copying archive-data/ -> $DATA_DIR/"
cp -r archive-data/. "$DATA_DIR/"

echo "building the index (embeds every chunk with BGE-M3 -- uses a GPU if"
echo "one is visible to torch, otherwise falls back to CPU and is slow)"
PY=.venv/bin/python3; [ -x "$PY" ] || PY=python3; "$PY" build_index.py

cat <<EOF

Done. Index built at $INDEX_DIR/ -- kept separate from the real
deployment's ./index/ via INDEX_DIR, so this never touches a live index.

Start the app with:
  DATA_DIR="$DATA_DIR" INDEX_DIR="$INDEX_DIR" .venv/bin/uvicorn app:app --port 8000

Then open http://localhost:8000 -- search and the AI-summary work; local
video playback won't, since the raw video files aren't checked into the
repo (only the transcripts derived from them are).
EOF
