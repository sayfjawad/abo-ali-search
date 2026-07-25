"""Build the search index from /data/ABO_ALI transcripts.

Reads every <base>.json transcript + <base>.metadata.json, merges consecutive
same-speaker segments into retrieval chunks, embeds them with BGE-M3 on GPU,
and writes:
  index/index.sqlite   - videos + chunks tables
  index/embeddings.npy - fp16 (n_chunks, 1024), row i == chunk id i
"""
import json
import os
import shutil
import sqlite3
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch

from embedder import Embedder, DIM

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/ABO_ALI"))
INDEX_DIR = Path(os.environ.get("INDEX_DIR", Path(__file__).parent / "index"))
MERGE_TARGET_CHARS = 700  # merge consecutive same-speaker segments up to this size


def iter_videos():
    for meta_path in sorted(DATA_DIR.glob("*.metadata.json")):
        base = meta_path.name[: -len(".metadata.json")]
        transcript_path = DATA_DIR / f"{base}.json"
        if not transcript_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"skip {base}: {e}", file=sys.stderr)
            continue
        yield base, meta, transcript


def merge_segments(segments):
    """Greedy merge of consecutive same-speaker segments into chunks."""
    chunks = []
    cur = None
    for seg in segments:
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        if (
            cur is not None
            and seg.get("speaker_id") == cur["speaker_id"]
            and len(cur["text"]) + len(text) <= MERGE_TARGET_CHARS
        ):
            cur["text"] += " " + text
            cur["end"] = seg["end"]
        else:
            if cur:
                chunks.append(cur)
            cur = {
                "speaker_id": seg.get("speaker_id") or "",
                "speaker": seg.get("speaker") or "",
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
            }
    if cur:
        chunks.append(cur)
    return chunks


BACKUP_RETENTION_DAYS = 14


def backup_and_prune_index():
    """Copy the current index.sqlite + embeddings.npy to
    index/backups/<YYYY-MM-DD>/ before this run overwrites them, and drop
    backups older than BACKUP_RETENTION_DAYS. Added 2026-07-24 after a
    rebuild wiped the live index down to 1 video with no way back short of
    restoring the source transcripts from a copy found on another host --
    never again should one bad build_index.py run be unrecoverable."""
    backups_dir = INDEX_DIR / "backups"
    today_dir = backups_dir / date.today().isoformat()
    src_db = INDEX_DIR / "index.sqlite"
    src_emb = INDEX_DIR / "embeddings.npy"
    if not today_dir.exists() and (src_db.exists() or src_emb.exists()):
        today_dir.mkdir(parents=True, exist_ok=True)
        for src in (src_db, src_emb):
            if src.exists():
                shutil.copy2(src, today_dir / src.name)
        print(f"backup: {today_dir}", flush=True)
    cutoff = date.today() - timedelta(days=BACKUP_RETENTION_DAYS)
    if backups_dir.exists():
        for d in backups_dir.iterdir():
            try:
                old = date.fromisoformat(d.name) < cutoff
            except ValueError:
                continue
            if old:
                shutil.rmtree(d, ignore_errors=True)


def find_media_file(base: str) -> str:
    for ext in (".mp4", ".f140.m4a", ".m4a"):
        if (DATA_DIR / f"{base}{ext}").exists():
            return f"{base}{ext}"
    return ""


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    backup_and_prune_index()
    db_path = INDEX_DIR / "index.sqlite"
    if db_path.exists():
        db_path.unlink()
    db = sqlite3.connect(db_path)
    db.executescript("""
        CREATE TABLE videos (
            video TEXT PRIMARY KEY, yt_id TEXT, title TEXT, url TEXT,
            upload_date TEXT, duration REAL, media_file TEXT
        );
        CREATE TABLE chunks (
            id INTEGER PRIMARY KEY, video TEXT, speaker TEXT,
            start REAL, end REAL, text TEXT
        );
    """)

    print("pass 1: parsing transcripts...", flush=True)
    all_texts = []
    chunk_id = 0
    n_videos = 0
    for base, meta, transcript in iter_videos():
        chunks = merge_segments(transcript.get("segments") or [])
        if not chunks:
            continue
        db.execute(
            "INSERT INTO videos VALUES (?,?,?,?,?,?,?)",
            (
                base,
                meta.get("id") or "",
                meta.get("title") or transcript.get("title") or base,
                meta.get("url") or "",
                meta.get("upload_date") or base[:8],
                meta.get("duration_seconds") or transcript.get("duration_seconds") or 0,
                find_media_file(base),
            ),
        )
        for c in chunks:
            db.execute(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?)",
                (chunk_id, base, c["speaker"], c["start"], c["end"], c["text"]),
            )
            all_texts.append(c["text"])
            chunk_id += 1
        n_videos += 1
        if n_videos % 250 == 0:
            print(f"  {n_videos} videos, {chunk_id} chunks", flush=True)
    db.commit()
    print(f"parsed {n_videos} videos -> {chunk_id} chunks", flush=True)

    print("pass 2: embedding on GPU...", flush=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    embedder = Embedder(device=device)
    tok = embedder.tokenizer

    # Sort by tokenized length for efficient batching, then restore order.
    lengths = [len(t) for t in all_texts]  # char length is a good proxy
    order = np.argsort(lengths)
    emb = np.zeros((len(all_texts), DIM), dtype=np.float16)

    TOKEN_BUDGET = 16384  # max tokens per batch (padded)
    batch_idx: list[int] = []
    batch_max_tok = 0
    done = 0
    t0 = time.time()

    def flush():
        nonlocal batch_idx, batch_max_tok, done
        if not batch_idx:
            return
        texts = [all_texts[i] for i in batch_idx]
        vecs = embedder.encode(texts).numpy().astype(np.float16)
        emb[batch_idx] = vecs
        done += len(batch_idx)
        if done % 5000 < len(batch_idx):
            rate = done / (time.time() - t0)
            eta = (len(all_texts) - done) / max(rate, 1)
            print(f"  {done}/{len(all_texts)}  {rate:.0f} chunks/s  eta {eta/60:.1f} min", flush=True)
        batch_idx = []
        batch_max_tok = 0

    for i in order:
        ntok = min(len(tok.encode(all_texts[i], add_special_tokens=True)), embedder.max_length)
        new_max = max(batch_max_tok, ntok)
        if batch_idx and new_max * (len(batch_idx) + 1) > TOKEN_BUDGET:
            flush()
            new_max = ntok
        batch_idx.append(int(i))
        batch_max_tok = new_max
    flush()

    np.save(INDEX_DIR / "embeddings.npy", emb)
    db.close()
    print(f"done in {(time.time()-t0)/60:.1f} min -> {INDEX_DIR}", flush=True)


if __name__ == "__main__":
    main()
