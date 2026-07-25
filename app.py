"""Abo Ali smart-search web application.

- POST /api/search  semantic search over all transcripts (date-filterable)
- POST /api/ask     RAG: retrieve relevant clips + Claude-generated answer with citations
- GET  /media/{f}   serve local mp4/m4a with HTTP Range (seekable playback)
- /                 single-page frontend (static/index.html)
"""
import os
import re
import sqlite3
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from embedder import Embedder

BASE = Path(__file__).parent
INDEX_DIR = Path(os.environ.get("INDEX_DIR", BASE / "index"))
MEDIA_DIR = Path(os.environ.get("DATA_DIR", "/data/ABO_ALI"))
STATS_DB = BASE / "stats.sqlite"  # outside index/ so re-indexing keeps history

app = FastAPI(title="Abo Ali Smart Search")

# ---------------------------------------------------------------- index state
_state: dict = {}


@app.on_event("startup")
def load_index():
    db = sqlite3.connect(INDEX_DIR / "index.sqlite", check_same_thread=False)
    db.row_factory = sqlite3.Row
    emb = np.load(INDEX_DIR / "embeddings.npy")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    matrix = torch.from_numpy(emb).to(device)  # (n, 1024) fp16
    # upload_date per chunk for fast date filtering
    rows = db.execute(
        "SELECT c.id, v.upload_date FROM chunks c JOIN videos v ON v.video = c.video ORDER BY c.id"
    ).fetchall()
    dates = np.array([int(r["upload_date"] or 0) for r in rows], dtype=np.int64)
    _state.update(
        db=db,
        matrix=matrix,
        dates=torch.from_numpy(dates).to(device),
        embedder=Embedder(device=device, max_length=512),
        device=device,
    )
    print(f"index loaded: {matrix.shape[0]} chunks on {device}")


# ------------------------------------------------------------- usage tracking
def _stats_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(STATS_DB, timeout=10)
    conn.execute("""CREATE TABLE IF NOT EXISTS queries (
        ts TEXT DEFAULT (datetime('now')), mode TEXT, ip TEXT, query TEXT)""")
    return conn


def client_ip(request: Request) -> str:
    # behind nginx: X-Real-IP / first hop of X-Forwarded-For; else socket peer
    ip = request.headers.get("x-real-ip") or ""
    if not ip:
        fwd = request.headers.get("x-forwarded-for", "")
        ip = fwd.split(",")[0].strip()
    return ip or (request.client.host if request.client else "unknown")


def log_query(request: Request, mode: str, query: str):
    try:
        with _stats_conn() as conn:
            conn.execute(
                "INSERT INTO queries (mode, ip, query) VALUES (?,?,?)",
                (mode, client_ip(request), query[:500]),
            )
    except Exception as e:  # stats must never break search
        print(f"stats logging failed: {e}")


@app.get("/api/statistics")
def api_statistics():
    with _stats_conn() as conn:
        total = conn.execute("SELECT COUNT(*) FROM queries").fetchone()[0]
        by_mode = dict(conn.execute(
            "SELECT mode, COUNT(*) FROM queries GROUP BY mode").fetchall())
        unique_ips = conn.execute(
            "SELECT COUNT(DISTINCT ip) FROM queries").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT ip) FROM queries WHERE date(ts) = date('now')"
        ).fetchone()
        per_day = conn.execute("""
            SELECT date(ts) d, COUNT(*), COUNT(DISTINCT ip)
            FROM queries WHERE ts >= datetime('now','-14 days')
            GROUP BY d ORDER BY d DESC""").fetchall()
        first = conn.execute("SELECT MIN(date(ts)) FROM queries").fetchone()[0]
    return {
        "total_queries": total,
        "search_queries": by_mode.get("search", 0),
        "ask_queries": by_mode.get("ask", 0),
        "unique_ips": unique_ips,
        "today_queries": today[0],
        "today_unique_ips": today[1],
        "since": first,
        "per_day": [{"date": d, "queries": q, "unique_ips": u} for d, q, u in per_day],
    }


# ------------------------------------------------------------------ retrieval
def fmt_time(seconds: float) -> str:
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def yt_link(url: str, start: float) -> str:
    if not url:
        return ""
    sep = "&" if "?" in url else "?"
    return f"{url}{sep}t={max(int(start) - 2, 0)}s"


def retrieve(query: str, top_k: int, date_from: str | None, date_to: str | None):
    st = _state
    q = st["embedder"].encode([query]).to(st["device"])  # (1, 1024)
    scores = (st["matrix"] @ q.T).squeeze(1).float()  # (n,)
    if date_from:
        scores = torch.where(st["dates"] >= int(date_from), scores, torch.tensor(-1.0, device=scores.device))
    if date_to:
        scores = torch.where(st["dates"] <= int(date_to), scores, torch.tensor(-1.0, device=scores.device))
    k = min(top_k, scores.shape[0])
    vals, idx = torch.topk(scores, k)
    results = []
    for score, cid in zip(vals.tolist(), idx.tolist()):
        if score < 0:
            continue
        row = st["db"].execute(
            """SELECT c.*, v.title, v.url, v.upload_date, v.media_file, v.yt_id
               FROM chunks c JOIN videos v ON v.video = c.video WHERE c.id = ?""",
            (cid,),
        ).fetchone()
        d = row["upload_date"] or ""
        results.append({
            "score": round(score, 4),
            "video": row["video"],
            "title": row["title"],
            "date": f"{d[:4]}-{d[4:6]}-{d[6:8]}" if len(d) == 8 else d,
            "speaker": row["speaker"],
            "start": row["start"],
            "end": row["end"],
            "start_fmt": fmt_time(row["start"]),
            "end_fmt": fmt_time(row["end"]),
            "text": row["text"],
            "youtube_url": yt_link(row["url"], row["start"]),
            "media_url": f"/media/{row['media_file']}#t={max(int(row['start']) - 2, 0)}" if row["media_file"] else "",
            "media_file": row["media_file"],
        })
    return results


class SearchReq(BaseModel):
    query: str
    top_k: int = 20
    date_from: str | None = None  # YYYYMMDD
    date_to: str | None = None


@app.post("/api/search")
def api_search(req: SearchReq, request: Request):
    if not req.query.strip():
        raise HTTPException(400, "empty query")
    log_query(request, "search", req.query)
    return {"results": retrieve(req.query, min(req.top_k, 100), req.date_from, req.date_to)}


# ------------------------------------------------------------------------ RAG
# Answer generation uses any OpenAI-compatible endpoint (llama.cpp, LM Studio,
# vLLM, Ollama...). Configure with env vars:
#   LLM_BASE_URL  e.g. http://localhost:1234/v1   (default: auto-discover the
#                 scrib-r llama.cpp container and use it)
#   LLM_MODEL_ID  default: qwen3-8b
#   LLM_API_KEY   default: none (local servers ignore it)
import subprocess

LLM_MODEL_ID = os.environ.get("LLM_MODEL_ID", "qwen3-8b")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "none")


def llm_base_url() -> str | None:
    url = os.environ.get("LLM_BASE_URL")
    if url:
        return url.rstrip("/")
    cached = _state.get("llm_base_url")
    if cached:
        return cached
    try:  # auto-discover the scrib-r llama.cpp container
        ip = subprocess.run(
            ["docker", "inspect", "-f",
             "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}",
             "scrib-r-backend-llama-1"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
        if ip:
            _state["llm_base_url"] = f"http://{ip}:8080/v1"
            return _state["llm_base_url"]
    except Exception:
        pass
    return None


ANSWER_SYSTEM = """أنت مساعد بحث في أرشيف مفرّغ نصياً لمقاطع فيديو للشيخ أبو علي الشيباني.
ستصلك مقتطفات مرقّمة من التفريغ النصي (قد تحتوي أخطاء تفريغ صوتي — تعامل معها بذكاء).
أجب على سؤال المستخدم اعتماداً على المقتطفات فقط:
- اذكر ماذا قال ومتى (تاريخ الحلقة والتوقيت داخلها).
- ضع رقم المرجع بين قوسين معقوفين مثل [3] بعد كل معلومة، ليتمكن المستخدم من فتح المقطع.
- إن لم تجد إجابة في المقتطفات فقل ذلك صراحةً.
- أجب بالعربية الفصحى الواضحة والموجزة."""


class AskReq(BaseModel):
    question: str
    top_k: int = 16
    date_from: str | None = None
    date_to: str | None = None


@app.post("/api/ask")
def api_ask(req: AskReq, request: Request):
    if not req.question.strip():
        raise HTTPException(400, "empty question")
    log_query(request, "ask", req.question)
    sources = retrieve(req.question, min(req.top_k, 60), req.date_from, req.date_to)
    answer, error = None, None
    base_url = llm_base_url()
    if not base_url:
        return {"answer": None, "error": "no_llm", "sources": sources}
    try:
        import httpx
        excerpts = "\n\n".join(
            f"[{i+1}] الحلقة: {s['title']} | التاريخ: {s['date']} | التوقيت: {s['start_fmt']}–{s['end_fmt']}\n{s['text']}"
            for i, s in enumerate(sources)
        )
        resp = httpx.post(
            f"{base_url}/chat/completions",
            headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            json={
                "model": LLM_MODEL_ID,
                "max_tokens": 2048,
                "temperature": 0.3,
                "messages": [
                    {"role": "system", "content": ANSWER_SYSTEM},
                    {"role": "user",
                     "content": f"المقتطفات:\n\n{excerpts}\n\nالسؤال: {req.question} /no_think"},
                ],
            },
            timeout=180.0,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"]
        # strip <think>...</think> reasoning blocks some local models emit
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        cited = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
        if cited:
            for i, s in enumerate(sources):
                s["cited"] = (i + 1) in cited
    except Exception as e:
        error = str(e)
    return {"answer": answer, "error": error, "sources": sources}


@app.get("/api/stats")
def api_stats():
    db = _state["db"]
    return {
        "videos": db.execute("SELECT COUNT(*) FROM videos").fetchone()[0],
        "chunks": db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0],
    }


# ---------------------------------------------------------------------- media
@app.get("/media/{filename}")
def media(filename: str):
    # prevent path traversal
    safe = os.path.basename(filename)
    path = MEDIA_DIR / safe
    if safe != filename or not path.is_file():
        raise HTTPException(404, "not found")
    mt = "video/mp4" if safe.endswith(".mp4") else "audio/mp4"
    return FileResponse(path, media_type=mt)


app.mount("/", StaticFiles(directory=BASE / "static", html=True), name="static")
