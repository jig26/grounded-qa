from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import re
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("grounded-qa")

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Models ---------------- #

class Chunk(BaseModel):
    chunk_id: str
    text: str

class Query(BaseModel):
    question: Optional[str] = ""
    chunks: Optional[List[Chunk]] = []

# ---------------- Helpers ---------------- #

STOPWORDS = {
    "what", "which", "when", "where", "who", "why", "how",
    "is", "are", "was", "were", "do", "does", "did",
    "the", "a", "an", "of", "to", "in", "on", "for", "and",
    "or", "with", "by", "from", "as", "at", "be", "this",
    "that", "these", "those", "it", "its", "can", "will",
    "would", "should", "has", "have", "had", "there", "their"
}

WORD_RE = re.compile(r"\b[A-Za-z0-9]+\b")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str):
    return [w.lower() for w in WORD_RE.findall(text)]


def weighted_keywords(text: str):
    """Map lowercase keyword -> weight. Tokens that carry any uppercase
    letter in the original text (proper nouns, acronyms like 'FAISS' or
    'RFID') are the actual subject of most questions, so they're weighted
    much higher than generic lowercase overlap words."""
    weights = {}
    for tok in WORD_RE.findall(text):
        low = tok.lower()
        if low in STOPWORDS:
            continue
        w = 2.0 if tok != low else 1.0
        if low not in weights or w > weights[low]:
            weights[low] = w
    return weights


def split_sentences(text: str):
    if not text or not text.strip():
        return []
    parts = SENT_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def unanswerable(confidence: float = 0.2):
    return {
        "answer": "I don't know",
        "citations": [],
        "confidence": round(min(confidence, 0.3), 2),
        "answerable": False,
    }


# ---------------- Routes ---------------- #

@app.get("/")
def home():
    return {"status": "running"}


@app.post("/grounded-answer")
async def grounded_answer(request: Request):
    # Handle malformed / non-JSON bodies gracefully instead of raising 4xx/5xx
    try:
        raw = await request.json()
    except Exception:
        logger.info("REQUEST_BODY_UNPARSEABLE")
        return JSONResponse(content=unanswerable(0.0))

    logger.info("INCOMING_REQUEST: %s", json.dumps(raw)[:4000])

    def respond(payload):
        logger.info("OUTGOING_RESPONSE for question=%r: %s", raw.get('question') if isinstance(raw, dict) else None, json.dumps(payload))
        return JSONResponse(content=payload)

    try:
        req = Query(**raw) if isinstance(raw, dict) else Query()
    except Exception:
        return respond(unanswerable(0.0))

    question = (req.question or "").strip()
    chunks = req.chunks or []

    if not question or not chunks:
        return respond(unanswerable(0.0))

    qweights = weighted_keywords(question)
    if not qweights:
        # Question had no meaningful (non-stopword) terms to ground against
        return respond(unanswerable(0.0))
    qwords = set(qweights.keys())
    total_weight = sum(qweights.values())

    # Build every candidate (chunk_id, sentence, matched_keywords, score) with any overlap
    candidates = []
    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        text = getattr(chunk, "text", "") or ""
        if not chunk_id or not text.strip():
            continue
        for sent in split_sentences(text):
            swords = set(tokenize(sent))
            matched = qwords & swords
            if matched:
                weighted_score = sum(qweights[w] for w in matched)
                candidates.append({
                    "chunk_id": chunk_id,
                    "sentence": sent,
                    "matched": matched,
                    "score": weighted_score / total_weight,
                })

    if not candidates:
        return respond(unanswerable(0.2))

    # Sort by strength of match (most weighted keyword coverage first, ties
    # broken by shorter/tighter sentence so we don't drag in extra
    # unsupported text)
    candidates.sort(key=lambda c: (-c["score"], len(c["sentence"])))

    # Group candidate sentences by chunk, and score each CHUNK by its best
    # individual sentence (more precise than aggregating the whole chunk,
    # since a chunk may contain unrelated sentences too).
    by_chunk = {}
    for cand in candidates:
        cid = cand["chunk_id"]
        if cid not in by_chunk or cand["score"] > by_chunk[cid]["score"]:
            by_chunk[cid] = cand

    ranked_chunks = sorted(by_chunk.values(), key=lambda c: -c["score"])
    top_score = ranked_chunks[0]["score"]
    logger.info(
        "CHUNK_SCORES: %s",
        json.dumps([{"chunk_id": c["chunk_id"], "score": round(c["score"], 3), "sentence": c["sentence"][:80]} for c in ranked_chunks])
    )

    if top_score < 0.34:
        return respond(unanswerable(0.2))

    # Primary source: the single best-supported chunk. Pull in every
    # sentence from THAT chunk (and only that chunk) that also has decent
    # overlap, so the answer can be a little fuller without ever citing a
    # chunk that isn't actually the source of the returned text.
    primary_chunk_id = ranked_chunks[0]["chunk_id"]
    primary_sentences = [
        c for c in candidates
        if c["chunk_id"] == primary_chunk_id and c["score"] >= max(0.34, top_score * 0.5)
    ]
    covered = set()
    for c in primary_sentences:
        covered |= c["matched"]
    selected_chunk_ids = [primary_chunk_id]

    # Only reach for a second chunk if the best single chunk clearly doesn't
    # cover the question on its own, AND a second chunk is independently a
    # strong (not incidental) match.
    if top_score < 0.5 and len(ranked_chunks) > 1:
        second = ranked_chunks[1]
        new_info = second["matched"] - covered
        if second["score"] >= 0.34 and new_info:
            selected_chunk_ids.append(second["chunk_id"])
            covered |= second["matched"]
            second_sentences = [
                c for c in candidates
                if c["chunk_id"] == second["chunk_id"] and c["score"] >= max(0.34, second["score"] * 0.5)
            ]
            primary_sentences += second_sentences

    # Preserve original sentence order within each cited chunk for readability
    order_index = {(c["chunk_id"], c["sentence"]): i for i, c in enumerate(candidates)}
    primary_sentences = sorted(
        {(c["chunk_id"], c["sentence"]) for c in primary_sentences},
        key=lambda k: order_index[k]
    )

    answer_text = " ".join(sent for _, sent in primary_sentences)
    citations = list(dict.fromkeys(selected_chunk_ids))  # ordered, deduped

    coverage_ratio = sum(qweights[w] for w in covered) / total_weight
    confidence = round(min(0.99, 0.5 + 0.49 * coverage_ratio), 2)

    return respond({
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
        "answerable": True,
    })
