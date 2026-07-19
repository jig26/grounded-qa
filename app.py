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
FRAGMENT_SPLIT_RE = re.compile(r"\s+and\s+", re.IGNORECASE)

MATCH_THRESHOLD = 0.34


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


def split_question_fragments(question: str, entity_tokens):
    """Split a compound question ('X, and who did Y?') into independent
    sub-questions. Each fragment inherits the entity tokens (acronyms /
    proper nouns) found anywhere in the full question, since a later
    fragment usually refers back to the same subject via a pronoun
    ('...and who created it?') rather than repeating the name."""
    raw_parts = [p.strip() for p in FRAGMENT_SPLIT_RE.split(question) if p.strip()]
    if len(raw_parts) < 2:
        return [question]

    fragments = []
    for part in raw_parts:
        part_weights = weighted_keywords(part)
        # Fragment must contribute at least one non-entity content word to
        # count as its own sub-question (otherwise it's likely just part of
        # a noun list like "cranes and forklifts" inside one clause, not a
        # true second question) -- but we still search it either way.
        for ent in entity_tokens:
            part_weights.setdefault(ent, 2.0)
        if part_weights:
            fragments.append(part_weights)

    return fragments if len(fragments) >= 2 else [question]


def build_candidates(qweights, chunks):
    """Score every sentence of every chunk against a given keyword-weight
    map, returning candidates with any overlap at all.

    A sentence that only shares the question's subject entity (e.g. a
    capitalized proper noun / acronym like "FAISS") but none of the
    question's other content words is NOT treated as a real match when
    the question has such content words to begin with. Otherwise any
    sentence that merely mentions the subject -- regardless of whether it
    states the fact being asked about -- can out-score the 0.34 threshold
    purely off the entity's 2x weight, producing a confident answer to a
    question the chunk never actually addresses (e.g. "FAISS is a popular
    vector search library" does not say what language FAISS is written
    in, even though it mentions "FAISS")."""
    qwords = set(qweights.keys())
    entity_toks = {w for w, wt in qweights.items() if wt > 1.0}
    non_entity_toks = qwords - entity_toks
    total_weight = sum(qweights.values()) or 1.0
    candidates = []
    for chunk in chunks:
        chunk_id = getattr(chunk, "chunk_id", None)
        text = getattr(chunk, "text", "") or ""
        if not chunk_id or not text.strip():
            continue
        for sent in split_sentences(text):
            swords = set(tokenize(sent))
            matched = qwords & swords
            if not matched:
                continue
            # Require the sentence to hit at least one non-entity content
            # word when the question has any -- an entity-only overlap
            # means "same topic" but not necessarily "answers the fact
            # asked for".
            if non_entity_toks and not (matched & non_entity_toks):
                continue
            weighted_score = sum(qweights[w] for w in matched)
            candidates.append({
                "chunk_id": chunk_id,
                "sentence": sent,
                "matched": matched,
                "score": weighted_score / total_weight,
            })
    candidates.sort(key=lambda c: (-c["score"], len(c["sentence"])))
    return candidates


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
        logger.info(
            "OUTGOING_RESPONSE for question=%r: %s",
            raw.get("question") if isinstance(raw, dict) else None,
            json.dumps(payload),
        )
        return JSONResponse(content=payload)

    try:
        req = Query(**raw) if isinstance(raw, dict) else Query()
    except Exception:
        return respond(unanswerable(0.0))

    question = (req.question or "").strip()
    chunks = req.chunks or []

    if not question or not chunks:
        return respond(unanswerable(0.0))

    full_qweights = weighted_keywords(question)
    if not full_qweights:
        return respond(unanswerable(0.0))

    entity_tokens = {w for w, wt in full_qweights.items() if wt > 1.0}

    # Try to split into independent sub-questions (e.g. "what year was X
    # released and who developed it?" -> 2 fragments). If the question
    # isn't compound, this just returns [question] and behaves as before.
    fragments = split_question_fragments(question, entity_tokens)
    fragment_weight_maps = (
        fragments if isinstance(fragments[0], dict) else [full_qweights]
    )

    selected = {}   # (chunk_id, sentence) -> candidate dict
    unsatisfied = False

    for frag_weights in fragment_weight_maps:
        candidates = build_candidates(frag_weights, chunks)
        best = candidates[0] if candidates else None

        # Fallback: if the fragment-scoped search comes up short, retry with
        # the full question's keyword set (covers cases where a fragment's
        # own wording is too sparse to search on its own).
        if not best or best["score"] < MATCH_THRESHOLD:
            fallback_candidates = build_candidates(full_qweights, chunks)
            fallback_best = fallback_candidates[0] if fallback_candidates else None
            if fallback_best and fallback_best["score"] >= MATCH_THRESHOLD:
                best = fallback_best
                candidates = fallback_candidates

        if not best or best["score"] < MATCH_THRESHOLD:
            unsatisfied = True
            continue

        key = (best["chunk_id"], best["sentence"])
        if key not in selected or best["score"] > selected[key]["score"]:
            selected[key] = best

        # Also pull in any other sentence from that SAME chunk that clears
        # a strong bar, so a multi-sentence answer within one chunk isn't
        # truncated to a single line.
        for c in candidates:
            if c["chunk_id"] == best["chunk_id"] and c["score"] >= max(MATCH_THRESHOLD, best["score"] * 0.85):
                k2 = (c["chunk_id"], c["sentence"])
                if k2 not in selected:
                    selected[k2] = c

    logger.info(
        "FRAGMENT_RESULTS: fragments=%d selected=%s unsatisfied=%s",
        len(fragment_weight_maps),
        json.dumps([{"chunk_id": cid, "sentence": s[:80]} for cid, s in selected.keys()]),
        unsatisfied,
    )

    if not selected:
        return respond(unanswerable(0.2))

    if unsatisfied and len(fragment_weight_maps) > 1:
        # Part of a compound question had no grounded support anywhere in
        # the provided chunks -- don't claim a confident, complete answer.
        return respond(unanswerable(0.3))

    # Order sentences by (chunk order in request, sentence order within chunk)
    chunk_order = {getattr(c, "chunk_id", None): i for i, c in enumerate(chunks)}
    ordered = sorted(
        selected.values(),
        key=lambda c: (chunk_order.get(c["chunk_id"], 0),)
    )

    answer_text = " ".join(c["sentence"] for c in ordered)
    citations = list(dict.fromkeys(c["chunk_id"] for c in ordered))

    covered = set()
    for c in ordered:
        covered |= c["matched"]
    total_weight = sum(full_qweights.values()) or 1.0
    coverage_ratio = sum(full_qweights.get(w, 0) for w in covered) / total_weight
    coverage_ratio = min(coverage_ratio, 1.0)
    confidence = round(min(0.99, 0.5 + 0.49 * coverage_ratio), 2)

    return respond({
        "answer": answer_text,
        "citations": citations,
        "confidence": confidence,
        "answerable": True,
    })
