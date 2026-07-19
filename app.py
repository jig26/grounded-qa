from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import re

app = FastAPI(title="Grounded QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Chunk(BaseModel):
    chunk_id: str
    text: str


class Query(BaseModel):
    question: str
    chunks: List[Chunk]


def tokenize(text: str):
    return set(re.findall(r"\w+", text.lower()))


@app.get("/")
def root():
    return {"status": "running"}


@app.post("/grounded-answer")
def grounded_answer(req: Query):

    if not req.question.strip() or len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False,
        }

    qwords = tokenize(req.question)

    scored = []

    for chunk in req.chunks:
        words = tokenize(chunk.text)
        overlap = len(qwords & words)

        if overlap > 0:
            scored.append((overlap, chunk))

    if not scored:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False,
        }

    scored.sort(reverse=True, key=lambda x: x[0])

    best_score = scored[0][0]

    citations = []
    answer_parts = []

    for score, chunk in scored:
        if score >= max(1, best_score - 1):
            citations.append(chunk.chunk_id)
            answer_parts.append(chunk.text)

    answer = " ".join(dict.fromkeys(answer_parts))

    confidence = min(
        0.99,
        round(best_score / max(len(qwords), 1) + 0.35, 2),
    )

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "answerable": True,
    }