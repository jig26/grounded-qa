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

# ---------------- Models ---------------- #

class Chunk(BaseModel):
    chunk_id: str
    text: str

class Query(BaseModel):
    question: str
    chunks: List[Chunk]

# ---------------- Helpers ---------------- #

STOPWORDS = {
    "what","which","when","where","who","why","how",
    "is","are","was","were","do","does","did",
    "the","a","an","of","to","in","on","for","and",
    "or","with","by","from","as","at","be","this",
    "that","these","those"
}

def tokenize(text: str):
    return re.findall(r"\w+", text.lower())

# ---------------- Routes ---------------- #

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/grounded-answer")
def grounded_answer(req: Query):

    if not req.question.strip():
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    if len(req.chunks) == 0:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.0,
            "answerable": False
        }

    qwords = set(tokenize(req.question)) - STOPWORDS

    best_chunk = None
    best_score = 0

    for chunk in req.chunks:
        words = set(tokenize(chunk.text))
        score = len(qwords & words)

        if score > best_score:
            best_score = score
            best_chunk = chunk

    if best_score < 2:
        return {
            "answer": "I don't know",
            "citations": [],
            "confidence": 0.2,
            "answerable": False
        }

    confidence = round(
        min(0.99, 0.5 + best_score / max(len(qwords), 1)),
        2
    )

    return {
        "answer": best_chunk.text,
        "citations": [best_chunk.chunk_id],
        "confidence": confidence,
        "answerable": True
    }
