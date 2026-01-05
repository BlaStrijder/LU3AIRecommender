from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
from sentence_transformers import CrossEncoder
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"      # disable parallelism more stable performance
torch.set_num_threads(1)                            # limit to 1 thread to reduce CPU usage

app = FastAPI()

# ---------- Models ----------

class Candidate(BaseModel):
    module_id: int
    title: str
    description: str
    score: float

class RerankRequest(BaseModel):
    query: str
    candidates: List[Candidate]

# ---------- Startup ----------

@app.on_event("startup")
def startup():
    # Load cross-encoder model
    app.state.cross_encoder = CrossEncoder(
        "cross-encoder/ms-marco-MiniLM-L6-v2"
    )

# ---------- Routes ----------

@app.post("/rerank", response_model=List[Candidate])
def rerank(req: RerankRequest):
    if not req.candidates:      # safety check for empty candidates
        return []

    pairs = [        # create pairs of (query, candidate description)
        (req.query, c.description)   
        for c in req.candidates
    ]

    scores = app.state.cross_encoder.predict(pairs)  # get scores from cross-encoder

    reranked = sorted(     
        zip(req.candidates, scores),  # pair candidates with their scores
        key=lambda x: x[1],           # sort by score
        reverse=True                  # highest score first
    )

    return [
        Candidate(
            module_id=c.module_id,
            title=c.title,
            description=c.description,
            score=c.score, # original score from retriever if change do this float(score)
        )
        for c, score in reranked
    ]

@app.get("/")       # check if everything is prima en load is gelukt
def health(): 
    return {"status": "ok", "model": "cross-encoder"}
