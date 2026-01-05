from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests

RERANKER_URL = os.getenv("RERANKER_URL", "http://reranker:8000")

app = FastAPI()

# ---------- Models ----------

class AnswerItem(BaseModel):
    keuze: str
    rating: int = 3  # can be 1–5

class Questionnaire(BaseModel):
    waardes: AnswerItem
    activiteiten: AnswerItem
    interesse: AnswerItem

class Candidate(BaseModel):
    module_id: int
    title: str
    description: str
    score: float

# ---------- Startup ----------

@app.on_event("startup")
def startup():
    # Load dataset
    app.state.df = pd.read_csv("Clean2_VKM_dataset.csv")

    # Load precomputed embeddings
    app.state.embeddings = np.load("embeddings.npy")

    # Load semantic embeddings
    app.state.semantic_embeddings = np.load("semantic_embeddings.npy", allow_pickle=True).item()



# ---------- Helpers ----------

def embed_query(q: Questionnaire):
    vectors = []
    # build weighted semantic vector
    for category, item in q.dict().items():
        sem_vec = app.state.semantic_embeddings[f"{category}:{item['keuze']}"]   # get semantic vector
        weight = item["rating"] / 5.0   # normalize to [0.2 ,1]
        vectors.append(weight * sem_vec)   # weight semantic vector hier komt bijvoorbeeld 2/5 * 0.456... uit 

    # mean of semantic vectors and normalize
    query_vec = np.mean(vectors, axis=0)  
    query_vec /= np.linalg.norm(query_vec)   
    return query_vec


def rerank(query: str, candidates: list):    # candidates is a list of dicts in the form of Candidate
    response = requests.post(
        f"{RERANKER_URL}/rerank",
        json={"query": query, "candidates": candidates},
        timeout=10,
    )
    response.raise_for_status()   # raise error if not 200 OK
    return response.json()

# ---------- Routes ----------

@app.post("/retrieve", response_model=List[Candidate])    # retrieve top 10 candidates
def retrieve(q: Questionnaire): 
    q_emb = embed_query(q)   
    sims = cosine_similarity(q_emb.reshape(1, -1), app.state.embeddings).flatten()  # compute cosine similarity
    idx = sims.argsort()[::-1][:10]  # get top 10 indices

    results = [] 
    for i in idx:  # for each index in top 10 add to results
        row = app.state.df.iloc[i]
        results.append(
            Candidate(
                module_id=int(i),
                title=str(row.get("name", "")),
                description=str(row.get("description", "")),
                score=float(sims[i]),
            )
        )
    return results

@app.post("/recommend")
def recommend(q: Questionnaire):
    candidates = retrieve(q)  # get top 10 candidates 

    query = (                                          # construct query string for reranker
        f"De gebruiker waardeert {q.waardes.keuze}, "
        f"doet graag {q.activiteiten.keuze}, "
        f"en heeft interesse in {q.interesse.keuze}."
    )

    return rerank(query, [c.dict() for c in candidates])   # returns query and candidates as list of dicts

@app.get("/")  # check if everything is prima en load is gelukt
def health():
    return {
        "status": "ok",
        "rows": len(app.state.df),
        "embeddings_loaded": True,
    }

@app.get("/ready")  # check if everything is prima en load is gelukt
def health():
    reranker_status = "unknown"

    try:
        r = requests.get(
            f"{RERANKER_URL}/",
            timeout=30
        )
        if r.status_code == 200:
            reranker_status = "ok"
        else:
            reranker_status = "error"
    except requests.RequestException:
        reranker_status = "down"

    return {
        "status": "ok",
        "rows": len(app.state.df),
        "embeddings_loaded": app.state.embeddings is not None,
        "reranker": reranker_status,
    }