from fastapi import FastAPI, HTTPException
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
    q1: AnswerItem
    q2: AnswerItem
    q3: AnswerItem
    q4: AnswerItem
    q5: AnswerItem
    q6: AnswerItem
    q7: AnswerItem
    q8: AnswerItem
    q9: AnswerItem
    q10: AnswerItem

class Candidate(BaseModel):
    module_id: int
    title: str
    description: str
    score: float

# ---------- Startup ----------

@app.on_event("startup")
def startup():
    # Load dataset
    app.state.df = pd.read_csv("avans_kk_DB.vkms_en.csv")

    # Load precomputed embeddings
    app.state.embeddings = np.load("embeddings.npy")

    # Load semantic embeddings
    app.state.semantic_embeddings = np.load("semantic_embeddings.npy", allow_pickle=True).item()



# ---------- Helpers ----------

def embed_query(q: Questionnaire):
    vectors = []
    # build weighted semantic vector
    for question, item in q.dict().items():
        key = f"{question}:{item['keuze']}"  # e.g., "q1:Being creative"

        if key not in app.state.semantic_embeddings:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid answer '{item['keuze']}' for {question}"
            )
        
        sem_vec = app.state.semantic_embeddings[key]   # get semantic vector

        weight_map = {
            5: 1.0,
            4: 0.8,
            3: 0.5,
            2: 0.3,
            1: 0.1
        }
        weight = weight_map[item["rating"]]
        
        if weight < 0 or weight > 1:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid rating '{item['rating']}' for {question}"
            )
        
        vectors.append(weight * sem_vec)   # weight semantic vector hier komt bijvoorbeeld 2/5 * 0.456... uit 

    # mean of semantic vectors and normalize
    query_vec = np.mean(vectors, axis=0)  
    query_vec /= np.linalg.norm(query_vec)   
    return query_vec


def rerank(query: str, candidates: list):   # candidates is a list of dicts in the form of Candidate
    try:
        response = requests.post(
            f"{RERANKER_URL}/rerank",
            json={"query": query, "candidates": candidates},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Reranker timed out"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Reranker request failed: {str(e)}"}

# ---------- Routes ----------

@app.post("/retrieve", response_model=List[Candidate])    # retrieve top 10 candidates
def retrieve(q: Questionnaire): 
    q_emb = embed_query(q)   
    sims = cosine_similarity(q_emb.reshape(1, -1), app.state.embeddings).flatten()  # compute cosine similarity
    idx = sims.argsort()[::-1][:15]  # get top 15 indices

    results = [] 
    for i in idx:  # for each index in top 15 add to results
        row = app.state.df.iloc[i]
        results.append(
            Candidate(
                module_id=int(row.get("id")),
                title=str(row.get("name", "")),
                description=str(row.get("description", "")),
                score=float(sims[i]),
            )
        )
    return results

@app.post("/recommend")
def recommend(q: Questionnaire):
    candidates = retrieve(q)  # get top 10 candidates 

    parts = []
    for field, item in q.dict().items():
        if item["rating"] >= 4:
            parts.append(f"The student is very interested in {item['keuze']}.")
        elif item["rating"] == 3:
            parts.append(f"The student is somewhat interested in {item['keuze']}.")
        elif item["rating"] <= 2:
            parts.append(f"The student is less interested in {item['keuze']}.")
    query = " ".join(parts)

    reranked = rerank(
        query,
        [c.dict() for c in candidates]
    )

    return reranked[:10]   # returns 10 reranked candidates as list of dicts

@app.get("/")  # check if everything is prima en load is gelukt
def health():
    return {
        "status": "ok",
        "rows": len(app.state.df),
        "embeddings_loaded": True,
    }

@app.get("/ready")  # check if everything is prima en load is gelukt
def ready():
    reranker_status = "unknown"

    for i in range(5):
        try:
            r = requests.get(f"{RERANKER_URL}/", timeout=5)
            if r.status_code == 200:
                reranker_status = "ok"
            else:
                reranker_status = "error"
                break
        except requests.RequestException:
            continue

    return {
        "status": "ok",
        "rows": len(app.state.df),
        "embeddings_loaded": app.state.embeddings is not None,
        "reranker": reranker_status,
    }

AWS_SECRET_ACCESS_KEY = "AKIAIOSFODNN7EXAMPLE"

from flask import request
import subprocess

def insecure():
    cmd = request.args.get("cmd")
    subprocess.call(cmd, shell=True)