from typing import Dict, Tuple, List, Union
import logging
from fastapi import HTTPException
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, CrossEncoder
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


class AnswerItem(BaseModel):
    keuze: str
    rating: int = 3  # default rating if user doesn't provide one


class Questionnaire(BaseModel):
    waardes: AnswerItem
    activiteiten: AnswerItem
    interesse: AnswerItem


class ModuleRecommendation(BaseModel):
    title: str
    description: str
    score: float
    module_id: int


app = FastAPI()


def initializeModel():
    app.state.model = SentenceTransformer("all-mpnet-base-v2")
    app.state.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    app.state.df = pd.read_csv("Clean2_VKM_dataset.csv")

    # Weights for combining similarity across fields
    app.state.WEIGHTS = {
        "description": 0.3,
        "theme_tags": 0.5,
        "module_tags": 0.2,
    }
    
    # Small mapping from questionnaire choices to representative text
    app.state.QUESTION_SEMANTICS = {
        "waardes": {
            "veiligheid": "veiligheid risico bescherming",
            "rechtvaardigheid": "recht eerlijkheid ethiek",
            "plezier": "creativiteit enthousiasme motivatie",
        },

        "activiteiten": {
            "programmeren": "programmeren software ontwikkeling",
            "mensen helpen": "zorg ondersteuning hulp",
            "analyseren": "analyse data onderzoek",
        },

        "interesse": {
            "technologie": "technologie ai software",
            "maatschappij": "maatschappij cultuur communicatie",
            "natuur": "natuur ecologie duurzaamheid",
            "creativiteit": "creativiteit design kunst innovatie",
        }
    }

    col_embeddings = {}
    for col in ["description", "theme_tags", "module_tags"]:
        texts = app.state.df[col].fillna("").astype(str).tolist()
        col_embeddings[col] = normalize(
            app.state.model.encode(texts, convert_to_numpy=True)
        )
    app.state.col_embeddings = col_embeddings
    print({col: col_embeddings[col].shape for col in col_embeddings})


def normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

def questionnaire_to_query_embeddings(answers):
    model = app.state.model
    QUESTION_SEMANTICS = app.state.QUESTION_SEMANTICS

    q_embs = {col: [] for col in ["description", "theme_tags", "module_tags"]}

    for question, answer in answers.items():

        if isinstance(answer, tuple):
            answer_text, weight = answer        
        else:
            answer_text = answer
            weight = 3                        

        weight = weight / 5.0
        
        # validate if answer exists in semantics mapping
        if answer_text not in QUESTION_SEMANTICS[question]:
            raise HTTPException(400, f"Invalid answer: {answer_text}")

        text = QUESTION_SEMANTICS[question][answer_text]

        emb = normalize(model.encode([text], convert_to_numpy=True))
    
        emb = emb * weight

        # doe embedding voor elke column tegenover elke module
        for col in q_embs.keys():
            q_embs[col].append(emb)
        # je krijgt dan (3, 768)
    # voeg het samen en neem het gemiddelde
    for col in q_embs:
        if len(q_embs[col]) > 0:
            stacked = np.vstack(q_embs[col])  
            mean_emb = stacked.mean(axis=0, keepdims=True)  
            q_embs[col] = normalize(mean_emb)               
        else:
            # Safe zero-vector (prevents NaNs in cosine similarity)
            q_embs[col] = np.zeros((1, 768), dtype=np.float32)
    return q_embs

def multi_field_search(q_embs, top_k=20):
    col_embeddings = app.state.col_embeddings
    WEIGHTS = app.state.WEIGHTS

    sims_total = None
    sims_raw = {}
    sims_weighted = {}

    for col, q_emb in q_embs.items():

        raw = cosine_similarity(q_emb, col_embeddings[col]).flatten()
        weighted = WEIGHTS[col] * raw

        sims_raw[col] = raw
        sims_weighted[col] = weighted

        if sims_total is None:
            sims_total = weighted.copy()
        else:
            sims_total += weighted

    idx = sims_total.argsort()[::-1][:top_k]

    return idx, sims_total[idx], sims_raw, sims_weighted

def rerank_with_cross_encoder(query, df, idx):
    cross_encoder = app.state.cross_encoder

    # Build sentence pairs for the candidates
    pairs = [(query, df.loc[i, "description"]) for i in idx]

    # If no candidates, return empty lists
    if len(idx) == 0:
        return [], []

    # Cross-encoder returns high-accuracy similarity scores
    ce_scores = cross_encoder.predict(pairs)

    # Sort candidates again using CE scores
    reranked = sorted(zip(idx, ce_scores), key=lambda x: x[1], reverse=True)

    new_idx = [x[0] for x in reranked]
    new_scores = [x[1] for x in reranked]
    return new_idx, new_scores


@app.on_event("startup")
async def startup_event():
    """Initialize models and data on startup"""
    initializeModel()

@app.post("/recommendation", response_model=List[ModuleRecommendation])
async def recommendation(q: Questionnaire):
    """Get module recommendations based on questionnaire answers"""
    try:
        # Convert list format to tuple format
        answers = {
            "waardes": (q.waardes.keuze, q.waardes.rating),
            "activiteiten": (q.activiteiten.keuze, q.activiteiten.rating),
            "interesse": (q.interesse.keuze, q.interesse.rating),
        }
        
        # Generate embeddings
        q_embs = questionnaire_to_query_embeddings(answers)
        
        # Multi-field search
        idx, sims_total, sims_raw, sims_weighted = multi_field_search(q_embs, top_k=10)
        
        # Build query string for cross-encoder
            # query_string = " ".join(value for value, rating in answers.values())
        
        query_string = (
            f"De gebruiker waardeert {answers['waardes'][0]}, "
            f"doet graag {answers['activiteiten'][0]}, "
            f"en heeft interesse in {answers['interesse'][0]}."
        )
        
        # Rerank with cross-encoder
        idx, ce_scores = rerank_with_cross_encoder(query_string, app.state.df, idx)
        
        # Return results
        results = []
        for i, score in zip(idx, ce_scores):
            module = app.state.df.iloc[i]
            results.append(ModuleRecommendation(
                description=str(module.get("description", "")),
                score=float(score),
                title=str(module.get("title", "")),
                module_id=int(i)
            ))
        
        return results
        
    except Exception as e:
        logger.error(f"Error in recommendation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "cross_encoder_loaded": app.state.cross_encoder is not None,
        "rows_loaded": len(app.state.df)
    }        