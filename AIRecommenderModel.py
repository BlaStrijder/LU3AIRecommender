import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder



model = SentenceTransformer("all-mpnet-base-v2")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

df = pd.read_csv("Clean2_VKM_dataset.csv")

WEIGHTS = {
    "description": 0.3,
    "theme_tags": 0.5,
    "module_tags": 0.2
}
QUESTION_SEMANTICS = {
    "waardes": {
        "veiligheid": "veiligheid risico bescherming",
        "rechtvaardigheid": "recht eerlijkheid ethiek",
        "plezier": "creativiteit enthousiasme motivatie"
    },

    "activiteiten": {
        "programmeren": "programmeren software ontwikkeling",
        "mensen helpen": "zorg ondersteuning hulp",
        "analyseren": "analyse data onderzoek"
    },

    "interesse": {
        "technologie": "technologie ai software",
        "maatschappij": "maatschappij cultuur communicatie",
        "natuur": "natuur ecologie duurzaamheid",
        "creativiteit": "creativiteit design kunst innovatie"
    }
}

def normalize(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)

col_embeddings = {}
for col in ["description", "theme_tags", "module_tags"]:
    texts = df[col].fillna("").astype(str).tolist()
    col_embeddings[col] = normalize(model.encode(texts, convert_to_numpy=True))

print({col: col_embeddings[col].shape for col in col_embeddings})
def questionnaire_to_query_embeddings(answers):
    q_embs = {col: [] for col in ["description", "theme_tags", "module_tags"]}

    for question, answer in answers.items():

        if isinstance(answer, tuple):
            answer_text, weight = answer        
        else:
            answer_text = answer
            weight = 3                        

        weight = weight / 5.0
        
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
            q_embs[col] = np.zeros((1, 768))

    return q_embs

def multi_field_search(q_embs, top_k=20):
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
    # Build sentence pairs for the candidates
    pairs = [(query, df.loc[i, "description"]) for i in idx]

    # Cross-encoder returns high-accuracy similarity scores
    ce_scores = cross_encoder.predict(pairs)

    # Sort candidates again using CE scores
    reranked = sorted(zip(idx, ce_scores), key=lambda x: x[1], reverse=True)

    new_idx = [x[0] for x in reranked]
    new_scores = [x[1] for x in reranked]
    return new_idx, new_scores
q_embs = questionnaire_to_query_embeddings(answers)

idx, sims_total, sims_raw, sims_weighted = multi_field_search(q_embs, top_k=20)

query_string = " ".join(value for value, rating in answers.values())

idx, ce_scores = rerank_with_cross_encoder(
    query=query_string,
    df=df,
    idx=idx
)

df.iloc[idx]
