import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-mpnet-base-v2") # paraphrase-MiniLM-L3-v2        multi-qa-MiniLM-L6-cos-v1

# ---- documents ----
df = pd.read_csv("Clean2_VKM_dataset.csv")
doc_texts = df["description"].fillna("").astype(str).tolist()

doc_emb = model.encode(doc_texts, convert_to_numpy=True, batch_size=32)
doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True)
np.save("embeddings.npy", doc_emb)

# ---- semantic phrases ----
SEMANTICS = {
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
    },
}

semantic_vectors = {}

for cat, values in SEMANTICS.items():
    for key, text in values.items():
        vec = model.encode([text], convert_to_numpy=True)[0]
        vec = vec / np.linalg.norm(vec)
        semantic_vectors[f"{cat}:{key}"] = vec

np.save("semantic_embeddings.npy", semantic_vectors)
