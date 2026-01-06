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
    "q1": {  # values
        "Being creative": "creativity innovation imagination design expression",
        "Helping others": "helping people support care empathy social impact",
        "Working independently": "independent work autonomy self-directed focus",
        "Collaborating in a team": "teamwork collaboration communication group work",
        "Learning new things": "learning curiosity growth development exploration",
    },

    "q2": {  # goals
        "Develop my own project": "project development entrepreneurship ownership initiative",
        "Learn a new skill": "skill development learning competence growth",
        "Gain internship or work experience": "practical experience internship professional practice",
        "Get a good grade": "academic performance achievement assessment results",
        "Study in a field that matches my interests": "personal interest alignment motivation engagement",
    },

    "q3": {  # interests
        "Technology and computers": "technology computers software hardware digital systems",
        "Art and design": "art design creativity visual aesthetics",
        "Science and research": "science research analysis experimentation investigation",
        "Languages and communication": "language communication writing presentation interaction",
        "Business and economics": "business economics management finance organization",
    },

    "q4": {  # working_style
        "Alone": "independent working focus self-paced",
        "In a small team": "small team collaboration cooperation",
        "With lots of guidance": "structured guidance supervision support",
        "Creatively and freely": "creative freedom open-ended exploration",
        "Following clear rules": "structured rules procedures guidelines",
    },

    "q5": {  # learning_style
        "By doing and practicing": "hands-on practice experiential learning",
        "By reading and understanding theory": "theoretical learning reading comprehension",
        "By listening to explanations": "listening instruction lectures explanation",
        "By experimenting": "experimentation trial-and-error discovery",
        "By learning together with others": "collaborative learning peer learning group study",
    },

    "q6": {  # skills
        "Problem-solving": "problem solving analytical thinking reasoning",
        "Communicating with others": "communication interpersonal skills presentation",
        "Project management": "project management planning coordination leadership",
        "Creative design": "creative design ideation visual thinking",
        "Technical expertise": "technical skills engineering implementation",
    },

    "q7": {  # motivation
        "Discovering new things independently": "self-discovery curiosity independent exploration",
        "Building something big with others": "collaborative creation teamwork large projects",
        "Seeing practical results": "practical outcomes real-world impact application",
        "Solving challenges and puzzles": "problem solving challenges puzzles logic",
        "Learning new knowledge and theory": "theoretical knowledge learning understanding",
    },

    "q8": {  # environment
        "Quiet environment without distractions": "quiet focused environment concentration",
        "Creative, flexible workspace": "creative flexible workspace open environment",
        "Team environment with lots of collaboration": "collaborative team environment interaction",
        "Structured environment with clear rules": "structured environment clear expectations",
        "Online and digital learning": "online digital learning remote tools",
    },

    "q9": {  # tech_tools
        "Programming and software development": "programming software development coding",
        "Data analysis and AI": "data analysis artificial intelligence machine learning",
        "Graphic design and UX/UI": "graphic design ux ui user experience",
        "Scientific research tools": "scientific tools research instrumentation",
        "Business and management software": "business software management tools",
    },

    "q10": {  # learning_method
        "By building my own projects": "project-based learning building creating",
        "By following tutorials and guides": "guided learning tutorials instructions",
        "By brainstorming with others": "brainstorming collaboration idea generation",
        "By experimenting and making prototypes": "prototyping experimentation iterative design",
        "By studying theory and making summaries": "theory study summarization reflection",
    },
}


semantic_vectors = {}

for cat, values in SEMANTICS.items():
    for key, text in values.items():
        vec = model.encode([text], convert_to_numpy=True)[0]
        vec = vec / np.linalg.norm(vec)
        semantic_vectors[f"{cat}:{key}"] = vec

np.save("semantic_embeddings.npy", semantic_vectors)
