import os

# Root of the whole project 
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Where trained artifacts (vectorizer + model) get saved/loaded from
ARTIFACTS_DIR = os.path.join(BASE_DIR, "Artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

TFIDF_PATH = os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.joblib")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "logistic_regression_model.joblib")

# Sentence-transformers model name 
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

DATASET_PATH = "hf://datasets/lytang/LLM-AggreFact/data/dev-00000-of-00001.parquet"
