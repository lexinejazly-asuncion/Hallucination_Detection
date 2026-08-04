"""
model.py
Train the selected production model and export artifacts for deployment.
"""

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer

from preprocess import fit_tfidf_vectorizer, transform_features

RANDOM_STATE = 42
NUMERIC_FEATS = [
    "coverage",
    "bigram_cov",
    "num_grounded",
    "has_novel_num",
    "claim_len",
    "n_content_words",
    "lexical_sim",
    "semantic_sim"
]


def train_and_export_best_model():
    """Train the selected production model and export joblib artifacts."""
    print("Loading HuggingFace LLM-AggreFact dataset...")
    dev = pd.read_parquet("hf://datasets/lytang/LLM-AggreFact/data/dev-00000-of-00001.parquet")

    if "contamination_identifier" in dev.columns:
        dev = dev.drop(columns=["contamination_identifier"])

    print("Fitting text transformers on the full development set...")
    tfidf = fit_tfidf_vectorizer(dev)
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    X_dev = transform_features(dev, tfidf, NUMERIC_FEATS, embedding_model)
    y_dev = dev["label"]

    print("Training the selected Random Forest model...")
    best_rf = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=1,
        n_jobs=-1
    )
    best_rf.fit(X_dev, y_dev)

    print("Exporting model and transformer artifacts...")
    joblib.dump(best_rf, "random_forest_model.joblib")
    joblib.dump(tfidf, "tfidf_vectorizer.joblib")

    print("Artifacts saved:")
    print("random_forest_model.joblib")
    print("tfidf_vectorizer.joblib")


def predict_hallucination(model, tfidf, document_text, claim_text, embedding_model=None, threshold=0.50):
    """Run prediction on raw inputs using saved transformers and the trained model."""
    input_df = pd.DataFrame([{"doc": document_text, "claim": claim_text}])
    X_input = transform_features(input_df, tfidf, NUMERIC_FEATS, embedding_model)

    probability = model.predict_proba(X_input)[0, 1]
    predicted_label = 1 if probability >= threshold else 0

    return {
        "document": document_text,
        "claim": claim_text,
        "predicted_label": predicted_label,
        "probability": float(probability),
        "output": "Supported" if predicted_label == 1 else "Hallucinated"
    }


if __name__ == "__main__":
    train_and_export_best_model()
