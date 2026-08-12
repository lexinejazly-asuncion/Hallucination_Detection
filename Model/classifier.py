import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer

from config.paths import MODEL_PATH, TFIDF_PATH, EMBEDDING_MODEL_NAME
from Model.scripts.preprocess import transform_features, NUMERIC_FEATS

# Created once when Python first imports classifier.py so Streamlit doesn't reload the model/embeddings on every rerun
_model = None
_tfidf = None
_embedding_model = None


def load_artifacts():
    """Load (and cache) the trained model, TF-IDF vectorizer, and embedding model."""
    global _model, _tfidf, _embedding_model

    if _model is None or _tfidf is None:
        if not TFIDF_PATH or not MODEL_PATH:
            raise FileNotFoundError("Model paths are not configured in config/paths.py")
        try:
            _tfidf = joblib.load(TFIDF_PATH)
            _model = joblib.load(MODEL_PATH)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                "Trained model artifacts not found. Run `python main.py` first "
                "to train the model and generate the .joblib files in /artifacts."
            ) from e

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _model, _tfidf, _embedding_model


def predict_hallucination(document_text, claim_text, threshold=0.50):
    """Classify whether `claim_text` is supported by `document_text`.

    Returns a dict with the predicted label and the model's probability
    that the claim is supported.
    """
    model, tfidf, embedding_model = load_artifacts()

    input_df = pd.DataFrame([{"doc": document_text, "claim": claim_text}])
    X_input = transform_features(input_df, tfidf, NUMERIC_FEATS, embedding_model)

    probability = model.predict_proba(X_input)[0, 1]
    predicted_label = 1 if probability >= threshold else 0

    return {
        "document": document_text,
        "claim": claim_text,
        "predicted_label": predicted_label,
        "probability": float(probability),
        "output": "Supported" if predicted_label == 1 else "Hallucinated",
    }
