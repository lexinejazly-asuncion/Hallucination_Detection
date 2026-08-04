import os

import joblib
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.paths import MODEL_PATH, TFIDF_PATH, EMBEDDING_MODEL_NAME
from Model.scripts.preprocess import preprocess_dataframe, transform_features, NUMERIC_FEATS

RANDOM_STATE = 42


def fit_tfidf_vectorizer(train_df):
    """Fit and return a TF-IDF Vectorizer on normalized training documents and claims."""
    train_processed = preprocess_dataframe(train_df)
    corpus = pd.concat([train_processed["doc_clean"], train_processed["claim_clean"]])
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    tfidf.fit(corpus)
    return tfidf


def train_and_initialize_models(train_data, save_artifacts=True):
    """Fit the TF-IDF vectorizer + logistic regression pipeline and export artifacts.

    Returns (fitted_model, fitted_tfidf) so callers (e.g. main.py) can use
    them immediately without re-reading from disk.
    """
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # Fit TF-IDF, then build the full feature matrix (lexical + semantic + overlap feats)
    tfidf = fit_tfidf_vectorizer(train_data)
    X_train = transform_features(train_data, tfidf, NUMERIC_FEATS, embedding_model)
    y_train = train_data["label"]

    best_lr = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                max_iter=1000,
                C=0.01,
                solver="lbfgs",
            ),
        ),
    ])
    best_lr.fit(X_train, y_train)

    if save_artifacts:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(tfidf, TFIDF_PATH)
        joblib.dump(best_lr, MODEL_PATH)
        print(f"Saved TF-IDF vectorizer -> {TFIDF_PATH}")
        print(f"Saved trained model     -> {MODEL_PATH}")

    return best_lr, tfidf
