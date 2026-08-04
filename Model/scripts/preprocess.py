import re
import numpy as np
import pandas as pd

STOPWORDS = set("a an the and or but if that this these those of to as be been being it its so such".split())

# The feature columns that get fed into the logistic regression model.
NUMERIC_FEATS = [
    "coverage",
    "bigram_cov",
    "num_grounded",
    "has_novel_num",
    "claim_len",
    "n_content_words",
    "lexical_sim",
    "semantic_sim",
]


def normalize(text):
    """Normalize raw text inputs."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s.,!?'-]", " ", text)
    return text.strip()


def overlap_features(row):
    """Calculate overlap, coverage, and length metrics between document and claim."""
    doc_toks = row["doc_clean"].split()
    claim_toks = row["claim_clean"].split()
    doc_set, claim_set = set(doc_toks), set(claim_toks)

    claim_content = claim_set - STOPWORDS
    doc_content = doc_set - STOPWORDS

    coverage = len(claim_content & doc_content) / len(claim_content) if claim_content else 0.0

    union = claim_set | doc_set
    jaccard = len(claim_set & doc_set) / len(union) if union else 0.0

    def bigrams(toks):
        return set(zip(toks, toks[1:]))

    cb, db = bigrams(claim_toks), bigrams(doc_toks)
    bigram_cov = len(cb & db) / len(cb) if cb else 0.0

    claim_nums = set(re.findall(r"\d+\.?\d*", row["claim_clean"]))
    doc_nums = set(re.findall(r"\d+\.?\d*", row["doc_clean"]))

    if claim_nums:
        num_grounded = len(claim_nums & doc_nums) / len(claim_nums)
        has_novel_num = int(len(claim_nums - doc_nums) > 0)
    else:
        num_grounded, has_novel_num = 1.0, 0

    return pd.Series({
        "coverage": coverage,
        "jaccard": jaccard,
        "bigram_cov": bigram_cov,
        "num_grounded": num_grounded,
        "has_novel_num": has_novel_num,
        "claim_len": len(claim_toks),
        "doc_len": len(doc_toks),
        "n_content_words": len(claim_content),
    })


def preprocess_dataframe(df):
    """Add clean text columns and lexical-overlap feature set to a doc/claim DataFrame."""
    df_copy = df.copy()
    df_copy["doc_clean"] = df_copy["doc"].apply(normalize)
    df_copy["claim_clean"] = df_copy["claim"].apply(normalize)
    feats = df_copy.apply(overlap_features, axis=1)
    return pd.concat([df_copy, feats], axis=1)


def compute_lexical_sim(df, tfidf):
    """Compute lexical TF-IDF cosine similarity between document and claim.

    Assumes df already has 'doc_clean' and 'claim_clean' columns
    (i.e. df has already been through preprocess_dataframe).
    """
    d = tfidf.transform(df["doc_clean"])
    c = tfidf.transform(df["claim_clean"])
    return np.asarray(d.multiply(c).sum(axis=1)).flatten()


def compute_semantic_sim(df, embedding_model):
    """Compute semantic embedding cosine similarity using a SentenceTransformer."""
    d = embedding_model.encode(df["doc_clean"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    c = embedding_model.encode(df["claim_clean"].tolist(), convert_to_numpy=True, show_progress_bar=False)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-9)
    c = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-9)
    return np.sum(d * c, axis=1)


def transform_features(df, tfidf, numeric_feats, embedding_model):
    """End-to-end: raw doc/claim DataFrame -> feature matrix ready for the model.

    Used identically at training time (on the full training set) and at
    inference time (on a single-row DataFrame built from user input).
    """
    processed = preprocess_dataframe(df)
    processed["lexical_sim"] = compute_lexical_sim(processed, tfidf)
    processed["semantic_sim"] = compute_semantic_sim(processed, embedding_model)
    return processed[numeric_feats]
