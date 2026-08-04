import streamlit as st
import joblib
from sentence_transformers import SentenceTransformer
from model import predict_hallucination

st.set_page_config(page_title="AI Hallucination Detector", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("random_forest_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
    threshold = joblib.load("threshold.joblib")
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return model, tfidf, threshold, embedding_model

model, tfidf, threshold, embedding_model = load_assets()

# Inputs
doc_input = st.text_area("Source Document", height=200, placeholder="Paste original source text here...")
claim_input = st.text_area("Claim to Verify", height=100, placeholder="Paste generated claim here...")

if st.button("Analyze Claim"):
    if not doc_input.strip() or not claim_input.strip():
            st.warning("Please provide both a source document and a claim.")
    else:
        with st.spinner("Analyzing claim against source document..."):
            res = predict_hallucination(
                model=model,
                tfidf=tfidf,
                document_text=doc_input,
                claim_text=claim_input,
                embedding_model=embedding_model,
                threshold=threshold
            )

        st.divider()
        if res["output"] == "Supported":
            st.success(f"### Result: {res['output']}")
        else:
            st.error(f"### Result: {res['output']}")

        st.metric(label="Groundedness Probability", value=f"{res['probability']:.2%}")
        st.caption(f"Decision Threshold: {threshold:.2f}")

   