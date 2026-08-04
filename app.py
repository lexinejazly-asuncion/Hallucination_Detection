import streamlit as st

from Model.classifier import predict_hallucination

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍")

st.title("Hallucination Detector")
st.write(
    "Paste a source document and a claim below. The model checks whether "
    "the claim is supported by the document, or whether it's "
    "hallucinated content not grounded in the text."
)

document_text = st.text_area(
    "Document", height=250, placeholder="Paste the source document here..."
)
claim_text = st.text_area(
    "Claim", height=100, placeholder="Paste the claim you want to verify here..."
)

THRESHOLD = 0.50

if st.button("Classify", type="primary"):
    if not document_text.strip() or not claim_text.strip():
        st.warning("Please provide both a document and a claim.")
    else:
        try:
            with st.spinner("Running model..."):
                result = predict_hallucination(document_text, claim_text, threshold=THRESHOLD)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            label = result["output"]
            probability = result["probability"]

            if label == "Supported":
                st.success(f"Claim is likely Supported — probability: {probability:.2%}")
            else:
                st.error(f"Claim is likely Hallucinated — probability: {probability:.2%}")
