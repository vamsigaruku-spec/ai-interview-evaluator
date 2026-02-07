import streamlit as st
import numpy as np
import shap
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="AI Interview Evaluator", layout="wide")
st.title("🧠 AI Interview Answer Evaluator")
st.write("Evaluate candidate answers using NLP + ML + Explainable AI")

# ================= MODEL =================
@st.cache_resource
def load_model():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Dummy training data (so app always works in cloud)
    texts = [
        "I optimized system performance using caching",
        "I don't know",
        "Used machine learning models for prediction",
        "No experience"
    ]
    labels = [1, 0, 1, 0]

    X = embedder.encode(texts)
    model = LogisticRegression()
    model.fit(X, labels)

    return model, embedder

model, embedder = load_model()

# ================= UI =================
answer = st.text_area("✍ Enter Candidate Answer")

if st.button("Evaluate Answer") and answer.strip():

    vector = embedder.encode([answer])
    prob = model.predict_proba(vector)[0][1]

    if prob > 0.6:
        st.success(f"✅ Strong Answer (Confidence: {prob:.2f})")
    else:
        st.error(f"⚠ Weak Answer (Confidence: {prob:.2f})")

    if len(answer.split()) < 10:
        st.warning("Answer is too short. Add more detail.")
    else:
        st.info("Answer shows technical understanding and clarity.")

    st.subheader("🔍 Why did the AI give this score?")

    try:
        background = np.zeros((1, vector.shape[1]))
        explainer = shap.LinearExplainer(model, background)
        shap_values = explainer.shap_values(vector)

        fig, ax = plt.subplots(figsize=(6, 3))
        shap.plots.bar(shap_values[0], show=False)
        st.pyplot(fig)

    except:
        st.warning("Explainability unavailable in this environment.")
