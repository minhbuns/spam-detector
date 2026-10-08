import joblib
import streamlit as st
from preprocess import clean_text

@st.cache_resource
def load():
    return joblib.load("vectorizer.joblib"), joblib.load("model_lr.joblib")

vectorizer, model = load()

st.title("Spam Detector")
text = st.text_area("Paste an email:", height=200)

if st.button("Check") and text.strip():
    vec = vectorizer.transform([clean_text(text)])
    p = model.predict_proba(vec)[0][1]
    if p >= 0.5:
        st.error(f"Spam ({p:.1%} probability)")
    else:
        st.success(f"Not spam ({1 - p:.1%} confidence)")