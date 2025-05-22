import streamlit as st
import whisper
from transformers import pipeline
import matplotlib.pyplot as plt
import tempfile
import os

try:
    import whisper
except ImportError:
    st.error("Whisper is not installed. Please install it using: pip install openai-whisper")
    st.stop()

# --- Login Page ---
def login():
    st.title("🔐 Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == "media@firsteconomy" and password == "FE@123":
            st.session_state.logged_in = True
            st.experimental_rerun()
        else:
            st.error("Invalid credentials. Try again.")

# --- Audio Processing Page ---
def main_app():
    st.title("🎙️ Audio Sentiment & Tone Analyzer")

    uploaded_file = st.file_uploader("Upload a .wav audio file", type=["wav"])
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        # Load Whisper model and transcribe
        st.info("Loading Whisper model...")
        model = whisper.load_model("small")
        result = model.transcribe(tmp_path)
        transcription = result["text"]
        st.subheader("📝 Transcription")
        st.write(transcription)

        # Sentiment Analysis (1-5 stars)
        sentiment_pipe = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        sentiment_result = sentiment_pipe(transcription)
        st.subheader("📊 Basic Sentiment (1-5 Stars)")
        st.write(sentiment_result)

        # Tone and Behavior Classification
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        labels = [
            "rude", "polite", "friendly", "angry", "sarcastic", "casual", "formal",
            "aggressive", "respectful", "smooth", "awkward", "relaxed", "excited", "tense"
        ]

        classification = classifier(transcription, candidate_labels=labels, multi_label=True)
        top_labels = sorted(zip(classification['labels'], classification['scores']), key=lambda x: x[1], reverse=True)

        st.subheader("🎭 Tone & Behavior Classification")
        for label, score in top_labels:
            st.write(f"- {label.capitalize()}: {round(score*100, 1)}%")

        # Bar graph
        st.subheader("📈 Classification Score Bar Graph")
        labels, scores = zip(*top_labels)
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], [s*100 for s in scores[::-1]], color='skyblue')
        ax.set_xlabel("Score (%)")
        ax.set_title("Tone & Behavior Classification")
        st.pyplot(fig)

        # Cleanup
        os.remove(tmp_path)

# --- App Execution ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if st.session_state.logged_in:
    main_app()
else:
    login()
