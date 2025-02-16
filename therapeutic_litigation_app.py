import streamlit as st
import torch
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import re
import os

# ✅ Use Local Model or API for Sentiment & Legal Tone Analysis
USE_LOCAL_MODEL = True  
USE_POE_API = True  # Set to True to use Poe's free GPT-3.5 API

# ✅ Model for Sentiment & Tone Analysis (Not Rule-Based)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  # Detects sentiment
LEGAL_TONE_MODEL = "nlpaueb/legal-bert-base-uncased"  # Checks legal tone

# ✅ Poe API Key (For Free GPT-3.5)
POE_API_KEY = os.getenv("POE_API_KEY")  # Set your Poe API Key in Environment Variables

# ✅ Load Models and Tokenizers (Local)
@st.cache_resource
def load_models():
    sentiment_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    sentiment_pipe = pipeline("text-classification", model=sentiment_model, tokenizer=sentiment_tokenizer)

    legal_tone_tokenizer = AutoTokenizer.from_pretrained(LEGAL_TONE_MODEL)
    legal_tone_model = AutoModelForSequenceClassification.from_pretrained(LEGAL_TONE_MODEL)
    legal_tone_pipe = pipeline("text-classification", model=legal_tone_model, tokenizer=legal_tone_tokenizer)

    return sentiment_pipe, legal_tone_pipe

sentiment_pipe, legal_tone_pipe = load_models()

# ✅ Function to Detect Aggressive Language (AI-Based, Not Rule-Based)
def analyze_text(text):
    """Detects negative sentiment, aggressive words, and legal tone issues."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    flagged_sentences = []

    for sentence in sentences:
        sentiment_result = sentiment_pipe(sentence)[0]
        legal_tone_result = legal_tone_pipe(sentence)[0]

        if sentiment_result["label"] in ["negative", "toxic", "hate"] or legal_tone_result["label"] != "neutral":
            flagged_sentences.append((sentence, sentiment_result["label"], sentiment_result["score"], legal_tone_result["label"], legal_tone_result["score"]))

    return flagged_sentences

# ✅ Function to Highlight Aggressive Words (AI-Based)
def highlight_text(text):
    """Highlights words flagged by AI as aggressive or overly emotional."""
    words = text.split()
    flagged_words = set()

    for word in words:
        sentiment = sentiment_pipe(word)[0]
        if sentiment["label"] in ["negative", "toxic", "hate"]:
            flagged_words.add(word.lower())

    highlighted_text = " ".join([f'**🔴 {word} 🔴**' if word.lower() in flagged_words else word for word in words])
    return highlighted_text

# ✅ Function to Rewrite Text Using Free GPT-3.5 (Poe API)
def rewrite_text_gpt(text):
    """Uses Poe's Free GPT-3.5 Turbo API to rewrite text in a professional, neutral tone."""
    if not POE_API_KEY:
        return "⚠️ Poe API Key is missing. Please set it as an environment variable."

    try:
        API_URL = "https://api.poe.com/v1/chat"
        headers = {"Authorization": f"Bearer {POE_API_KEY}"}
        payload = {"bot": "gpt-3.5-turbo", "message": f"Rewrite this legal document in a more professional and neutral tone:\n\n{text}"}

        response = requests.post(API_URL, headers=headers, json=payload)
        if response.status_code == 200:
            return response.json()["text"]
        else:
            return f"Error: {response.json()}"
    except Exception as e:
        return f"Error using Poe GPT-3.5 Turbo: {e}"

# ✅ Streamlit UI
st.title("📝 AI-Powered Litigation Assistant (Now with Free GPT)")
st.write("Identify aggressive language, negative sentiment, and legal tone issues in case submissions.")

# 🔹 Step 1: User Inputs Legal Case Submission
st.markdown("## Step 1: Identify Unacceptable Language & Legal Tone Issues")
user_text = st.text_area("Enter your legal submission for AI analysis:")

if st.button("Analyze Text"):
    if user_text:
        flagged_sentences = analyze_text(user_text)
        highlighted_text = highlight_text(user_text)

        st.markdown("### 🔍 Flagged Sentences & Required Rewriting")
        if flagged_sentences:
            for sent, sentiment_label, sentiment_score, tone_label, tone_score in flagged_sentences:
                st.markdown(f"- **{sent}** _(Sentiment: {sentiment_label} {sentiment_score:.2f}, Legal Tone: {tone_label} {tone_score:.2f})_")
            st.warning("⚠️ Please rewrite the above sentences in a more professional and neutral tone before submission.")
        else:
            st.success("✅ No aggressive or unacceptable language detected.")

        st.markdown("### ✏️ Highlighted Aggressive Words")
        st.write(highlighted_text)

        st.markdown("### ✏️ Your Turn: Rewrite the Flagged Sentences")
        st.write("You can manually rewrite the flagged sentences below. If you need AI assistance, proceed to Step 2.")
    else:
        st.warning("Please enter some text to analyze.")

# 🔹 Step 2 (Optional): AI-Powered Rewriting with Free GPT
st.markdown("## Step 2: AI-Powered Rewriting (Optional)")
use_ai_rewriting = st.radio("Would you like AI to rewrite the text for you?", ["No", "Yes"])

if use_ai_rewriting == "Yes":
    if user_text:
        rewritten_text = rewrite_text_gpt(user_text)
        st.markdown("### ✅ AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using AI to rewrite.")
