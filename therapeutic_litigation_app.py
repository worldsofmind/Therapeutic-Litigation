import streamlit as st
import os
import torch
import nltk
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Ensure NLTK resources are available
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)

@st.cache_resource
def load_models():
    """Load AI models for sentiment, toxicity, and text rewording."""
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
    toxicity_analyzer = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=0 if torch.cuda.is_available() else -1)
    rewrite_model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(rewrite_model_name)
    rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model_name)
    return sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model

# ✅ Load AI models
sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model = load_models()

def analyze_text(text):
    """Use AI to analyze sentiment and toxicity of the input text."""
    sentiment_result = sentiment_analyzer(text)[0]
    toxicity_result = toxicity_analyzer(text)[0]
    
    return sentiment_result, toxicity_result

def rewrite_text(text):
    """Use AI to generate a neutral and professional version of the input text."""
    input_prompt = f"Rewrite the following legal text in a professional and neutral tone: {text}"
    inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = rewrite_model.generate(inputs.input_ids, max_length=250, min_length=100, length_penalty=2.0, num_beams=4)
    rewritten_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return rewritten_text

# ✅ Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

# 🔹 User input through a text box
user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze & Rewrite"):
    if user_text:
        # AI Analysis
        sentiment_result, toxicity_result = analyze_text(user_text)
        rewritten_text = rewrite_text(user_text)
        
        # ✅ Display Results
        st.markdown("### Sentiment Analysis")
        st.write(f"**Label:** {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")

        st.markdown("### Toxicity Analysis")
        st.write(f"**Label:** {toxicity_result['label']} (Confidence: {toxicity_result['score']:.2f})")

        st.markdown("### Suggested Rewording")
        st.write(rewritten_text)
    else:
        st.warning("Please enter some text to analyze.")
