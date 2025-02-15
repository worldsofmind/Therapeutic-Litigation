import streamlit as st
import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

# NLTK Data Directory and Downloads (CRITICAL)
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab/english/')  # Download specific English data
except LookupError:
    nltk.download('punkt_tab/english', download_dir=NLTK_DATA_DIR, quiet=True)

nltk.data.path.append(NLTK_DATA_DIR)  # Set NLTK path (do this ONCE)

@st.cache_resource
def load_models():
    try:
        sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
        toxicity_analyzer = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=0 if torch.cuda.is_available() else -1)
        rewrite_model_name = "facebook/bart-large-cnn"
        tokenizer = AutoTokenizer.from_pretrained(rewrite_model_name)
        rewrite_model = AutoModelForSeq2SeqLM.from_pretrained(rewrite_model_name)
        return sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None, None

sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model = load_models()

if not all([sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model]):  # Check if models loaded
    st.stop()  # Stop execution if models failed to load

def analyze_text(text):
    sentences = sent_tokenize(text)  # Should work now!
    results = []
    for sent in sentences:
        sentiment_result = sentiment_analyzer(sent)[0]
        toxicity_result = toxicity_analyzer(sent)[0]
        results.append((sent, sentiment_result, toxicity_result))
    return results

def rewrite_sentence(sentence):
    try:  # Error handling for rewrite_sentence
        input_prompt = f"Rewrite this legal sentence in a neutral and professional tone: {sentence}"
        inputs = tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = rewrite_model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error rewriting sentence: {e}")
        return sentence  # Return original sentence if rewrite fails

def rewrite_text(text):
    try:
        paragraphs = text.split("\n\n")  # Chunking by paragraphs
        rewritten_paragraphs = []
        for paragraph in paragraphs:
            input_prompt = f"Rewrite the following legal paragraph in a professional and neutral tone: {paragraph}"
            inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
            summary_ids = rewrite_model.generate(inputs.input_ids, max_length=300, min_length=150, length_penalty=2.0, num_beams=4)
            rewritten_paragraph = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            rewritten_paragraphs.append(rewritten_paragraph)
        return "\n\n".join(rewritten_paragraphs)
    except Exception as e:
        st.error(f"Error rewriting text: {e}")
        return "An error occurred during rewriting."

# Streamlit UI (Rest of your Streamlit code remains the same)
# ... (no changes needed in the UI part)
