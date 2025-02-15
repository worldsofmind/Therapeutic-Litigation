import streamlit as st
import os
import torch
import nltk
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize

# ✅ Ensure NLTK resources are available
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)

@st.cache_resource
def load_models():
    """Load AI models for sentiment, toxicity, and text rewriting."""
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
    sentences = sent_tokenize(text)
    results = []

    for sent in sentences:
        sentiment_result = sentiment_analyzer(sent)[0]
        toxicity_result = toxicity_analyzer(sent)[0]
        results.append((sent, sentiment_result, toxicity_result))

    return results

def rewrite_text(text):
    """Use AI to generate a neutral and professional version of the input text."""
    input_prompt = f"Rewrite the following legal text in a professional and neutral tone: {text}"
    inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = rewrite_model.generate(inputs.input_ids, max_length=250, min_length=100, length_penalty=2.0, num_beams=4)
    rewritten_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    return rewritten_text

def rewrite_sentence(sentence):
    """Use AI to rewrite a single sentence into a professional tone."""
    input_prompt = f"Rewrite this sentence in a neutral and professional tone: {sentence}"
    inputs = tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = rewrite_model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ✅ Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

# 🔹 User input through a text box
user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze & Rewrite"):
    if user_text:
        # AI Analysis
        analysis_results = analyze_text(user_text)
        rewritten_text = rewrite_text(user_text)

        # ✅ Display Sentences and Rewording
        st.markdown("## 🧐 Sentence-by-Sentence AI Analysis & Rewriting")

        for original_sentence, sentiment_result, toxicity_result in analysis_results:
            rewritten_sentence = rewrite_sentence(original_sentence)

            # 🔸 Display problematic sentences if flagged by AI
            if sentiment_result["label"] == "NEGATIVE" or toxicity_result["label"] in ["toxic", "insult", "threat", "identity_hate"]:
                st.markdown("### 🚨 Issue Identified")
                st.write(f"**Before:** {original_sentence}")
                st.write(f"**Sentiment:** {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")
                st.write(f"**Toxicity:** {toxicity_result['label']} (Confidence: {toxicity_result['score']:.2f})")

                # 🔹 Suggested Rewrite
                st.markdown("### ✅ Suggested Rewording")
                st.write(f"**After:** {rewritten_sentence}")
                st.write("---")

        # ✅ Display Full AI-Rewritten Text
        st.markdown("## ✨ Full AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter some text to analyze.")
