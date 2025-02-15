import streamlit as st
import os
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import nltk

# ✅ NLTK Data Directory and Downloads (FIXED)
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)

@st.cache_resource
def load_models():
    """Load AI models for sentiment analysis, toxicity detection, and rewriting."""
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

# ✅ Load models
sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model = load_models()

if None in [sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model]:  # ✅ Stop execution if models fail
    st.error("Models failed to load. Please refresh the app.")
    st.stop()

def analyze_text(text):
    """Use AI to analyze sentiment and toxicity of each sentence."""
    sentences = sent_tokenize(text)
    results = []
    for sent in sentences:
        sentiment_result = sentiment_analyzer(sent)[0]
        toxicity_result = toxicity_analyzer(sent)[0]
        results.append((sent, sentiment_result, toxicity_result))
    return results

def rewrite_sentence(sentence):
    """Use AI to rewrite a single sentence into a neutral and professional tone."""
    try:
        input_prompt = f"Rewrite this legal sentence in a neutral and professional tone: {sentence}"
        inputs = tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = rewrite_model.generate(
            inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, pad_token_id=tokenizer.pad_token_id
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error rewriting sentence: {e}")
        return sentence  # Return original sentence if rewrite fails

def rewrite_text(text):
    """Use AI to generate a full neutral and professional rewrite of the input text."""
    try:
        paragraphs = text.split("\n\n")  # ✅ Chunking by paragraphs
        rewritten_paragraphs = []
        for paragraph in paragraphs:
            input_prompt = f"Rewrite the following legal paragraph in a professional and neutral tone: {paragraph}"
            inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
            
            # ✅ Adjust `max_length` dynamically to prevent truncation
            max_len = min(len(paragraph) + 50, 512)
            
            summary_ids = rewrite_model.generate(
                inputs.input_ids, max_length=max_len, min_length=100, length_penalty=2.0, num_beams=4, pad_token_id=tokenizer.pad_token_id
            )
            rewritten_paragraph = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            rewritten_paragraphs.append(rewritten_paragraph)
        return "\n\n".join(rewritten_paragraphs)
    except Exception as e:
        st.error(f"Error rewriting text: {e}")
        return "An error occurred during rewriting."

# ✅ Streamlit UI remains the same
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze & Rewrite"):
    if user_text:
        analysis_results = analyze_text(user_text)
        rewritten_text = rewrite_text(user_text)

        st.markdown("## 🧐 AI Analysis & Sentence-by-Sentence Rewriting")
        for original_sentence, sentiment_result, toxicity_result in analysis_results:
            rewritten_sentence = rewrite_sentence(original_sentence)

            if sentiment_result["label"] == "NEGATIVE" or toxicity_result["label"] in ["toxic", "insult", "threat", "identity_hate"]:
                st.markdown("### 🚨 Issue Identified")
                st.write(f"**Before:** {original_sentence}")
                st.write(f"**Sentiment:** {sentiment_result['label']} (Confidence: {sentiment_result['score']:.2f})")
                st.write(f"**Toxicity:** {toxicity_result['label']} (Confidence: {toxicity_result['score']:.2f})")

                st.markdown("### ✅ AI-Suggested Rewording")
                st.write(f"**After:** {rewritten_sentence}")
                st.write("---")

        st.markdown("## ✨ Full AI-Rewritten Version")
        st.write(rewritten_text)
