import streamlit as st
import os
import torch
import openai
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

# ✅ Load sentence tokenizer using Hugging Face (Replaces NLTK)
sentence_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# ✅ Load AI Models
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

sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model = load_models()

if None in [sentiment_analyzer, toxicity_analyzer, tokenizer, rewrite_model]:
    st.error("Models failed to load. Please refresh the app.")
    st.stop()

# ✅ Alternative to NLTK: Sentence Splitting with Hugging Face
def split_into_sentences(text):
    """Use a simple sentence-splitting method as a fallback."""
    sentences = text.split('. ')  
    return sentences

def analyze_text(text):
    """Use AI to analyze sentiment and toxicity of each sentence."""
    sentences = split_into_sentences(text)
    results = []
    for sent in sentences:
        sentiment_result = sentiment_analyzer(sent)[0]
        toxicity_result = toxicity_analyzer(sent)[0]
        results.append((sent, sentiment_result, toxicity_result))
    return results

def rewrite_sentence(sentence):
    """Use AI to rewrite a single sentence in a neutral and professional tone."""
    try:
        input_prompt = (
            f"Rewrite the following legal sentence in a professional, neutral, and respectful tone, "
            f"removing emotionally charged language while maintaining legal clarity:\n\n"
            f'"{sentence}"'
        )
        inputs = tokenizer(input_prompt, return_tensors="pt", max_length=512, truncation=True)
        summary_ids = rewrite_model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, pad_token_id=tokenizer.pad_token_id)
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        st.error(f"Error rewriting sentence: {e}")
        return sentence  # Return original sentence if rewrite fails

def rewrite_text(text):
    """Use AI to rewrite the full text into a professional, neutral legal version."""
    try:
        input_prompt = (
            f"Rewrite the following legal statement in a neutral, professional, and respectful tone, "
            f"removing emotionally charged words while preserving the factual and legal argument:\n\n"
            f"{text}"
        )
        inputs = tokenizer(input_prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = rewrite_model.generate(
            inputs.input_ids, max_length=400, min_length=200, length_penalty=2.0, num_beams=4, pad_token_id=tokenizer.pad_token_id
        )
        rewritten_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return rewritten_text
    except Exception as e:
        st.error(f"Error rewriting text: {e}")
        return "An error occurred during rewriting."

# ✅ Use GPT-4 for More Advanced Rewriting
def rewrite_text_gpt4(text):
    """Use OpenAI's GPT-4 to rewrite the text in a neutral, professional tone."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in legal writing and diplomacy."},
                {"role": "user", "content": f"Rewrite the following legal complaint to be professional, neutral, and persuasive:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"Error rewriting text with GPT-4: {e}")
        return "An error occurred during rewriting."

# ✅ Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

# 🔹 Add a dropdown to choose model type
model_choice = st.radio("Choose rewriting model:", ["Hugging Face (BART)", "GPT-4 (OpenAI)"])

user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze & Rewrite"):
    if user_text:
        analysis_results = analyze_text(user_text)
        
        # Choose the rewriting model
        if model_choice == "GPT-4 (OpenAI)":
            rewritten_text = rewrite_text_gpt4(user_text)
        else:
            rewritten_text = rewrite_text(user_text)

        # ✅ Display AI-rewritten version
        st.markdown("## ✨ Full AI-Rewritten Version")
        st.write(rewritten_text)

