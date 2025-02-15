import os
import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ✅ Load Mistral 7B (Free & Open-Source Model)
MODEL_NAME = "mistralai/Mistral-7B-Instruct"

@st.cache_resource
def load_mistral():
    """Load Mistral model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

mistral_pipeline = load_mistral()

# ✅ Function to Identify Negative Sentences
def identify_negative_statements(text):
    """Use Mistral to detect negative or aggressive statements."""
    prompt = f"Identify sentences in the following text that contain aggressive, negative, or hostile language:\n\n{text}"
    response = mistral_pipeline(prompt, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

# ✅ Function to Rewrite Negative Statements
def rewrite_text_mistral(text):
    """Use Mistral to rewrite text in a professional, neutral tone."""
    prompt = f"Rewrite the following text in a more professional, neutral, and respectful tone:\n\n{text}"
    response = mistral_pipeline(prompt, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

# ✅ Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant (Free & Open-Source)")
st.write("Ensure legal submissions are neutral and constructive using AI.")

# 🔹 Step 1: User Inputs Text for Analysis
st.markdown("## Step 1: Identify Negative or Aggressive Language")
user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze Text"):
    if user_text:
        analysis_result = identify_negative_statements(user_text)
        st.markdown("### 🔍 Identified Negative Sentences")
        st.write(analysis_result)

        st.markdown("### ✏️ Your Turn: Rewrite the Negative Sentences")
        st.write("You can manually rewrite the identified negative sentences. If you need help, proceed to Step 2.")
    else:
        st.warning("Please enter some text to analyze.")

# 🔹 Step 2: Allow User to Use AI for Rewriting
st.markdown("## Step 2: Use AI to Rewrite (Optional)")
use_ai_rewriting = st.radio("Would you like AI to rewrite the text for you?", ["No", "Yes"])

if use_ai_rewriting == "Yes":
    if user_text:
        rewritten_text = rewrite_text_mistral(user_text)
        st.markdown("### ✅ AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using AI to rewrite.")
