import streamlit as st
import requests
import os

# ✅ Hugging Face API Key (Free Models)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")  # Ensure API key is set in environment variables

if not HUGGINGFACE_API_KEY:
    st.error("⚠️ Hugging Face API key is missing. Set it as an environment variable before running the app.")
    st.stop()

# ✅ Function to Identify Negative Sentences (Using Mistral 7B)
def identify_negative_statements(text):
    """Uses Mistral 7B to detect aggressive or negative statements in the text."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": f"Identify sentences in the following text that contain aggressive, negative, or hostile language:\n\n{text}",
               "parameters": {"max_length": 500, "num_return_sequences": 1}}
    
    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.json()}"

# ✅ Function to Rewrite Text (Using Mistral 7B)
def rewrite_text_mistral(text):
    """Uses Mistral 7B to rewrite text in a professional, neutral tone."""
    API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
    payload = {"inputs": f"Rewrite this legal document in a more professional and neutral tone:\n\n{text}",
               "parameters": {"max_length": 500, "num_return_sequences": 1}}

    response = requests.post(API_URL, headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return f"Error: {response.json()}"

# ✅ Streamlit UI
st.title("📝 Free AI-Powered Legal Writing Assistant")
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

# 🔹 Step 2 (Optional): AI Rewriting
st.markdown("## Step 2: AI-Powered Rewording (Optional)")
use_ai_rewriting = st.radio("Would you like AI to rewrite the text for you?", ["No", "Yes"])

if use_ai_rewriting == "Yes":
    if user_text:
        rewritten_text = rewrite_text_mistral(user_text)
        st.markdown("### ✅ AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using AI to rewrite.")
