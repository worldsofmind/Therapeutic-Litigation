import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ✅ Load Mistral 7B Locally
MODEL_NAME = "mistralai/Mistral-7B-Instruct"

@st.cache_resource
def load_mistral():
    """Load Mistral model and tokenizer locally."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return pipe

mistral_pipeline = load_mistral()

# ✅ Function to Identify Negative Sentences (Local LLM)
def identify_negative_statements(text):
    """Uses Mistral to detect aggressive or negative statements."""
    prompt = f"Identify sentences in the following text that contain aggressive, negative, or hostile language:\n\n{text}"
    response = mistral_pipeline(prompt, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

# ✅ Function to Rewrite Text (Local LLM)
def rewrite_text_mistral(text):
    """Uses Mistral to rewrite text in a professional, neutral tone."""
    prompt = f"Rewrite this legal document in a more professional and neutral tone:\n\n{text}"
    response = mistral_pipeline(prompt, max_length=500, num_return_sequences=1)
    return response[0]["generated_text"]

# ✅ Streamlit UI
st.title("📝 Local LLM-Powered Legal Writing Assistant (No GPT)")
st.write("Ensure legal submissions are neutral and constructive using AI-free LLMs.")

# 🔹 Step 1: User Inputs Text for Analysis
st.markdown("## Step 1: Identify Negative or Aggressive Language")
user_text = st.text_area("Enter your legal text for LLM analysis:")

if st.button("Analyze Text"):
    if user_text:
        analysis_result = identify_negative_statements(user_text)
        st.markdown("### 🔍 Identified Negative Sentences")
        st.write(analysis_result)

        st.markdown("### ✏️ Your Turn: Rewrite the Negative Sentences")
        st.write("You can manually rewrite the identified negative sentences. If you need help, proceed to Step 2.")
    else:
        st.warning("Please enter some text to analyze.")

# 🔹 Step 2 (Optional): LLM Rewriting
st.markdown("## Step 2: LLM-Powered Rewording (Optional)")
use_llm_rewriting = st.radio("Would you like the LLM to rewrite the text for you?", ["No", "Yes"])

if use_llm_rewriting == "Yes":
    if user_text:
        rewritten_text = rewrite_text_mistral(user_text)
        st.markdown("### ✅ LLM-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using LLM to rewrite.")
