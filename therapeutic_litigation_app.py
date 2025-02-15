import os
import streamlit as st
import openai

# ✅ Set API Key for OpenAI Free GPT (GPT-3.5 Turbo)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Ensure API key is set in environment variables

if not OPENAI_API_KEY:
    st.error("⚠️ OpenAI API key is missing. Set it as an environment variable before running the app.")
    st.stop()

client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Function to Identify Negative Sentences (Using Free GPT)
def identify_negative_statements(text):
    """Uses GPT-3.5 Turbo to detect aggressive or negative statements in the text."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in sentiment analysis and legal writing."},
                {"role": "user", "content": f"Identify sentences in the following text that contain aggressive, negative, or hostile language:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing text: {e}"

# ✅ Function to Rewrite Text (Using Free GPT)
def rewrite_text_gpt(text):
    """Uses GPT-3.5 Turbo to rewrite text in a professional, neutral tone."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert in legal writing. Rewrite text to be professional and neutral."},
                {"role": "user", "content": f"Rewrite this legal document in a more professional and neutral tone:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error rewriting text with GPT-3.5 Turbo: {e}"

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
        rewritten_text = rewrite_text_gpt(user_text)
        st.markdown("### ✅ AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using AI to rewrite.")
