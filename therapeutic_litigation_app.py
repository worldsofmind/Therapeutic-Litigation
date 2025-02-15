import streamlit as st
import openai

# ✅ OpenAI API Configuration
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY"  # Replace with your API key
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# ✅ Function to analyze text and detect negative/toxic sentences
def analyze_text(text):
    """Use GPT-4 to identify sentences with negative sentiment or aggressive language."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in sentiment analysis and legal writing."},
                {"role": "user", "content": f"Identify sentences in the following text that contain aggressive, negative, or hostile language:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error analyzing text: {e}")
        return "An error occurred during analysis."

# ✅ Function to rewrite sentences using GPT-4
def rewrite_sentence_gpt4(sentence):
    """Use GPT-4 to rewrite a single sentence in a neutral and professional tone."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in legal writing and diplomacy. Rewrite text to be neutral and professional."},
                {"role": "user", "content": f"Rewrite this legal sentence in a professional, neutral, and respectful tone:\n\n{sentence}"}
            ],
            temperature=0.5,
            max_tokens=100
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error rewriting sentence with GPT-4: {e}")
        return "An error occurred during rewriting."

# ✅ Function to rewrite full text using GPT-4
def rewrite_text_gpt4(text):
    """Use GPT-4 to rewrite the full text in a neutral and professional tone."""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert in legal writing. Rewrite the following text to be professional, neutral, and respectful."},
                {"role": "user", "content": f"Rewrite this legal document in a more professional and neutral tone:\n\n{text}"}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error rewriting text with GPT-4: {e}")
        return "An error occurred during rewriting."

# ✅ Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

# 🔹 Step 1: User Inputs Text for Analysis
st.markdown("## Step 1: Identify Negative or Aggressive Language")
user_text = st.text_area("Enter your legal text for AI analysis:")

if st.button("Analyze Text"):
    if user_text:
        analysis_result = analyze_text(user_text)
        st.markdown("### 🔍 Identified Negative Sentences/Words")
        st.write(analysis_result)

        st.markdown("### ✏️ Your Turn: Rewrite the Negative Sentences")
        st.write("You can manually rewrite the identified negative sentences. If you need help, proceed to Step 2.")
    else:
        st.warning("Please enter some text to analyze.")

# 🔹 Step 2: Allow User to Use GPT-4 for Rewriting
st.markdown("## Step 2: Use AI to Rewrite")
use_gpt4 = st.radio("Would you like GPT-4 to rewrite the text for you?", ["No", "Yes"])

if use_gpt4 == "Yes":
    if user_text:
        rewritten_text = rewrite_text_gpt4(user_text)
        st.markdown("### ✅ AI-Rewritten Version")
        st.write(rewritten_text)
    else:
        st.warning("Please enter text in Step 1 before using GPT-4 to rewrite.")
