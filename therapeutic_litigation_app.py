import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re

# ✅ Load VADER Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# ✅ Function to Detect Aggressive Sentences
def analyze_text(text):
    """Detects aggressive or negative sentiment using VADER."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    flagged_sentences = []

    for sentence in sentences:
        score = analyzer.polarity_scores(sentence)["compound"]  # VADER sentiment score
        if score < -0.5:  # Threshold for aggressive/unprofessional language
            flagged_sentences.append((sentence, score))

    return flagged_sentences

# ✅ Function to Highlight Aggressive Words
def highlight_text(text):
    """Highlights aggressive words detected by VADER."""
    words = text.split()
    highlighted_text = " ".join(
        [f'**🔴 {word} 🔴**' if analyzer.polarity_scores(word)["compound"] < -0.5 else word for word in words]
    )
    return highlighted_text

# ✅ Streamlit UI
st.title("📝 AI-Powered Litigation Assistant (Using VADER)")
st.write("Identify aggressive language and unprofessional tone in legal case submissions.")

# 🔹 Step 1: User Inputs Legal Case Submission
st.markdown("## Step 1: Identify Aggressive Language & Legal Tone Issues")
user_text = st.text_area("Enter your legal submission for analysis:")

if st.button("Analyze Text"):
    if user_text:
        flagged_sentences = analyze_text(user_text)
        highlighted_text = highlight_text(user_text)

        st.markdown("### 🔍 Flagged Sentences & Required Rewriting")
        if flagged_sentences:
            for sent, score in flagged_sentences:
                st.markdown(f"- **{sent}** _(Aggressiveness Score: {score})_")
            st.warning("⚠️ Please rewrite the above sentences in a more professional and neutral tone before submission.")
        else:
            st.success("✅ No aggressive language detected.")

        st.markdown("### ✏️ Highlighted Aggressive Words")
        st.write(highlighted_text)

        st.markdown("### ✏️ Your Turn: Rewrite the Flagged Sentences")
        st.write("You can manually rewrite the flagged sentences below. If you need AI assistance, proceed to Step 2.")
    else:
        st.warning("Please enter some text to analyze.")

# 🔹 Step 2 (Optional): AI-Powered Rewriting (Using GPT4All or Other LLMs)
st.markdown("## Step 2: AI-Powered Rewriting (Optional)")
use_ai_rewriting = st.radio("Would you like AI to rewrite the text for you?", ["No", "Yes"])

if use_ai_rewriting == "Yes":
    if user_text:
        st.markdown("### ✅ AI-Rewritten Version (Coming Soon)")
        st.write("AI-generated rewording will be available in the next update.")
    else:
        st.warning("Please enter text in Step 1 before using AI to rewrite.")
