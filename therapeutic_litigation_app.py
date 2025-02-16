import streamlit as st
import fasttext
import os
import re
import numpy as np

# ✅ Pre-trained FastText Model (Download from FastText)
MODEL_PATH = "cc.en.300.bin"  # Pre-trained model file (Download from FastText: https://fasttext.cc/docs/en/crawl-vectors.html)
if not os.path.exists(MODEL_PATH):
    st.error("⚠️ FastText model missing. Download from https://fasttext.cc/docs/en/crawl-vectors.html")
    st.stop()

# ✅ Load FastText Model
fasttext_model = fasttext.load_model(MODEL_PATH)

# ✅ Load Pre-Trained Aggressive Words Dictionary
AGGRESSIVE_WORDS = [
    "scammer", "fraud", "cheat", "thief", "liar", "stupid", "pathetic", "dishonest", "corrupt", 
    "idiot", "lazy", "criminal", "untrustworthy", "snake", "moron", "terrible", "evil"
]  # You can expand this list with real datasets like LIWC or Hatebase

# ✅ Function to Compute Semantic Similarity with Aggressive Words
def is_aggressive(word):
    """Check if a word is semantically similar to aggressive words using FastText embeddings."""
    word_vector = fasttext_model.get_word_vector(word.lower())
    similarities = []

    for aggressive_word in AGGRESSIVE_WORDS:
        aggressive_vector = fasttext_model.get_word_vector(aggressive_word)
        similarity = np.dot(word_vector, aggressive_vector) / (np.linalg.norm(word_vector) * np.linalg.norm(aggressive_vector))
        similarities.append(similarity)

    return max(similarities) > 0.7  # Adjust threshold as needed

# ✅ Function to Detect Aggressive Sentences
def analyze_text(text):
    """Detects aggressive sentences based on word similarity."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    flagged_sentences = []

    for sentence in sentences:
        words = sentence.split()
        aggressive_found = any(is_aggressive(word) for word in words)

        if aggressive_found:
            flagged_sentences.append(sentence)

    return flagged_sentences

# ✅ Function to Highlight Aggressive Words
def highlight_text(text):
    """Highlights aggressive words detected via semantic similarity."""
    words = text.split()
    highlighted_text = " ".join([f'**🔴 {word} 🔴**' if is_aggressive(word) else word for word in words])
    return highlighted_text

# ✅ Streamlit UI
st.title("📝 AI-Powered Litigation Assistant (No Rule-Based Methods)")
st.write("Identify aggressive language using AI-powered word embeddings.")

# 🔹 Step 1: User Inputs Legal Case Submission
st.markdown("## Step 1: Identify Aggressive Language")
user_text = st.text_area("Enter your legal submission for analysis:")

if st.button("Analyze Text"):
    if user_text:
        flagged_sentences = analyze_text(user_text)
        highlighted_text = highlight_text(user_text)

        st.markdown("### 🔍 Flagged Sentences & Required Rewriting")
        if flagged_sentences:
            for sent in flagged_sentences:
                st.markdown(f"- **{sent}**")
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
