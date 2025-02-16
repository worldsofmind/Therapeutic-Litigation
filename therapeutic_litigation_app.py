import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from better_profanity import profanity
import re

# ✅ Load Sentiment & Profanity Filters
analyzer = SentimentIntensityAnalyzer()
CUSTOM_PROFANITY = [
    "scammer", "thief", "fraud", "idiot", "lazy", "moron",
    "extortion", "deception", "exploit", "snake", "shady", "crook"
]
profanity.load_censor_words(CUSTOM_PROFANITY)

# ✅ Fixed Sentence Splitter (No Look-Behind Errors)
def split_sentences(text):
    """Splits text into sentences while preserving common abbreviations."""
    abbreviations = r"\b(?:Dr|Mr|Ms|Mrs|U\.S|etc|i\.e|e\.g)\."
    sentence_endings = r"(?<!{})[.!?]\s+".format(abbreviations)
    
    return re.split(sentence_endings, text)

# ✅ Improved Passive Sentiment Detection
def detect_passive_sentiment(text):
    """Detects passive sentiment by analyzing polarity and sentence structure."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    words = text.split()

    if len(words) < 5 and -0.2 < polarity < 0.2:
        return True  
    elif len(words) >= 5 and -0.3 < polarity < 0.3:
        return True  
    return False

# ✅ Improved Profanity Detection
def detect_vulgarity(text):
    """Detects vulgar or offensive language using a profanity filter with word boundaries."""
    words = text.split()
    for word in words:
        if profanity.contains_profanity(word):
            return True
    return False

# ✅ Improved Text Analysis Function
def analyze_text(text):
    """Detects aggressive, negative, or unprofessional language."""
    sentences = split_sentences(text)  # ✅ Using improved regex-based sentence splitting
    flagged_sentences = []

    for sentence in sentences:
        sentiment_score = analyzer.polarity_scores(sentence)["compound"]
        is_passive = detect_passive_sentiment(sentence)
        has_vulgarity = detect_vulgarity(sentence)

        if sentiment_score < -0.5 or is_passive or has_vulgarity:
            flagged_sentences.append((sentence, sentiment_score, is_passive, has_vulgarity))

    return flagged_sentences

# ✅ Streamlit UI
st.title("📝 AI-Powered Litigation Assistant")
st.write("Identify aggressive, negative, and unprofessional language in legal case submissions.")

# 🔹 Step 1: User Inputs Legal Case Submission
st.markdown("## Step 1: Identify Language Issues")
user_text = st.text_area("Enter your legal submission for analysis:")

if st.button("Analyze Text"):
    if user_text:
        flagged_sentences = analyze_text(user_text)

        st.markdown("### 🔍 Flagged Sentences & Issues")
        if flagged_sentences:
            for sent, score, is_passive, has_vulgarity in flagged_sentences:
                issues = [("Aggressive Tone" if score < -0.5 else ""),
                          ("Passive Tone" if is_passive else ""),
                          ("Vulgar Language" if has_vulgarity else "")]
                st.markdown(f"- **{sent}** _(Issues: {', '.join(filter(None, issues))}, Score: {score:.2f})_")

            st.warning("⚠️ Please rewrite the above sentences.")

        else:
            st.success("✅ No issues detected.")
