import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from better_profanity import profanity
import re

# ✅ Load Sentiment & Profanity Filters
analyzer = SentimentIntensityAnalyzer()
CUSTOM_PROFANITY = ["scammer", "thief", "fraud", "idiot", "lazy", "moron"]
profanity.load_censor_words(CUSTOM_PROFANITY)

# ✅ Function to Detect Negative/Aggressive Sentences
def analyze_text(text):
    """Detects aggressive, negative, or unprofessional language."""
    sentences = re.split(r'(?<=[.!?])\s+', text)  # Split text into sentences
    flagged_sentences = []

    for sentence in sentences:
        sentiment_score = analyzer.polarity_scores(sentence)["compound"]
        is_passive = detect_passive_sentiment(sentence)
        has_vulgarity = detect_vulgarity(sentence)

        if sentiment_score < -0.5 or is_passive or has_vulgarity:
            flagged_sentences.append((sentence, sentiment_score, is_passive, has_vulgarity))

    return flagged_sentences

# ✅ Function to Detect Passive Sentiment
def detect_passive_sentiment(text):
    """Checks if sentiment is weak (passive) instead of aggressive."""
    blob = TextBlob(text)
    if -0.3 < blob.sentiment.polarity < 0.3:  # Low polarity means neutral/passive
        return True
    return False

# ✅ Function to Detect Profanity
def detect_vulgarity(text):
    """Checks if text contains profanity or offensive words."""
    return profanity.contains_profanity(text)

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
                issues = []
                if score < -0.5:
                    issues.append("Aggressive Tone")
                if is_passive:
                    issues.append("Passive Tone")
                if has_vulgarity:
                    issues.append("Vulgar Language")
                
                issue_text = ", ".join(issues)
                st.markdown(f"- **{sent}** _(Issues: {issue_text}, Score: {score:.2f})_")

            st.warning("⚠️ Please rewrite the above sentences in a more professional and neutral tone before submission.")
        else:
            st.success("✅ No issues detected.")

    else:
        st.warning("Please enter some text to analyze.")
