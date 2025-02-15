import streamlit as st
import os
import docx
import torch
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

# Set a persistent NLTK data directory to avoid missing dependencies
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

# Ensure necessary NLTK resources are available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=NLTK_DATA_DIR, quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)

@st.cache_resource
def load_models():
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english", device=0 if torch.cuda.is_available() else -1)
    toxicity_analyzer = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target", device=0 if torch.cuda.is_available() else -1)
    gpt_model_name = "facebook/bart-large-cnn"
    tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(gpt_model_name)
    return sentiment_analyzer, toxicity_analyzer, tokenizer, model

sentiment_analyzer, toxicity_analyzer, tokenizer, model = load_models()

def analyze_text(text):
    flagged_words = defaultdict(list)
    sentiment_scores = []  # Properly initialized
    toxicity_scores = []  # Properly initialized

    sentences = sent_tokenize(text)
    for sent in sentences:
        sentiment = sentiment_analyzer(sent)[0]  # Extract first element
        toxicity = toxicity_analyzer(sent)[0]  # Extract first element

        sentiment_scores.append(sentiment)
        toxicity_scores.append(toxicity)

        words = word_tokenize(sent)
        for word in words:
            toxicity_word = toxicity_analyzer(word)[0]  # Extract first element
            if toxicity_word['label'] in ['toxic', 'severe_toxic', 'insult', 'threat', 'identity_hate']:
                flagged_words[sent].append((word, toxicity_word['score']))

    return flagged_words, sentiment_scores, toxicity_scores

def highlight_text(text, flagged_words):
    highlighted_text = text
    offset = 0
    for sentence, flagged_data in flagged_words.items():
        for word, score in flagged_data:
            idx = text.find(word, text.find(sentence))  # Find within the sentence
            if idx != -1:
                idx += offset
                highlighted_text = (
                    highlighted_text[:idx]
                    + f' **[{word}]** '  # Highlight flagged word
                    + highlighted_text[idx + len(word):]
                )
                offset += 6  # Adjust for markdown bold tags
    return highlighted_text

def suggest_rewording(text, context=""):  
    inputs = tokenizer("Rewrite this in a more neutral and respectful way for a legal document: " + context + " " + text, return_tensors="pt", max_length=1024, truncation=True)  
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    reworded_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)  # Extract first output
    return reworded_text

def extract_text_from_docx(docx_file):
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure legal submissions are neutral and constructive using AI.")

input_option = st.radio("Choose input method:", ("Text Box", "Upload Word Document"))

if input_option == "Text Box":
    user_text = st.text_area("Enter your text here:")
    if st.button("Analyze Text"):
        if user_text:
            flagged_words, sentiment_scores, toxicity_scores = analyze_text(user_text)
            highlighted_text = highlight_text(user_text, flagged_words)
            reworded_text = suggest_rewording(user_text)

            st.markdown("### Highlighted Text")
            st.markdown(highlighted_text)

            st.markdown("### Sentiment Analysis")
            st.write(sentiment_scores)

            st.markdown("### Toxicity Analysis")
            st.write(toxicity_scores)

            st.markdown("### Suggested Rewording")
            st.write(reworded_text)
        else:
            st.warning("Please enter some text to analyze.")

elif input_option == "Upload Word Document":
    uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])
    if uploaded_file is not None:
        user_text = extract_text_from_docx(uploaded_file)
        flagged_words, sentiment_scores, toxicity_scores = analyze_text(user_text)
        highlighted_text = highlight_text(user_text, flagged_words)
        reworded_text = suggest_rewording(user_text)

        st.markdown("### Extracted Text")
        st.text_area("Document Content", user_text, height=200)

        st.markdown("### Highlighted Text")
        st.markdown(highlighted_text)

        st.markdown("### Sentiment Analysis")
        st.write(sentiment_scores)

        st.markdown("### Toxicity Analysis")
        st.write(toxicity_scores)

        st.markdown("### Suggested Rewording")
        st.write(reworded_text)
