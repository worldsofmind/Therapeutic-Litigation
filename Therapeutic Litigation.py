import streamlit as st
import docx
import re
import torch
from transformers import pipeline
from collections import defaultdict
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Download required nltk resources
nltk.download('punkt')
nltk.download('stopwords')

# Load a pre-trained sentiment analysis model
sentiment_analyzer = pipeline("sentiment-analysis")

# Example replacement suggestions using GPT-based rewriting
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

gpt_model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(gpt_model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(gpt_model_name)


def analyze_text(text):
    flagged_sections = defaultdict(list)
    sentiment_scores = sentiment_analyzer(text)
    
    words = word_tokenize(text)
    for word in words:
        if word.lower() in stopwords.words('english'):
            flagged_sections[word].append(text.find(word))
    
    return flagged_sections, sentiment_scores


def highlight_text(text, flagged_sections):
    """Highlight flagged words in the text."""
    highlighted_text = text
    offset = 0
    for word, indices in flagged_sections.items():
        for idx in indices:
            idx += offset
            highlighted_text = (
                highlighted_text[:idx]
                + f' **[{word}]** '
                + highlighted_text[idx + len(word):]
            )
            offset += 6  # Adjust for markdown bold tags
    
    return highlighted_text


def suggest_rewording(text):
    """Use GPT to rephrase text in a constructive manner."""
    inputs = tokenizer("Summarize: " + text, return_tensors="pt", max_length=512, truncation=True)
    summary_ids = model.generate(inputs.input_ids, max_length=150, min_length=50, length_penalty=2.0, num_beams=4)
    reworded_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return reworded_text


def extract_text_from_docx(docx_file):
    """Extract text from a Word document."""
    doc = docx.Document(docx_file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

# Streamlit UI
st.title("📝 AI-Powered Therapeutic Litigation Assistant")
st.write("Ensure submissions are constructive and free from aggressive or hostile language using AI.")

# User input options
input_option = st.radio("Choose input method:", ("Text Box", "Upload Word Document"))

if input_option == "Text Box":
    user_text = st.text_area("Enter your text here:")
    if st.button("Analyze Text"):
        if user_text:
            flagged_sections, sentiment_scores = analyze_text(user_text)
            highlighted_text = highlight_text(user_text, flagged_sections)
            reworded_text = suggest_rewording(user_text)
            
            st.markdown("### Highlighted Text")
            st.markdown(highlighted_text)
            
            st.markdown("### Sentiment Analysis")
            st.write(sentiment_scores)
            
            st.markdown("### Suggested Rewording")
            st.write(reworded_text)
        else:
            st.warning("Please enter some text to analyze.")

elif input_option == "Upload Word Document":
    uploaded_file = st.file_uploader("Upload a .docx file", type=["docx"])
    if uploaded_file is not None:
        user_text = extract_text_from_docx(uploaded_file)
        flagged_sections, sentiment_scores = analyze_text(user_text)
        highlighted_text = highlight_text(user_text, flagged_sections)
        reworded_text = suggest_rewording(user_text)
        
        st.markdown("### Extracted Text")
        st.text_area("Document Content", user_text, height=200)
        
        st.markdown("### Highlighted Text")
        st.markdown(highlighted_text)
        
        st.markdown("### Sentiment Analysis")
        st.write(sentiment_scores)
        
        st.markdown("### Suggested Rewording")
        st.write(reworded_text)
