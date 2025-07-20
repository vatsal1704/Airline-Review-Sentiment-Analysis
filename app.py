import streamlit as st
import nltk
import re
import pickle
import os # Import the 'os' module to check for file paths

# --- DIAGNOSTIC MODE: Force NLTK Download ---
st.set_page_config(layout="wide")
st.title("👨‍⚕️ NLTK Downloader Diagnostics")

# Define the download directory within the app's running environment
# This is a common path used by Streamlit Cloud
nltk_data_path = "/home/adminuser/nltk_data"
nltk.data.path.append(nltk_data_path)

st.info(f"Step 1: NLTK will try to download data to: **{nltk_data_path}**")

# --- Download Punkt ---
st.subheader("1. Tokenizer Model ('punkt')")
try:
    nltk.data.find("tokenizers/punkt", paths=[nltk_data_path])
    st.success("✅ 'punkt' is already downloaded.")
except LookupError:
    st.warning("🚨 'punkt' not found. Attempting download...")
    try:
        nltk.download('punkt', download_dir=nltk_data_path)
        # Verify download
        if os.path.exists(os.path.join(nltk_data_path, 'tokenizers/punkt')):
             st.success("✅ 'punkt' downloaded successfully!")
        else:
             st.error("❌ 'punkt' download command ran, but files are still missing.")
    except Exception as e:
        st.error(f"An error occurred during 'punkt' download: {e}")

# --- Download Stopwords ---
st.subheader("2. Stopwords Corpus")
try:
    nltk.data.find("corpora/stopwords", paths=[nltk_data_path])
    st.success("✅ 'stopwords' are already downloaded.")
except LookupError:
    st.warning("🚨 'stopwords' not found. Attempting download...")
    try:
        nltk.download('stopwords', download_dir=nltk_data_path)
        # Verify download
        if os.path.exists(os.path.join(nltk_data_path, 'corpora/stopwords')):
             st.success("✅ 'stopwords' downloaded successfully!")
        else:
             st.error("❌ 'stopwords' download command ran, but files are still missing.")
    except Exception as e:
        st.error(f"An error occurred during 'stopwords' download: {e}")

st.divider()

# --- Main App Logic ---
st.title("✈️ Airline Tweet Sentiment Analysis")

# This part will only run if the downloads above succeed
try:
    # Initialize tools using the specified path
    from nltk.corpus import stopwords
    from nltk.stem import SnowballStemmer, WordNetLemmatizer
    from nltk.tokenize import word_tokenize

    stop_words = set(stopwords.words("english"))
    sb = SnowballStemmer("english")
    le = WordNetLemmatizer()

    # Load model and vectorizer
    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    def text_preprocessing(text):
        text = re.sub("[^a-zA-Z]", " ", text).lower()
        clean = " ".join([le.lemmatize(sb.stem(t), pos="v") for t in word_tokenize(text) if t not in stop_words])
        return clean

    # Streamlit Interface
    user_input = st.text_area("Your tweet:", placeholder="e.g., 'My flight was wonderfully smooth!'")
    if st.button("Analyze Sentiment"):
        if user_input.strip():
            preprocessed_input = text_preprocessing(user_input)
            vectorized_input = vectorizer.transform([preprocessed_input])
            prediction = model.predict(vectorized_input)
            st.subheader("Analysis Result:")
            if prediction[0] == 0: st.error("Negative Sentiment 😞")
            elif prediction[0] == 1: st.warning("Neutral Sentiment 😐")
            else: st.success("Positive Sentiment 😊")
        else:
            st.warning("Please enter a tweet to analyze.")

except Exception as e:
    st.error(f"A critical error occurred after the download phase: {e}")
    st.error("This likely means one of the downloads failed silently. Please review the diagnostic messages above.")
