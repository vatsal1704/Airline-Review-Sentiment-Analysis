import streamlit as st
import nltk
import re
import pickle
import os

# --- THE DEFINITIVE FIX: Download all data locally ---

# Define a local directory to store NLTK data
LOCAL_NLTK_DATA_PATH = os.path.join(os.path.dirname(__file__), "nltk_data")

# Create the directory if it doesn't exist
os.makedirs(LOCAL_NLTK_DATA_PATH, exist_ok=True)

# Add the local path to NLTK's data path
nltk.data.path.append(LOCAL_NLTK_DATA_PATH)

@st.cache_resource
def download_nltk_packages():
    """
    Downloads required NLTK packages to the local directory.
    This function is cached to run only once.
    """
    # ADDED 'punkt_tab' TO THE LIST OF PACKAGES
    packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
    
    for package in packages:
        try:
            # Adjust find path for different package types
            if package.startswith('punkt'):
                nltk.data.find(f"tokenizers/{package}")
            else:
                nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package, download_dir=LOCAL_NLTK_DATA_PATH)

# Run the download function
download_nltk_packages()

# --- End of Fix ---


# --- Main App Logic ---
st.title("✈️ Airline Tweet Sentiment Analysis")

try:
    # Initialize tools
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
        """Cleans, tokenizes, and lemmatizes the input text."""
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
            if prediction[0] == 0:
                st.error("Negative Sentiment 😞")
            elif prediction[0] == 1:
                st.warning("Neutral Sentiment 😐")
            else:
                st.success("Positive Sentiment 😊")
        else:
            st.warning("Please enter a tweet to analyze.")

except FileNotFoundError:
    st.error("Model or vectorizer file not found! 🚨 Please ensure 'xgb_pkl' and 'vectorizer.pkl' are in your project directory.")
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
