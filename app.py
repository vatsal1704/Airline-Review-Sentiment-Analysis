import streamlit as st
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- THIS IS THE CRUCIAL FIX ---
# This function downloads all necessary NLTK data.
# The decorator @st.cache_resource ensures this runs only ONCE when the app starts.
@st.cache_resource
def download_nltk_data():
    """Downloads all necessary NLTK data models."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)

# We call the function right at the start to ensure everything is downloaded.
download_nltk_data()
# --- END OF FIX ---


# --- App Functions ---
# Load the saved model and vectorizer
try:
    # Use the correct model name you specified: xgb_pkl
    with open('xgb_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer file not found! 🚨 Please ensure 'xgb_pkl' and 'vectorizer.pkl' are uploaded to your project directory.")
    st.stop()


# Initialize text processing tools
sb = SnowballStemmer("english")
le = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def text_preprocessing(text):
    """Cleans, tokenizes, and lemmatizes the input text."""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub("[^a-zA-Z]", " ", text) # Keep only letters
    text = text.lower() # Convert to lowercase
    
    # This line requires the NLTK data to be downloaded
    clean = " ".join([le.lemmatize(sb.stem(t), pos="v") for t in word_tokenize(text) if t not in stop_words])
    return clean


# --- Streamlit App Interface ---
st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("Enter a tweet about an airline to analyze its sentiment.")

user_input = st.text_area("Your tweet:", placeholder="e.g., 'My flight was wonderfully smooth and the crew was amazing!'")

if st.button("Analyze Sentiment"):
    if user_input.strip():
        # 1. Preprocess the input
        preprocessed_input = text_preprocessing(user_input)
        
        # 2. Vectorize the preprocessed text
        vectorized_input = vectorizer.transform([preprocessed_input])
        
        # 3. Predict the sentiment
        prediction = model.predict(vectorized_input)
        
        # 4. Display the result
        st.subheader("Analysis Result:")
        # Assuming 0: Negative, 1: Neutral, 2: Positive
        if prediction[0] == 0:
            st.error("Negative Sentiment 😞")
        elif prediction[0] == 1:
            st.warning("Neutral Sentiment 😐")
        else:
            st.success("Positive Sentiment 😊")
    else:
        st.warning("Please enter a tweet to analyze.")
