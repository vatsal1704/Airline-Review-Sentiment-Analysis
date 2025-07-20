import streamlit as st
import nltk
import re
import pickle
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

# --- FIX: Download NLTK data ---
# This function downloads the required NLTK packages.
# The @st.cache_resource decorator ensures this runs only once when the app starts.
@st.cache_resource
def download_nltk_data():
    """Downloads necessary NLTK data models."""
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Call the function to ensure the data is available before the app runs
download_nltk_data()
# --- END OF FIX ---


# --- App Functions ---
# Load the saved model and vectorizer
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same directory as this script.")
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
    
    # The line below is where the original error occurred
    clean = " ".join([le.lemmatize(sb.stem(t), pos="v") for t in word_tokenize(text) if t not in stop_words])
    return clean


# --- Streamlit App Interface ---
st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("Enter a tweet about an airline to analyze its sentiment.")

user_input = st.text_area("Your tweet:", placeholder="e.g., 'The flight was delayed but the crew was fantastic!'")

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
