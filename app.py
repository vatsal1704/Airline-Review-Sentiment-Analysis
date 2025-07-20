# import streamlit as st
# import pickle
# import re
# import nltk
# import spacy
# import numpy as np
# import pandas as pd
# from sklearn.feature_extraction.text import CountVectorizer
# from nltk.stem import WordNetLemmatizer
# from nltk.stem import SnowballStemmer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import warnings
# warnings.filterwarnings("ignore")


# # Force download NLTK data (safe on Streamlit Cloud)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

# # Load the trained model and vectorizer
# try:
#     with open('xgb_model.pkl', 'rb') as f:
#         xgb_model = pickle.load(f)
#     with open('vectorizer.pkl', 'rb') as f:
#         vectorizer = pickle.load(f)
# except FileNotFoundError:
#     st.error("Model or vectorizer file not found. Please ensure 'xgb_model.pkl' and 'vectorizer.pkl' are in the same directory.")
#     st.stop()


# # Initialize lemmatizer and stemmer
# le = WordNetLemmatizer()
# sb = SnowballStemmer("english")
# stop_words = set(stopwords.words("english"))

# # Text preprocessing function
# def text_preprocessing(text):
#     text = text.lower()
#     text = re.sub(r"@\w+", "", text)
#     text = re.sub(r"http\S+", "", text)
#     text = re.sub(r"[^a-z\s]+", "", text)
#     clean = " ".join([le.lemmatize(sb.stem(t), pos="v") for t in word_tokenize(text) if t not in stop_words])
#     return clean

# # Streamlit app
# st.title("Airline Tweet Sentiment Analysis")

# user_input = st.text_area("Enter your tweet about the airline:")

# if st.button("Predict Sentiment"):
#     if user_input:
#         # Preprocess the input
#         preprocessed_input = text_preprocessing(user_input)

#         # Vectorize the input
#         input_vectorized = vectorizer.transform([preprocessed_input])

#         # Predict the sentiment
#         prediction = xgb_model.predict(input_vectorized)

#         # Map prediction to sentiment label
#         sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
#         predicted_sentiment = sentiment_map[prediction[0]]

#         st.write(f"Predicted Sentiment: {predicted_sentiment}")
#     else:
#         st.write("Please enter a tweet to predict the sentiment.")










import streamlit as st
import pickle
import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import warnings
warnings.filterwarnings("ignore")

# --- Force download required NLTK resources with error handling ---
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except Exception as e:
    st.error("Failed to download NLTK resources. Please check internet connection or app permissions.")
    st.stop()

# --- Validate required resources are available ---
try:
    _ = word_tokenize("test")  # Test tokenizer works
    stop_words = set(stopwords.words("english"))
    le = WordNetLemmatizer()
    sb = SnowballStemmer("english")
except LookupError as e:
    st.error("Required NLTK data not found. Please make sure 'punkt', 'stopwords', and 'wordnet' are available.")
    st.stop()

# --- Load model and vectorizer ---
try:
    with open('xgb_model.pkl', 'rb') as f:
        xgb_model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please ensure 'xgb_model.pkl' and 'vectorizer.pkl' are in the same directory.")
    st.stop()

# --- Preprocessing function ---
def text_preprocessing(text):
    text = text.lower()
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]+", "", text)
    clean = " ".join([le.lemmatize(sb.stem(t), pos="v") for t in word_tokenize(text) if t not in stop_words])
    return clean

# --- Streamlit UI ---
st.title("Airline Tweet Sentiment Analysis")

user_input = st.text_area("Enter your tweet about the airline:")

if st.button("Predict Sentiment"):
    if user_input:
        preprocessed_input = text_preprocessing(user_input)
        input_vectorized = vectorizer.transform([preprocessed_input])
        prediction = xgb_model.predict(input_vectorized)
        sentiment_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
        predicted_sentiment = sentiment_map[prediction[0]]
        st.write(f"Predicted Sentiment: {predicted_sentiment}")
    else:
        st.warning("Please enter a tweet to predict the sentiment.")





