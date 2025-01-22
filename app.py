import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk

# Ensure punkt and stopwords are downloaded
nltk.download('punkt')
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [
        ps.stem(token) for token in tokens
        if token.isalnum() and token not in stopwords.words('english') and token not in string.punctuation
    ]
    return " ".join(filtered_tokens)


# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Email/SMS Spam Classifier")
input_sms = st.text_input("Enter the message")

if st.button('Predict'):
    if input_sms:  # Check if input is not empty
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display result
        if result == 1:
            st.header("Spam")
        else:
            st.header("Not Spam")
    else:
        st.warning("Please enter a message.")
