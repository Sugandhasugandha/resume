import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pickle

# Load the saved vectorizer and model using pickle
with open('resume_vectorizer.pkl', 'rb') as file:
    word_vectorizer = pickle.load(file)

with open('resume_classifier.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Define Streamlit app
def main():
    st.title('Resume Screening App')
    st.write('Enter your resume text below:')
    
    # Text input for resume
    resume_text = st.text_area('Resume Text', height=200)

    # Predict button
    if st.button('Predict'):
        if resume_text:
            # Vectorize the resume text
            resume_vectorized = word_vectorizer.transform([resume_text])

            # Make prediction using the loaded model
            prediction = loaded_model.predict(resume_vectorized)

            # Get the predicted category
            predicted_category = prediction[0]

            # Display predicted category
            st.success(f'Predicted Category: {predicted_category}')
        else:
            st.warning('Please enter some text for prediction.')

if __name__ == '__main__':
    main()
