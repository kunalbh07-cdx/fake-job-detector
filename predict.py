# Import required libraries
import joblib
import re

# Load saved model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Function to clean input text (same logic as Module 2)
def clean_text(text):
    text = text.lower()                       # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)   # Remove special characters
    return text

# Take job description as input from user
job_text = input("Enter job/internship description: ")

# Clean the input text
cleaned_text = clean_text(job_text)

# Convert text into TF-IDF features
text_vector = vectorizer.transform([cleaned_text])

# Predict using trained model
prediction = model.predict(text_vector)

# Show result
if prediction[0] == 1:
    print("⚠️ Prediction: FAKE Job / Internship")
else:
    print("✅ Prediction: REAL Job / Internship")
