# Import required libraries
import pandas as pd                 # For data handling
import re                            # For text cleaning using regex
import nltk                          # For NLP tasks

# Import stopwords and lemmatizer from NLTK
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

# Load the dataset (CSV file downloaded from Kaggle)
df = pd.read_csv("fake_job_postings.csv")

# Combine title and description columns into a single text column
# This gives more information to the ML model
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')

# Convert all text to lowercase for uniformity
df['text'] = df['text'].str.lower()

# Function to remove special characters, numbers, and symbols
def clean_text(text):
    # Keep only alphabets and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

# Apply text cleaning function to the text column
df['text'] = df['text'].apply(clean_text)

# Load English stopwords (common words like 'the', 'is', 'and')
stop_words = set(stopwords.words('english'))

# Remove stopwords from the text
df['text'] = df['text'].apply(
    lambda x: " ".join(word for word in x.split() if word not in stop_words)
)

# Initialize lemmatizer (converts words to base form)
lemmatizer = WordNetLemmatizer()

# Apply lemmatization to each word in the text
df['text'] = df['text'].apply(
    lambda x: " ".join(lemmatizer.lemmatize(word) for word in x.split())
)

# Display the cleaned text and target label
print(df[['text', 'fraudulent']].head())
