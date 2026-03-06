# Import required libraries
import pandas as pd

# Import ML tools
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the preprocessed dataset
df = pd.read_csv("fake_job_postings.csv")

# Combine title and description (same as Module 2)
df['text'] = df['title'].fillna('') + " " + df['description'].fillna('')
df['text'] = df['text'].str.lower()

# ---------------- FEATURE EXTRACTION ----------------

# Initialize TF-IDF Vectorizer
# max_features limits vocabulary size (prevents overfitting)
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english'
)

# Convert text data into numerical feature vectors
X = vectorizer.fit_transform(df['text'])

# Target variable (0 = Real, 1 = Fake)
y = df['fraudulent']

# ---------------- TRAIN–TEST SPLIT ----------------

# Split data into training and testing sets
# 80% training, 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL TRAINING ----------------

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model using training data
model.fit(X_train, y_train)

# ---------------- MODEL EVALUATION ----------------

# Predict on test data
y_pred = model.predict(X_test)

# Print accuracy of the model
print("Accuracy:", accuracy_score(y_test, y_pred))

# Print detailed classification report
print(classification_report(y_test, y_pred))

# Import joblib to save model
import joblib

# Save trained Logistic Regression model
joblib.dump(model, "model.pkl")

# Save TF-IDF vectorizer
joblib.dump(vectorizer, "vectorizer.pkl")

print("Model and Vectorizer saved successfully!")
