from flask import Flask, render_template, request
import joblib
import re

app = Flask(__name__)

# Load model and vectorizer (use raw strings for Windows)
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "vectorizer.pkl"))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text 

# Scam indicator keywords
red_flags = [
    "registration fee",
    "fees applicable",
    "whatsapp",
    "urgent hiring",
    "no experience required",
    "pay money"
]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    job_text = ""

    if request.method == 'POST':
        job_text = request.form['job_text']
        cleaned_text = clean_text(job_text)

        # Red flag detection
        flag_count = sum(1 for word in red_flags if word in cleaned_text)

        vector = vectorizer.transform([cleaned_text])

        # ML prediction + confidence
        result = model.predict(vector)
        prob = model.predict_proba(vector)[0]
        confidence = round(max(prob) * 100, 2)

        # Hybrid decision
        if flag_count >= 2:
            prediction = "⚠️ Fake Job / Internship (Red-Flag Detected)"
        else:
            if result[0] == 1:
                prediction = "⚠️ Fake Job / Internship (ML Based)"
            else:
                prediction = "✅ Real Job / Internship"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        job_text=job_text
    )

if __name__ == "__main__":
    app.run(debug=True)
