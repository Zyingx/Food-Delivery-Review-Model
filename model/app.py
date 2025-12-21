import joblib
import os
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model & vectorizer
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
vectorizer_path = os.path.join(os.path.dirname(__file__), "tfidf_vectorizer.pkl")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

@app.route("/prediction", methods=["POST"])
def predict():
    data = request.get_json()

    if not data or "sentence" not in data or not data["sentence"].strip():
        return jsonify({"error": "Empty Sentence"}), 400

    vector = vectorizer.transform([data["sentence"]])
    result = model.predict(vector)[0]

    return jsonify({"prediction": str(result)})

if __name__ == "__main__":
    app.run(port=4400, debug=True)
