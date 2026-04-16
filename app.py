from flask import Flask, request, jsonify
from utils import load_model, preprocess_input

app = Flask(__name__)

model, encoders = load_model()

@app.route("/")
def home():
    return "Loan Prediction API is Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        processed = preprocess_input(data, encoders)
        prediction = model.predict(processed)[0]

        result = "Approved" if prediction == 1 else "Rejected"

        return jsonify({
            "prediction": result
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
