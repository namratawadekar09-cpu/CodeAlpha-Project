from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)
CORS(app)

MODEL_PATH = Path(__file__).parent / "model" / "iris_model.joblib"

model_data = None


def load_model():
    global model_data

    if model_data is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Model not found. Run 'python train_model.py' first."
            )
        model_data = joblib.load(MODEL_PATH)

    return model_data


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "Iris classification API is running"
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True)

        if not data:
            return jsonify({
                "success": False,
                "error": "Please send JSON input."
            }), 400

        required_fields = [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width"
        ]

        for field in required_fields:
            if field not in data:
                return jsonify({
                    "success": False,
                    "error": f"Missing field: {field}"
                }), 400

        values = []
        for field in required_fields:
            value = float(data[field])

            if not np.isfinite(value):
                raise ValueError(f"{field} must be a valid number.")

            if value <= 0:
                raise ValueError(f"{field} must be greater than zero.")

            values.append(value)

        loaded = load_model()
        model = loaded["model"]
        target_names = loaded["target_names"]

        input_array = np.array([values])
        predicted_index = int(model.predict(input_array)[0])
        probabilities = model.predict_proba(input_array)[0]

        prediction = target_names[predicted_index]
        confidence = float(probabilities[predicted_index] * 100)

        probability_result = {
            target_names[index]: round(float(probability * 100), 2)
            for index, probability in enumerate(probabilities)
        }

        return jsonify({
            "success": True,
            "prediction": prediction,
            "confidence": round(confidence, 2),
            "probabilities": probability_result
        })

    except ValueError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 400

    except FileNotFoundError as error:
        return jsonify({
            "success": False,
            "error": str(error)
        }), 500

    except Exception:
        return jsonify({
            "success": False,
            "error": "An unexpected server error occurred."
        }), 500


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
