from flask import Flask, request, render_template
import pickle
import numpy as np
import os

# Load model and (optional) scaler if present
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load model (required)
model = pickle.load(open(MODEL_PATH, "rb"))

# Load scaler if available (optional)
scaler = None
if os.path.exists(SCALER_PATH):
    try:
        scaler = pickle.load(open(SCALER_PATH, "rb"))
    except Exception:
        scaler = None

# IMPORTANT: order must match the order you used during training
FEATURE_ORDER = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "LSTAT"]

app = Flask(__name__)

@app.route("/")
def home():
    # sensible defaults to help non-expert users
    defaults = {
        "CRIM": 0.2, "ZN": 0.0, "INDUS": 10.0, "CHAS": 0,
        "NOX": 0.5, "RM": 6.0, "AGE": 45.0, "DIS": 4.0,
        "RAD": 1, "TAX": 300.0, "PTRATIO": 18.0, "LSTAT": 12.0
    }
    return render_template("index.html", defaults=defaults, prediction_text=None, error=None)

@app.route("/predict", methods=["POST"])
def predict():
    errors = []
    values = []
    for feat in FEATURE_ORDER:
        raw = request.form.get(feat, "").strip()
        if raw == "":
            errors.append(f"Missing value: {feat}")
            values.append(np.nan)
            continue
        try:
            if feat in ("CHAS", "RAD"):  # categorical/integer features
                v = int(float(raw))
            else:
                v = float(raw)
            values.append(v)
        except ValueError:
            errors.append(f"Invalid value for {feat}: {raw}")
            values.append(np.nan)

    if errors:
        # show errors and keep the entered values as defaults so user can correct
        return render_template("index.html", prediction_text=None, error="; ".join(errors), defaults=dict(zip(FEATURE_ORDER, values)))

    X = np.array([values])

    # If you used a scaler during training, apply it here
    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            return render_template("index.html", prediction_text=None, error=f"Scaler transform failed: {e}", defaults=dict(zip(FEATURE_ORDER, values)))

    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        return render_template("index.html", prediction_text=None, error=f"Prediction failed: {e}", defaults=dict(zip(FEATURE_ORDER, values)))

    # If your model was trained on median home values scaled (e.g. in $1000s), adjust here.
    return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:.2f}", error=None, defaults=dict(zip(FEATURE_ORDER, values)))

if __name__ == "__main__":
    # Render will set PORT; keep default for local testing
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
