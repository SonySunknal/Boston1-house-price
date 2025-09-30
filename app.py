from flask import Flask, request, render_template
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Initialize Flask
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get values from form
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)[0]

    return render_template("index.html", prediction_text=f"Predicted House Price: ${prediction:.2f}")

if __name__ == "__main__":
    app.run(debug=True)