"""
CO2 Emission Prediction Web Application

Uses a pre-trained linear regression model to predict CO2 emissions
based on vehicle engine size. Built with Flask for web interface.

Dataset: FuelConsumptionCo2.csv
Model: Linear Regression (saved as model.pkl)
"""

from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load the pre-trained model at startup
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
with open(model_path, "rb") as f:
    regr = pickle.load(f)


@app.route("/", methods=["GET", "POST"])
def home():
    """Render the prediction form and handle POST requests."""
    if request.method == "POST":
        # Extract engine size from form data
        engine_size = float(request.form["Engine"])

        # Predict CO2 emission using the loaded model
        # Model expects a 2D array: [[engine_size]]
        prediction = regr.predict([[engine_size]])[0]

        # Round to 2 decimal places for display
        return render_template("result.html", EMI=round(prediction, 2))

    # GET request: show the input form
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)

