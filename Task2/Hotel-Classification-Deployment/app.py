from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    scaler, model = pickle.load(f)
# model = pickle.load(open("model.pkl", "rb"))


@app.route("/")
def Home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    user_inputs_scaled = scaler.transform([float_features])
    prediction = model.predict(user_inputs_scaled)
    if request.method == 'POST':
        if (format(prediction) == '[0]'):
            pred = "Not going to be Canceled"
        else:
            pred = "Going to be Canceled"
    return render_template("index.html", prediction_text="This Booking's prediction is {}".format(pred))


if __name__ == "__main__":
    app.run(debug=True)
