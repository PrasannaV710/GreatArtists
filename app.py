from flask import Flask, render_template, request, jsonify
from utils import model_predict
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/predict', methods=['POST'])
def predict():
    pic = request.form.get('content')
    prediction = model_predict(pic)
    return render_template("display.html",prediction=prediction)
    #return render_template("index.html", prediction=prediction, email=email)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)