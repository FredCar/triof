from flask import Flask, render_template, request
from src.utils import *
from src import config
import os
import random
from tensorflow.keras.models import load_model


import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration

sentry_sdk.init(
    dsn=config.SENTRY_DSN,
    integrations=[FlaskIntegration()],
    traces_sample_rate=1.0
)


app = Flask(__name__)


# Load Clean / Dirty predition model
model = load_model("../modele/Model/Model_dirty_clean_0")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/start')
def insert():
    # Load a random image
    img = load_random_image()

    open_waste_slot()

    return render_template('insert.html', data=img)


@app.route('/waste/pick-type', methods=["GET"])
def pick_type():
    if request.values:
        img = request.values["image"]

        # Preprocess image
        X = preprocess_image(img)

        # Prediction
        pred = model.predict(X)

        # Control and redirection
        if pred < 0.5: # Image is clean
            # Connect to Azure API
            result = request_azure_api(img)

            close_waste_slot()
            return render_template('type.html', img=img, result=result)

        else: # Image is dirty
            result = "dirty"
            return render_template('type.html', img=img, result=result)


@app.route('/confirmation', methods=['POST'])
def confirmation():
    waste_type = request.form['type']

    process_waste(waste_type)
    return render_template('confirmation.html')


@app.route('/debug-sentry')
def trigger_error():
    division_by_zero = 1 / 0


if __name__ == "__main__":
    app.run(debug=True)
