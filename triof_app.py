from flask import Flask, render_template, request
from src.utils import *
import os
import random
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/start')
def insert():
    # Load a random image
    img_list = os.listdir("static/camera")
    img = random.choice(img_list)

    open_waste_slot()

    return render_template('insert.html', data=img)



@app.route('/waste/pick-type', methods=["GET"])
def pick_type():
    if request.values:
        img = request.values["image"]

        # Connect to Azure API
        ENDPOINT = "https://triofcv.cognitiveservices.azure.com/"
        prediction_key = "45a365eacbd74858ab02810d7a72c89c"
        publish_iteration_name = "Iteration2"
        projectId = "72e7b78b-9edd-4dd2-a90d-7870228699f3"

        prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
        predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

        with open(f"static/camera/{img}", "rb") as image_contents:
            results = predictor.classify_image(
                projectId, publish_iteration_name, image_contents.read())
            result = {}
            for prediction in results.predictions:
                result[prediction.tag_name] = prediction.probability

        result = sorted(result, key=lambda item: item[1])[0]
        print("=========================================")
        print(result)
        print("=========================================")

    close_waste_slot()

    return render_template('type.html', img=img, result=result)


@app.route('/confirmation', methods=['POST'])
def confirmation():
    waste_type = request.form['type']

    process_waste(waste_type)
    return render_template('confirmation.html')



if __name__ == "__main__":
    app.run(debug=True)
