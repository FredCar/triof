import os
import random
import numpy as np
from matplotlib.image import imread

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array

from src import config

def open_waste_slot():

    """
        open the machine so that
        an user can enter the machine

    :return:
    """

    send_command_to_machine("open_waste_slot")
    return True


def close_waste_slot():
    """
    close the waste box for user safety
    :return:
    """

    send_command_to_machine("close_waste_slot")
    return True


def process_waste(waste_type):

    """
    move the good slot and shredd the waste
    :return:
    """

    move_container(waste_type)
    was_sucessful = shred_waste()

    return was_sucessful


def move_container(waste_type):

    BOTTLE_BOX = 0
    GLASS_BOX = 1
    command_name = "move_container"

    if waste_type == "bottle":
        send_command_to_machine(command_name, BOTTLE_BOX)
    elif waste_type == "glass":
        send_command_to_machine(command_name, GLASS_BOX)

    return True


def send_command_to_machine(command_name, value=None):

    """
    simulate command sending to rasberry pi
    do nothing to work even if the machine is not connected

    :param command_name:
    :param value:
    :return:
    """
    return True


def shred_waste():

    send_command_to_machine("shred_waste")

    return True


def take_trash_picture():

    """
        function simulating the picture taking
        inside the machine. 

        Call this function to ask the machine to 
        take picture of the trash

        return : np array of the picture
    """

    send_command_to_machine("take_picture")

    paths = os.listdir('camera')
    path = random.choice(paths)

    return imread(os.path.join("./camera", path))


def load_random_image(path="static/camera"):
    
    """
    function who load a random image from a local directory
    """

    img_list = os.listdir(path)
    img = random.choice(img_list)

    return img


def request_azure_api(image_name):

    """
    function who requests the Azure Custom Vision API
    and returns the classe prediction
    """

    # Parameters
    ENDPOINT = config.ENDPOINT
    prediction_key = config.PREDICTION_KEY
    publish_iteration_name = "Iteration1"
    projectId = config.PROJECT_ID

    prediction_credentials = ApiKeyCredentials(in_headers={"Prediction-key": prediction_key})
    predictor = CustomVisionPredictionClient(ENDPOINT, prediction_credentials)

    # Request
    with open(f"static/camera/{image_name}", "rb") as image_contents:
        results = predictor.classify_image(projectId, publish_iteration_name, image_contents.read())
        result = {}
        for prediction in results.predictions:
            result[prediction.tag_name] = prediction.probability

    # Sort result
    result = sorted(result, key=lambda item: item[1])[0]

    return result


def preprocess_image(image):

    """
    function which applies some preprocessing to the image
    """

    image = load_img(path=f"static/camera/{image}", target_size=(150,150,3))
    image = img_to_array(image).astype('float32')/255
    image = np.expand_dims(image, axis=0)

    return image