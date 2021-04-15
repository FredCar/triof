import pytest
import requests
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import *
from src import config
import azure


def test_load_random_image():
    img = load_random_image()

    assert type(img) == str
    assert img[-3:] in ["jpg", "png", "peg"]


def test_azure_endpoint():
    result = requests.get(config.ENDPOINT)
    assert result.status_code == 200


def test_request_azure_api():
    img = load_random_image()
    prediction = request_azure_api(img)

    # with pytest.raises("azure.cognitiveservices.vision.customvision.prediction.models._models_py3.CustomVisionErrorException"):
    #         request_azure_api(img)

    assert prediction == 1


def test_preprocess_image():
    img = load_random_image()
    img = preprocess_image(img)
    assert type(img) == np.ndarray
    assert img.shape == (1, 150, 150, 3)
