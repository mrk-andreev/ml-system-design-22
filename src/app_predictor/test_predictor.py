import os
import time

import cv2
import pytest

from src.app_predictor.app_predictor import Cv2CascadeClassifierPredictor
from src.app_predictor.app_predictor import Rect
from src.app_predictor.app_predictor import blur
from src.app_predictor.predict import BiSeNetPredictor

IMAGE_IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/sample-0.png")
IMAGE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/sample-0_out.png")


@pytest.fixture()
def cv_predictor():
    return Cv2CascadeClassifierPredictor(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
    )


@pytest.fixture()
def bisenet_predictor():
    predictor = BiSeNetPredictor()
    predictor.load_context(None)
    return predictor


def test_cv_predict(cv_predictor):
    img = cv2.imread(IMAGE_IN)
    _, faces = cv_predictor.predict(img)
    assert len(faces) >= 9


def test_bisenet_predict(bisenet_predictor):
    img = cv2.imread(IMAGE_IN)
    _, faces = bisenet_predictor.predict_from_picture(img)
    assert len(faces) >= 9


def test_memory_besinet():
    from memory_profiler import profile

    @profile
    def call():
        predictor = BiSeNetPredictor()
        predictor.load_context(None)
        img = cv2.imread(IMAGE_IN)
        predictor.predict_from_picture(img)

    call()


def test_speed_besinet():
    predictor = BiSeNetPredictor()
    predictor.load_context(None)
    img = cv2.imread(IMAGE_IN)

    def call():
        predictor.predict_from_picture(img)

    def calc():
        t = time.time()
        call()
        return time.time() - t

    print([
        calc()
        for _ in range(10)
    ])


def test_blur():
    img = cv2.imread(IMAGE_IN)
    img_after_blur = blur(img, [Rect(*[0, 0, 10, 10])])
    assert (img != img_after_blur).any()


def test_blur_empty():
    img = cv2.imread(IMAGE_IN)
    img_after_blur = blur(img, [])
    assert (img == img_after_blur).all()
