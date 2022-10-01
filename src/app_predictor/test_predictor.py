import os

import cv2

from src.app_predictor.app_predictor import blur
from src.app_predictor.app_predictor import predict
from src.app_predictor.app_predictor import Rect

IMAGE_IN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/sample-0.png")
IMAGE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data/sample-0_out.png")


def test_predict():
    img = cv2.imread(IMAGE_IN)
    faces = predict(img)
    assert len(faces) >= 9


def test_blur():
    img = cv2.imread(IMAGE_IN)
    img_after_blur = blur(img, [Rect(*[0, 0, 10, 10])])
    assert (img != img_after_blur).any()


def test_blur_empty():
    img = cv2.imread(IMAGE_IN)
    img_after_blur = blur(img, [])
    assert (img == img_after_blur).all()
