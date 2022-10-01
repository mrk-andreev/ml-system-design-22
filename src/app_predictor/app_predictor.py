import json
import logging
import os
import tempfile
import time
import typing
import warnings

import cv2
import redis
import requests
from telegram import Bot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)

logger = logging.getLogger(__name__)
CACHE = {}


class Rect:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return f"Rect({self.x}, {self.y}, {self.w}, {self.h})"


def predict(img) -> typing.List[Rect]:
    if "MODEL" not in CACHE:
        CACHE["MODEL"] = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
        )
    return [
        Rect(*f)
        for f in CACHE["MODEL"].detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)
    ]


BLUR_KERNEL_SIZE = (30, 30)


def blur(img, rects: typing.List[Rect]):
    img = img.copy()
    for rect in rects:
        if (rect.y < 0
                or rect.y > img.shape[0]
                or (rect.y + rect.h) > img.shape[0]
                or rect.x < 0
                or rect.x > img.shape[1]
                or rect.x + rect.w > img.shape[1]
        ):
            warnings.warn("Rect out of image")
            continue

        img[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w, :] = cv2.blur(
            img[rect.y:rect.y + rect.h, rect.x:rect.x + rect.w, :], BLUR_KERNEL_SIZE)

    return img


def transform(src_filename, desc_filename):
    img = cv2.imread(src_filename)
    faces = predict(img)
    logger.info(f"Found '{len(faces)}' faces")
    if faces:
        logger.info(f"Rects: '{faces}'")
    img = blur(img, faces)

    cv2.imwrite(desc_filename, img)


class DataReceiver:
    def __init__(self):
        self._r = redis.Redis(host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"], db=0)

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            req: bytes = self._r.lpop(os.environ["REDIS_QUEUE"])
            if req is None:
                time.sleep(1)
                continue
            return json.loads(req.decode())


def load_image(req, dest_filename):
    logger.info(f"Load {req}")
    resp = requests.get(req["img"])
    if not resp.ok:
        raise ValueError(resp)

    with open(dest_filename, 'wb') as f:
        f.write(resp.content)


class DataUploader:
    def __init__(self):
        self._b = Bot(os.environ["TELEGRAM_BOT_TOKEN"])

    def upload(self, req, out_file):
        with open(out_file, "rb") as f:
            self._b.send_photo(chat_id=req["chat_id"], photo=f)


def main():
    data_uploader = DataUploader()
    for req in DataReceiver():
        try:
            with tempfile.TemporaryDirectory() as tdir:
                in_file = os.path.join(tdir, "input.jpg")
                out_file = os.path.join(tdir, "output.jpg")

                load_image(req, in_file)
                transform(in_file, out_file)
                data_uploader.upload(req, out_file)
        except Exception as e:
            logging.error(e)


if __name__ == '__main__':
    main()
