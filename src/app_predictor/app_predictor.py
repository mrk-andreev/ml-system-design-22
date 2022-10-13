import abc
import datetime
import json
import logging
import os
import tempfile
import time
import typing
import uuid
import warnings
from json import JSONEncoder
from typing import List

import boto3
import cv2
import redis
import requests
from telegram import Bot
from telegram.inline.inlinekeyboardbutton import InlineKeyboardButton
from telegram.inline.inlinekeyboardmarkup import InlineKeyboardMarkup

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)

logger = logging.getLogger(__name__)
CACHE = {}


class PredictWriter(abc.ABC):
    @abc.abstractmethod
    def save(self, req, predicts):
        pass


class S3PredictWriter(PredictWriter):
    @classmethod
    def init(cls):
        return S3PredictWriter(
            endpoint_url=os.environ["S3_ENDPOINT_URL"],
            access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
            bucket_name=os.environ["S3_BUCKET_NAME"],
            path_prefix=os.environ["S3_PATH_PREFIX"],
            verify=os.environ.get("S3_VERIFY", "false") == "true",
        )

    def __init__(self, endpoint_url, access_key_id, secret_access_key, bucket_name, path_prefix, verify=False):
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,  # 'https://<minio>:9000'
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=None,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=verify,
        )
        self._bucket_name = bucket_name
        self._path_prefix = path_prefix

    def save(self, req, predicts):
        obj_name = f"{self._path_prefix}/{datetime.datetime.now().timestamp()}-{uuid.uuid4()}.json"
        obj = self._s3.Object(self._bucket_name, obj_name)
        obj.put(Body=json.dumps({"req": req, "predicts": predicts}, cls=CustomEncoder))


def init_predict_writer():
    return S3PredictWriter.init()


class Rect:
    __slots__ = ("x", "y", "w", "h")

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __repr__(self):
        return f"Rect({self.x}, {self.y}, {self.w}, {self.h})"

    def __dict__(self):
        return {
            "x": int(self.x),
            "y": int(self.y),
            "w": int(self.w),
            "h": int(self.h),
        }


class CustomEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__()


class BasePredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, img: List[List[int]]) -> List[Rect]:
        pass


class Cv2CascadeClassifierPredictor(BasePredictor):
    def __init__(self, path):
        self._m = cv2.CascadeClassifier(path)

    def predict(self, img: List[List[int]]) -> List[Rect]:
        return [Rect(*f) for f in self._m.detectMultiScale(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 1.1, 4)]


def init_predictor():
    return Cv2CascadeClassifierPredictor(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
    )


def blur(img, rects: typing.List[Rect]):
    img = img.copy()
    for rect in rects:
        if (
            rect.y < 0
            or rect.y > img.shape[0]
            or (rect.y + rect.h) > img.shape[0]
            or rect.x < 0
            or rect.x > img.shape[1]
            or rect.x + rect.w > img.shape[1]
        ):
            warnings.warn("Rect out of image")
            continue

        img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w, :] = cv2.blur(
            img[rect.y : rect.y + rect.h, rect.x : rect.x + rect.w, :], ksize=(rect.w // 2, rect.h // 2)
        )

    return img


def transform(predict_saver, predictor: BasePredictor, req, img):
    faces = predictor.predict(img)
    predict_saver.save(req, faces)

    logger.info(f"Found '{len(faces)}' faces")
    if faces:
        logger.info(f"Rects: '{faces}'")
        return blur(img, faces)

    return img


class DataReceiver:
    def __init__(self, host, port, queue, username, password):
        self._r = redis.Redis(host=host, port=port, db=0, username=username, password=password)
        self._q = queue
        self._r.ping()

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            req: bytes = self._r.lpop(self._q)
            if req is None:
                time.sleep(1)
                continue
            return json.loads(req.decode())


class BlobStorageReader(abc.ABC):
    @abc.abstractmethod
    @property
    def storage(self):
        pass

    @abc.abstractmethod
    def load(self, file_link, dest_filename):
        pass


class RemoteLinkBlobStorageReader(BlobStorageReader):
    @classmethod
    def init(cls, storages_):
        s = RemoteLinkBlobStorageReader()
        storages_[s.storage] = s

    @property
    def storage(self):
        return "REMOTE_LINK"

    def load(self, file_link, dest_filename):
        resp = requests.get(file_link)
        if not resp.ok:
            raise ValueError(resp)

        with open(dest_filename, "wb") as f:
            f.write(resp.content)


class S3BlobStorageReader(BlobStorageReader):
    @classmethod
    def init(cls, storages_):
        try:
            s = S3BlobStorageReader(
                endpoint_url=os.environ["S3_ENDPOINT_URL"],
                access_key_id=os.environ["S3_ACCESS_KEY_ID"],
                secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
                bucket_name=os.environ["S3_BUCKET_NAME"],
                verify=os.environ.get("S3_VERIFY", "false") == "true",
            )
            storages_[s.storage] = s
        except Exception as e:
            logger.info("Skip S3BlobStorageLoader creation.")
            logger.info(e)

    def __init__(self, endpoint_url, access_key_id, secret_access_key, bucket_name, verify=False):
        self._s3 = boto3.resource(
            "s3",
            endpoint_url=endpoint_url,  # 'https://<minio>:9000'
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=None,
            config=boto3.session.Config(signature_version="s3v4"),
            verify=verify,
        )
        self._bucket_name = bucket_name

    @property
    def storage(self):
        return "S3"

    def load(self, file_link, dest_filename):
        bucket, obj_name = file_link["bucket"], file_link["obj_name"]
        with open(dest_filename, "wb") as f:
            f.write(self._s3.Object(bucket, obj_name).get()["Body"].read())


def load_image(req, storages: typing.Dict[str, BlobStorageReader]):
    logger.info(f"Load {req}")
    storage, href = req["img"]["storage"], req["img"]["href"]

    if storage not in storages:
        raise ValueError(f"Unknown storage = {storage}")

    with tempfile.TemporaryDirectory() as tdir:
        fpath = os.path.join(tdir, "sample.jpg")
        storages[storage].load(href, fpath)
        return cv2.imread(fpath)


class DataUploader:
    def __init__(self, token):
        self._b = Bot(token)

    def upload(self, req, img):
        with tempfile.TemporaryDirectory() as tdir:
            fpath = os.path.join(tdir, "sample.jpg")
            cv2.imwrite(fpath, img)

            with open(fpath, "rb") as f:
                self._b.send_photo(
                    chat_id=req["chat_id"],
                    photo=f,
                    reply_to_message_id=req["message_id"],
                    reply_markup=InlineKeyboardMarkup(
                        [
                            [
                                InlineKeyboardButton("👍", callback_data="like"),
                                InlineKeyboardButton("👎", callback_data="dislike"),
                            ]
                        ]
                    ),
                )


def init_data_uploader():
    return DataUploader(os.environ["TELEGRAM_BOT_TOKEN"])


def init_storages():
    storages = {}
    for s in [RemoteLinkBlobStorageReader, S3BlobStorageReader]:
        s.init(storages)
    return storages


def init_data_receiver():
    return DataReceiver(
        host=os.environ["REDIS_HOST"],
        port=os.environ["REDIS_PORT"],
        queue=os.environ["REDIS_QUEUE"],
        username=os.environ.get("REDIS_USERNAME"),
        password=os.environ.get("REDIS_PASSWORD"),
    )


def main():
    data_uploader = init_data_uploader()
    storages = init_storages()
    predict_saver = init_predict_writer()
    predictor = init_predictor()
    data_receiver = init_data_receiver()
    for req in data_receiver:
        try:
            img = load_image(req, storages)
            img_after_transform = transform(predict_saver, predictor, req, img)
            data_uploader.upload(req, img_after_transform)
        except Exception as e:
            logging.error(e)


if __name__ == "__main__":
    main()
