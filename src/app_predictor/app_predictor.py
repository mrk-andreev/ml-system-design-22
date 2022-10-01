import os
import typing
import warnings

import cv2

CACHE = {}


class Rect:
    __slots__ = ('x', 'y', 'w', 'h')

    def __init__(self, x: int, y: int, w: int, h: int):
        self.x = x
        self.y = y
        self.w = w
        self.h = h


def predict(img) -> typing.List[Rect]:
    if "MODEL" not in CACHE:
        CACHE["MODEL"] = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
        )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return [
        Rect(*f)
        for f in CACHE["MODEL"].detectMultiScale(gray, 1.1, 4)
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
    img = blur(img, faces)

    cv2.imwrite(desc_filename, img)


def main():
    pass


if __name__ == '__main__':
    main()
