import abc
import logging
import os
import typing
import warnings
from typing import List
from typing import Tuple

import mlflow
import numpy as np

logger = logging.getLogger(__name__)


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


class BaseMlFlowModel(mlflow.pyfunc.PythonModel, abc.ABC):
    @property
    @abc.abstractmethod
    def name(self):
        pass

    @abc.abstractmethod
    def load_context(self, context):
        pass

    @abc.abstractmethod
    def predict_from_picture(self, img):
        pass

    def predict(self, context, model_input) -> List[Rect]:
        return self.predict_from_picture(model_input)


class BiSeNetPredictor(BaseMlFlowModel):
    def __init__(self):
        self._face_cascade = None
        self._net = None
        self._to_tensor = None

    @property
    def name(self):
        return "biSeNet"

    def load_context(self, context):
        logger.info("Start loading BiSeNetPredictor")
        import cv2
        import torch
        import torchvision.transforms as transforms
        from bisenet import BiSeNet

        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
        )
        torch.jit.enable_onednn_fusion(True)
        torch.set_num_threads(1)
        net = BiSeNet(n_classes=19)
        save_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/bisenet.pth")
        net.load_state_dict(torch.load(save_pth, map_location=torch.device('cpu')))
        net.eval()

        self._net = net
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        sample_input = torch.rand(1, 3, 512, 512)
        traced_model = torch.jit.trace(net, sample_input)
        self._net = torch.jit.freeze(traced_model)
        # warmup
        with torch.no_grad():
            self._net(sample_input)
            self._net(sample_input)
        logger.info("Complete loading BiSeNetPredictor")

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)

    def predict_from_picture(self, img_copy: List[List[int]]) -> Tuple[List[List[int]], List[Rect]]:
        import cv2
        import torch
        from PIL import Image

        img_copy = img_copy.copy()
        faces = self._face_cascade.detectMultiScale(
            cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY),
            scaleFactor=1.1,
            minNeighbors=4
        )

        for (x, y, w, h) in faces:
            img = img_copy[y:y + h, x:x + w, :]
            with torch.no_grad():
                img = Image.fromarray(img)
                img_size = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                img = self._to_tensor(image)
                img = torch.unsqueeze(img, 0)
                out = self._net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

            points = np.asarray(np.where((parsing > 0) & (parsing < 14))).T

            im = np.array(image)
            im_blur = im.copy()
            im_blur[:, :] = cv2.blur(
                im[:, :],
                ksize=(len(im) // 2, len(im[0]) // 2)
            )

            final = im.copy()
            final[points[:, 0], points[:, 1]] = im_blur[points[:, 0], points[:, 1]]
            final = Image.fromarray(final)
            final = final.resize(img_size, Image.BILINEAR)
            final = np.array(final)
            img_copy[y:y + h, x:x + w, :] = final.copy()
        return img_copy, [Rect(*f) for f in faces]


def blur(img, rects: typing.List[Rect]):
    import cv2
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

        img[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w, :] = cv2.blur(
            img[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w, :], ksize=(rect.w // 2, rect.h // 2)
        )

    return img
