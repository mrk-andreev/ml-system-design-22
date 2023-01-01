import abc
import os
import tempfile
import typing
import warnings
from typing import List
from typing import Optional
from typing import Tuple

import mlflow
import numpy as np


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
        import cv2
        import torch
        import torchvision.transforms as transforms
        from bisenet import BiSeNet

        self._face_cascade = cv2.CascadeClassifier(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/haarcascade_frontalface_default.xml")
        )

        net = BiSeNet(n_classes=19)
        save_pth = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/bisenet.pth")
        net.load_state_dict(torch.load(save_pth))
        net.eval()

        self._net = net
        self._to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)

    def predict_from_picture(self, img_copy: List[List[int]]) -> Tuple[List[List[int]], List[Rect]]:
        import cv2
        import torch
        from PIL import Image

        img_copy = img_copy.copy()
        gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            img = img_copy[y:y + h, x:x + w, :]
            with torch.no_grad():
                img = Image.fromarray(img)
                img_size = img.size
                image = img.resize((512, 512), Image.BILINEAR)
                img = self._to_tensor(image)
                img = torch.unsqueeze(img, 0)
                img = img.cuda()
                out = self._net(img)[0]
                parsing = out.squeeze(0).cpu().numpy().argmax(0)

            points = []
            for row in range(parsing.shape[0]):
                for col in range(parsing.shape[1]):
                    if parsing[row][col] != 0 and parsing[row][col] < 14:
                        points += [[row, col]]
            points = np.asarray(points)

            im = np.array(image)
            im_blur = im.copy()
            im_blur[:, :] = cv2.blur(im[:, :], (100, 100))

            final = im.copy()
            for i in points:
                for j in i:
                    final[i, j] = im_blur[i, j]

            final = Image.fromarray(final)
            final = final.resize(img_size, Image.BILINEAR)
            final = np.array(final)

            img_copy[y:y + h, x:x + w, :] = final.copy()
            return img_copy, faces


class YoloPredictor(BaseMlFlowModel):
    def __init__(self):
        from yolov5.models.common import DetectMultiBackend

        self._model: Optional[DetectMultiBackend] = None

    @property
    def name(self):
        return "yolov5"

    def load_context(self, context):
        from yolov5.models.common import DetectMultiBackend

        with tempfile.TemporaryDirectory() as tdir:
            weight_path = os.path.join(tdir, "model.pt")
            with open(context.artifacts['weight'], 'rb') as f_in, open(weight_path, 'wb') as f_out:
                f_out.write(f_in.read())
            self._model = DetectMultiBackend(weights=weight_path)

    def predict(self, context, model_input):
        return self.predict_from_picture(model_input)

    def predict_from_picture(self, img: List[List[int]]) -> Tuple[List[List[int]], List[Rect]]:
        import torch
        from yolov5.utils.augmentations import letterbox
        from yolov5.utils.general import non_max_suppression
        from yolov5.utils.general import scale_boxes

        img = np.asarray(img)
        stride, names, pt = self._model.stride, self._model.names, self._model.pt
        imgsz = (640, 640)
        conf_thres = 0.25  # confidence threshold
        iou_thres = 0.5  # NMS IOU threshold
        classes = None
        agnostic_nms = False  # class-agnostic NMS
        max_det = 1000  # maximum detections per image
        bs = 1
        self._model.warmup(imgsz=(1 if pt or self._model.triton else bs, 3, *imgsz))  # warmup

        im = letterbox(img, imgsz, stride=stride, auto=pt)[0]  # padded resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)  # contiguous
        im = torch.from_numpy(im).to(self._model.device)
        im = im.half() if self._model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self._model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        for i, det in enumerate(pred):  # per image
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img.shape).round()

        y_pred = np.array(det[:, :4]).astype(int)
        ret: List[Rect] = []
        for i in range(y_pred.shape[0]):
            ret += [
                Rect(
                    x=int(y_pred[i][0]),
                    y=int(y_pred[i][1]),
                    w=int(y_pred[i][2] - y_pred[i][0]),
                    h=int(y_pred[i][3] - y_pred[i][1])
                )
            ]

        return blur(img, ret), ret


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

        img[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w, :] = cv2.blur(
            img[rect.y: rect.y + rect.h, rect.x: rect.x + rect.w, :], ksize=(rect.w // 2, rect.h // 2)
        )

    return img
