import abc
from typing import List

import numpy as np
import torch

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.augmentations import letterbox
from yolov5.utils.general import non_max_suppression
from yolov5.utils.general import scale_boxes


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


class BasePredictor(abc.ABC):
    @abc.abstractmethod
    def predict(self, img: List[List[int]]) -> List[Rect]:
        pass


class YoloPredictor(BasePredictor):
    def __init__(self, weight_path):
        self._model = DetectMultiBackend(weights=weight_path)

    def predict(self, img: List[List[int]]) -> List[Rect]:
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
        return ret
