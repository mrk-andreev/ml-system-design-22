import os
from glob import glob

import cv2
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from experiments.v2_tracking.src.base import BaseModel

DATA_DIR = os.path.join(os.path.dirname(__file__), '../data')


class DatasetIterator:
    def __init__(self, labels, dataset_dir):
        self._labels = labels
        self._dataset_dir = dataset_dir
        self._idx = -1
        self._files = glob(os.path.join(dataset_dir, '*/*.jpg'))

    def _load_ans(self, filename):
        name = filename[len(self._dataset_dir):]
        ind_lab = self._labels[self._labels[0] == name].index[0]
        num_faces = int(self._labels.iloc[ind_lab + 1][0])
        return np.array(list(self._labels.iloc[ind_lab + 2:ind_lab + num_faces + 2][0] \
                             .apply(lambda x: x.split()[:4]).apply(lambda x: [int(i) for i in x])))

    @classmethod
    def _load_img(cls, filename):
        return cv2.imread(filename)

    def __next__(self):
        if self._idx == len(self._files):
            raise StopIteration()
        self._idx += 1
        filename = self._files[self._idx]

        return self._load_img(filename), self._load_ans(filename)

    def __len__(self):
        return len(self._files)


class Dataset:
    def __init__(self, dataset_dir, labels_path):
        self._labels = pd.read_csv(labels_path, header=None)
        self._dataset_dir = dataset_dir

    def __iter__(self):
        return DatasetIterator(self._labels, self._dataset_dir)

    def __len__(self):
        return len(DatasetIterator(self._labels, self._dataset_dir))


class IouMetric:
    def __init__(self):
        self._sum_iou = 0
        self._cnt = 0

    @classmethod
    def _get_iou(cls, box_a, box_b):
        """
            Calculate the Intersection over Union (IoU) of two bounding boxes.
            Parameters
            ----------
            boxA = np.array( [ xmin,ymin,xmax,ymax ] )
            boxB = np.array( [ xmin,ymin,xmax,ymax ] )
            Returns
            -------
            float
            in [0, 1]
            """

        bb1 = dict()
        bb1['x1'] = box_a[0]
        bb1['y1'] = box_a[1]
        bb1['x2'] = box_a[2]
        bb1['y2'] = box_a[3]

        bb2 = dict()
        bb2['x1'] = box_b[0]
        bb2['y1'] = box_b[1]
        bb2['x2'] = box_b[2]
        bb2['y2'] = box_b[3]

        # Determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            # print('check')
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # Compute the area of both bounding boxes area
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # Compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the intersection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)

        assert iou >= 0.0
        assert iou <= 1.0

        return iou

    def update(self, y_true, y_pred):
        """Note: function with side effect"""
        self._correct_predict(y_true, y_pred)
        total_iou = 0
        pred_dict = dict()
        for gt in y_true:
            max_iou_per_gt = 0
            for i, pred in enumerate(y_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = self._get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt
        iou = total_iou / len(y_true)

        self._sum_iou += iou
        self._cnt += 1

    def evaluate(self):
        return self._sum_iou / self._cnt if self._cnt != 0 else 0

    @classmethod
    def _correct_predict(cls, y_true, y_pred):
        for i in range(len(y_pred)):
            y_pred[i][2] = y_pred[i][2] + y_pred[i][0]
            y_pred[i][3] = y_pred[i][3] + y_pred[i][1]
        for i in range(len(y_true)):
            y_true[i][2] = y_true[i][2] + y_true[i][0]
            y_true[i][3] = y_true[i][3] + y_true[i][1]


def _score_model(model_cls, weight_path, is_dry_run):
    artifacts = {
        'weight': weight_path
    }
    model = model_cls()

    mlflow.pyfunc.log_model(
        artifact_path="py_model",
        python_model=model,
        artifacts=artifacts,
        registered_model_name=model.name,
    )

    context = mlflow.pyfunc.PythonModelContext(artifacts)
    model.load_context(context)

    if is_dry_run:
        mlflow.log_param("is_dry_run", is_dry_run)

    dataset = Dataset(
        os.path.join(DATA_DIR, 'WIDER_val/images/'),
        os.path.join(DATA_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt'),
    )
    metric = IouMetric()

    for img, y_true in tqdm(dataset):
        y_pred = model.predict_from_picture(img)
        metric.update(y_true, y_pred)

        if is_dry_run:
            break

    mlflow.log_metric('iou', metric.evaluate())


def score_model(model_cls, weight_path, is_dry_run=False):
    if not issubclass(model_cls, BaseModel):
        raise ValueError(f"{model_cls} must be subclass of mlflow.pyfunc.PythonModel")

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URL", 'http://localhost:5000'))
    with mlflow.start_run():
        mlflow.autolog()
        _score_model(model_cls, weight_path, is_dry_run)
