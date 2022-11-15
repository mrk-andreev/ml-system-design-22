import os
from glob import glob

import cv2
import mlflow
import numpy as np
import pandas as pd
from tqdm import tqdm

from model_utils import DATASET_DIR
from model_utils import download_dataset

MODEL_NAME = os.environ['MODEL_NAME']
MODEL_STAGE = os.environ['MODEL_STAGE']
TRACKING_URI = os.environ['MLFLOW_TRACKING_URL']


# metric
def get_iou(boxA, boxB):
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
    bb1['x1'] = boxA[0]
    bb1['y1'] = boxA[1]
    bb1['x2'] = boxA[2]
    bb1['y2'] = boxA[3]

    bb2 = dict()
    bb2['x1'] = boxB[0]
    bb2['y1'] = boxB[1]
    bb2['x2'] = boxB[2]
    bb2['y2'] = boxB[3]

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


def evaluate_score():
    predictor = mlflow.pyfunc.load_model(f'models:/{MODEL_NAME}/{MODEL_STAGE}')

    img_dir = os.path.join(DATASET_DIR, 'WIDER_val/images/')
    labels = pd.read_csv(os.path.join(DATASET_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt'), header=None)

    sum_iou = 0
    cnt = 0
    files = glob(img_dir + '*/*.jpg')
    files = files[:len(glob(img_dir + '*/*.jpg')) // 10]  # TODO: remove me
    for filename in tqdm(files):
        name = filename[len(img_dir):]

        y_pred_raw = predictor.predict(cv2.imread(filename))

        # convert y_pred_raw to aliasable format
        y_pred = np.ones((len(y_pred_raw), 4)) if len(y_pred_raw) else np.array([])
        for i in range(len(y_pred_raw)):
            y_pred[i, 0] = y_pred_raw[i].x
            y_pred[i, 1] = y_pred_raw[i].y
            y_pred[i, 2] = y_pred_raw[i].x + y_pred_raw[i].w
            y_pred[i, 3] = y_pred_raw[i].y + y_pred_raw[i].h
        y_pred = y_pred.astype(int)

        # fact
        ind_lab = labels[labels[0] == name].index[0]
        num_faces = int(labels.iloc[ind_lab + 1][0])
        y_true = np.array(list(labels.iloc[ind_lab + 2:ind_lab + num_faces + 2][0] \
                               .apply(lambda x: x.split()[:4]).apply(lambda x: [int(i) for i in x])))

        # correct for metric
        for i in range(len(y_true)):
            y_true[i][2] = y_true[i][2] + y_true[i][0]
            y_true[i][3] = y_true[i][3] + y_true[i][1]

        # evaluate
        total_iou = 0
        pred_dict = dict()
        for gt in y_true:
            max_iou_per_gt = 0
            for i, pred in enumerate(y_pred):
                if i not in pred_dict.keys():
                    pred_dict[i] = 0
                iou = get_iou(gt, pred)
                if iou > max_iou_per_gt:
                    max_iou_per_gt = iou
                if iou > pred_dict[i]:
                    pred_dict[i] = iou
            total_iou = total_iou + max_iou_per_gt
        iou = total_iou / len(y_true)

        sum_iou += iou
        cnt += 1
    total_iou = sum_iou / cnt
    return total_iou


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run():
        mlflow.autolog()
        download_dataset()
        score = evaluate_score()
        mlflow.log_metric('iou', score)


if __name__ == '__main__':
    main()
