import os
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import non_max_suppression
from yolov5.utils.general import Profile
from yolov5.utils.general import scale_boxes


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


DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset/')


def evaluate_score():
    weight_path = os.path.join(os.path.dirname(__file__), '../sample_weight/face_detection_yolov5s.pt')
    model = DetectMultiBackend(weights=weight_path)
    stride, names, pt = model.stride, model.names, model.pt

    imgsz = (640, 640)
    conf_thres = 0.25  # confidence threshold
    iou_thres = 0.5  # NMS IOU threshold
    classes = None
    agnostic_nms = False  # class-agnostic NMS
    max_det = 1000  # maximum detections per image

    img_dir = os.path.join(DATASET_DIR, 'WIDER_val/images/')
    labels = pd.read_csv(os.path.join(DATASET_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt'), header=None)

    sum_iou = 0
    cnt = 0
    for filename in tqdm(glob(img_dir + '*/*.jpg')):
        name = filename[len(img_dir):]

        # Dataloader
        bs = 1  # batch_size
        dataset = LoadImages(filename, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

        # Predict
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                # visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
                pred = model(im, augment=False, visualize=False)

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1

                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

        # fact
        ind_lab = labels[labels[0] == name].index[0]
        num_faces = int(labels.iloc[ind_lab + 1][0])
        y_true = np.array(list(labels.iloc[ind_lab + 2:ind_lab + num_faces + 2][0] \
                               .apply(lambda x: x.split()[:4]).apply(lambda x: [int(i) for i in x])))

        # correct for metric
        for i in range(len(y_true)):
            y_true[i][2] = y_true[i][2] + y_true[i][0]
            y_true[i][3] = y_true[i][3] + y_true[i][1]
        y_pred = np.array(det[:, :4]).astype(int)

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
    score = evaluate_score()
    print(score)


if __name__ == '__main__':
    main()
