import os
import tempfile
from glob import glob
from shutil import copy

import cv2
import mlflow
import numpy as np
from tqdm import tqdm

from predict import YoloPredictor
from model_utils import download_dataset

YOLOV5_DIR = os.path.join(os.path.dirname(__file__), '../yolov5/')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset/')
TRACKING_URI = os.environ['MLFLOW_TRACKING_URL']
DEVICE = os.environ['DEVICE']


def export_dataset(task_name, output_val_img_dir, output_val_labels_dir, val_labels_path, val_img_path):
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_labels_dir, exist_ok=True)

    with open(val_labels_path) as annots:
        lines = annots.readlines()

    names = [x for x in lines if 'jpg' in x]
    # names = names[:len(names) // 10] # TODO: remove me

    indices = [lines.index(x) for x in names]

    for n in tqdm(range(len(names[:])), desc=task_name):
        i = indices[n]
        name = lines[i].rstrip()
        old_img_path = os.path.join(val_img_path, name)
        name = name.split('/')[-1]
        label_path = os.path.join(output_val_labels_dir, name.split('.')[0] + '.txt')
        img_path = os.path.join(output_val_img_dir, name)

        num_objs = int(lines[i + 1].rstrip())
        bboxs = lines[i + 2: i + 2 + num_objs]
        bboxs = list(map(lambda x: x.rstrip(), bboxs))
        bboxs = list(map(lambda x: x.split()[:4], bboxs))

        img = cv2.imread(old_img_path)
        img_h, img_w, _ = img.shape
        with open(label_path, 'w') as f:
            count = 0  # Num of bounding box
            for bbx in bboxs:
                x1 = int(bbx[0])
                y1 = int(bbx[1])
                w = int(bbx[2])
                h = int(bbx[3])
                #     #yolo:
                x = (x1 + w // 2) / img_w
                y = (y1 + h // 2) / img_h
                w = w / img_w
                h = h / img_h
                if w * h * 100 > 2:
                    yolo_line = f'{0} {x} {y} {w} {h}\n'
                    f.write(yolo_line)
                    count += 1
        if count > 0:
            copy(old_img_path, img_path)
        else:
            os.remove(label_path)


def _resize_img(input_name, output_name, target_width):
    im = cv2.imread(input_name)
    h, w, _ = im.shape
    target_height = int(h / w * target_width)
    im = cv2.resize(im, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite(output_name, im)


def prepare_images(imgs_dir, target_width=640):
    names = glob(os.path.join(imgs_dir, '*'))
    for img in tqdm(names, desc='prepare_images'):
        _resize_img(img, img, target_width)


def prepare_index(index_path, train_img_path, val_img_path):
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    with open(index_path, 'w') as f:
        f.write(f'train: {train_img_path}')
        f.write(f'\nval: {val_img_path}')
        f.write('\nnc: {}'.format(1))
        f.write("\nnames: ['Face']")


def train(
        index_path,
        img_size,
        batch,
        workers,
        epochs,
        weights,
        cfg,
        device,
):
    from yolov5.train import run

    os.makedirs(os.path.dirname(weights), exist_ok=True)

    run(
        data=index_path,
        imgsz=img_size,
        batch=batch,
        workers=workers,
        epochs=epochs,
        weights=weights,
        cfg=cfg,
        device=device
    )

    artifacts = {
        'weight': weights
    }
    predictor = YoloPredictor()
    context = mlflow.pyfunc.PythonModelContext(artifacts)
    predictor.load_context(context)
    mlflow.pyfunc.log_model(
        artifact_path="py_data",
        python_model=predictor,
        artifacts=artifacts,
        registered_model_name="yolov5",
    )


def run(workdir, device):
    np.random.seed(101)

    download_dataset()

    export_dataset(
        "export_val",
        output_val_img_dir=f'{workdir}/newDataset/images/val',
        output_val_labels_dir=f'{workdir}/newDataset/labels/val',
        val_labels_path=os.path.join(DATASET_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt'),
        val_img_path=os.path.join(DATASET_DIR, 'WIDER_val/images')
    )

    export_dataset(
        "export_train",
        output_val_img_dir=f'{workdir}/newDataset/images/train',
        output_val_labels_dir=f'{workdir}/newDataset/labels/train',
        val_labels_path=os.path.join(DATASET_DIR, 'wider_face_split/wider_face_train_bbx_gt.txt'),
        val_img_path=os.path.join(DATASET_DIR, 'WIDER_train/images')
    )

    prepare_images(f'{workdir}/newDataset/images/*', target_width=640)

    prepare_index(
        index_path=f'{workdir}/yolov5/data/dataset.yaml',
        train_img_path=f'{workdir}/newDataset/images/train',
        val_img_path=f'{workdir}/newDataset/images/val'
    )

    train(
        index_path=f'{workdir}/yolov5/data/dataset.yaml',
        img_size=640,
        batch=64,
        workers=1,
        epochs=1,
        weights=f'{workdir}/output/yolov5m.pt',
        cfg=os.path.join(YOLOV5_DIR, 'models/yolov5n.yaml'),
        device=device,
    )


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    with mlflow.start_run():
        mlflow.autolog()
        with tempfile.TemporaryDirectory() as workdir:
            run(workdir, DEVICE)


if __name__ == '__main__':
    main()
