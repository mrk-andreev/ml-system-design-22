import os
from glob import glob
from shutil import copy

import cv2
import numpy as np
from tqdm import tqdm

YOLOV5_DIR = os.path.join(os.path.dirname(__file__), '../yolov5/')
DATASET_DIR = os.path.join(os.path.dirname(__file__), '../dataset/')


def export_dataset(task_name, output_val_img_dir, output_val_labels_dir, val_labels_path, val_img_path):
    os.makedirs(output_val_img_dir, exist_ok=True)
    os.makedirs(output_val_labels_dir, exist_ok=True)

    with open(val_labels_path) as annots:
        lines = annots.readlines()

    names = [x for x in lines if 'jpg' in x]
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
):
    from yolov5.train import run

    run(
        data=index_path,
        imgsz=img_size,
        batch=batch,
        workers=workers,
        epochs=epochs,
        weights=weights,
        cfg=cfg,
    )


def main():
    np.random.seed(101)

    export_dataset(
        "export_val",
        output_val_img_dir='/tmp/newDataset/images/val',
        output_val_labels_dir='/tmp/newDataset/labels/val',
        val_labels_path=os.path.join(DATASET_DIR, 'wider_face_split/wider_face_val_bbx_gt.txt'),
        val_img_path=os.path.join(DATASET_DIR, 'WIDER_val/images')
    )

    export_dataset(
        "export_train",
        output_val_img_dir='/tmp/newDataset/images/train',
        output_val_labels_dir='/tmp/newDataset/labels/train',
        val_labels_path=os.path.join(DATASET_DIR, 'wider_face_split/wider_face_train_bbx_gt.txt'),
        val_img_path=os.path.join(DATASET_DIR, 'WIDER_train/images')
    )

    prepare_images('/tmp/newDataset/images/*', target_width=640)

    prepare_index(
        index_path='/tmp/yolov5/data/dataset.yaml',
        train_img_path='/tmp/newDataset/images/train',
        val_img_path='/tmp/newDataset/images/val'
    )

    train(
        index_path='/tmp/yolov5/data/dataset.yaml',
        img_size=640,
        batch=64,
        workers=1,
        epochs=1,
        weights='/tmp/yolov5m.pt',
        cfg=os.path.join(YOLOV5_DIR, 'models/yolov5s.yaml'),
    )


if __name__ == '__main__':
    main()
