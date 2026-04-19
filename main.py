import os, json, shutil, argparse, random
import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO
from sklearn.model_selection import train_test_split


WORKSPACE = "/data/workspace"

TRAIN_IMG_ROOT = f"{WORKSPACE}/images/train"
TEST_IMG_ROOT  = f"{WORKSPACE}/images/test"

TRAIN_JSON = f"{WORKSPACE}/train.json"
TEST_JSON  = f"{WORKSPACE}/test.json"

TRAINSET_ROOT = f"{WORKSPACE}/yolo_dataset_train"
TESTSET_ROOT  = f"{WORKSPACE}/yolo_dataset_test"
PRED_ROOT     = f"{WORKSPACE}/yolo_test_predictions"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def imread_unicode(path, flags=cv2.IMREAD_COLOR):
    data = np.fromfile(path, dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def build_ann_map(json_path):
    data = json.load(open(json_path, "r"))
    images = data["images"]
    anns = data["annotations"]
    categories = data["categories"]

    cat_id_map = {c["id"]: i for i, c in enumerate(categories)}

    ann_map = {}
    for ann in anns:
        ann_map.setdefault(ann["image_id"], []).append({
            "bbox": ann["bbox"],
            "cls": cat_id_map[ann["category_id"]]
        })

    return images, ann_map


def convert_train_json_to_yolo(json_path, img_root, save_root, split_ratio=0.9):
    images, ann_map = build_ann_map(json_path)
    real_files = set(os.listdir(img_root))

    imgs, image_level_labels = [], []
    for img in images:
        fname = img["file_name"].strip()
        if fname in real_files:
            imgs.append(img)
            cls_list = [a["cls"] for a in ann_map.get(img["id"], [])]
            image_level_labels.append(1 if 1 in cls_list else 0)

    train_imgs, val_imgs = train_test_split(
        imgs,
        test_size=1 - split_ratio,
        stratify=image_level_labels,
        random_state=42
    )

    def save_item(img_item, split):
        fname = img_item["file_name"].strip()
        src = os.path.join(img_root, fname)
        img = imread_unicode(src)
        if img is None:
            return

        h, w = img.shape[:2]

        img_dir = os.path.join(save_root, f"images/{split}")
        lab_dir = os.path.join(save_root, f"labels/{split}")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        shutil.copy(src, os.path.join(img_dir, fname))

        base, _ = os.path.splitext(fname)
        with open(os.path.join(lab_dir, base + ".txt"), "w") as f:
            for ann in ann_map.get(img_item["id"], []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{ann['cls']} {cx} {cy} {nw} {nh}\n")

    for item in train_imgs:
        save_item(item, "train")
    for item in val_imgs:
        save_item(item, "val")


def convert_test_json_to_yolo(json_path, img_root, save_root):
    images, ann_map = build_ann_map(json_path)
    real_files = set(os.listdir(img_root))

    for img in images:
        fname = img["file_name"].strip()
        if fname not in real_files:
            continue

        src = os.path.join(img_root, fname)
        img_data = imread_unicode(src)
        if img_data is None:
            continue

        h, w = img_data.shape[:2]

        img_dir = os.path.join(save_root, "images/test")
        lab_dir = os.path.join(save_root, "labels/test")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lab_dir, exist_ok=True)

        shutil.copy(src, os.path.join(img_dir, fname))

        base, _ = os.path.splitext(fname)
        with open(os.path.join(lab_dir, base + ".txt"), "w") as f:
            for ann in ann_map.get(img["id"], []):
                x, y, bw, bh = ann["bbox"]
                cx = (x + bw / 2) / w
                cy = (y + bh / 2) / h
                nw = bw / w
                nh = bh / h
                f.write(f"{ann['cls']} {cx} {cy} {nw} {nh}\n")


def create_yaml(root, split):
    if split == "train":
        content = f"""
train: {os.path.abspath(root + "/images/train")}
val: {os.path.abspath(root + "/images/val")}
nc: 2
names: ["normal", "anomaly"]
"""
    else:
        content = f"""
path: {os.path.abspath(root)}
train: images/test
val: images/test
test: images/test
nc: 2
names: ["normal", "anomaly"]
"""
    open(os.path.join(root, "data.yaml"), "w").write(content)


def train_yolo(model_pt, epochs):
    model = YOLO(model_pt)
    model.train(
        data=os.path.join(TRAINSET_ROOT, "data.yaml"),
        epochs=epochs,
        imgsz=640,
        batch=8,
        optimizer="SGD",
        lr0=0.0008,
        momentum=0.95,
        weight_decay=0.0005,
        warmup_epochs=3,
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.02,
        hsv_s=0.03,
        hsv_v=0.03,
        translate=0.05,
        scale=0.1,
        fliplr=0.2,
        flipud=0.05,
        multi_scale=False
    )


def evaluate_test(weights):
    model = YOLO(weights)
    metrics = model.val(
        data=os.path.join(TESTSET_ROOT, "data.yaml"),
        imgsz=640,
        conf=0.25,
        device=0 if torch.cuda.is_available() else "cpu"
    )

    mAP = float(metrics.box.map)
    precision = float(metrics.box.mp)
    recall = float(metrics.box.mr)

    final_score = mAP * 0.8 + precision * 0.1 + recall * 0.1

    result = {
        "mAP50-95": mAP,
        "Precision": precision,
        "Recall": recall,
        "FinalScore": final_score
    }

    os.makedirs(PRED_ROOT, exist_ok=True)
    pd.DataFrame([result]).to_csv(
        os.path.join(PRED_ROOT, "test_metrics.csv"),
        index=False
    )

    print("\n===== TEST RESULT =====")
    print(f"mAP50–95 : {mAP}")
    print(f"Precision: {precision}")
    print(f"Recall   : {recall}")
    print(f"Final Score: {final_score}")
    print("=======================\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["train", "test"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weights", default=None)
    parser.add_argument("--model", default="yolov8x.pt")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.mode == "train":
        if os.path.exists(TRAINSET_ROOT):
            shutil.rmtree(TRAINSET_ROOT)
        os.makedirs(TRAINSET_ROOT)

        convert_train_json_to_yolo(TRAIN_JSON, TRAIN_IMG_ROOT, TRAINSET_ROOT)
        create_yaml(TRAINSET_ROOT, "train")
        train_yolo(args.model, args.epochs)

    else:
        if not args.weights:
            raise ValueError("--weights is required in test mode")

        if os.path.exists(TESTSET_ROOT):
            shutil.rmtree(TESTSET_ROOT)
        os.makedirs(TESTSET_ROOT)

        convert_test_json_to_yolo(TEST_JSON, TEST_IMG_ROOT, TESTSET_ROOT)
        create_yaml(TESTSET_ROOT, "test")
        evaluate_test(args.weights)
