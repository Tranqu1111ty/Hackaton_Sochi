import cv2
from ultralytics import YOLO
import os
import pickle


def resize(image_path):
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (720, 480))
    img1 = resized_img[180:, ::]
    return img1


def procces(images_path):
    model = YOLO('best.pt')
    results = []
    for filename in os.listdir(images_path):
        img = resize(images_path + filename)
        result = model.predict(img, verbose=False)[0]
        if len(result.boxes.conf) == 0:
            result = {
                "filename": filename,
                "id": [],
                "bbox": [],
                "conf": [],
                "classes": []
            }
        else:
            bbox_data = []
            for bbox in result.boxes.xywh.cpu().detach().numpy():
                bbox_data.append([bbox[0] / 720, (bbox[1] + 180) / 480, bbox[2] / 720, bbox[3] / 480])
            result = {
                "filename": filename,
                "id": [i for i in range(len(result.boxes.conf))],
                "bbox": bbox_data,
                "conf": result.boxes.conf.cpu().detach().numpy(),
                "classes": result.boxes.cls
            }

        results.append(result)
    with open("result.pkl", "wb") as f:
        pickle.dump(results, f)


procces('images/')
