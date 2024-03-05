import cv2
from ultralytics import YOLO


def resize(img):
    resized_img = cv2.resize(img, (720, 480))
    img1 = resized_img[180:, ::]
    return img1


def procces(img):
    model = YOLO('best.pt')
    img1 = resize(img)
    result = model.predict(img1, verbose=False)[0]
    if len(result.boxes.conf) == 0:
        result = {
            "filename": "",
            "id": [],
            "bbox": [],
            "conf": [],
            "classes": []
        }
    else:
        bbox_data = []
        for bbox in result.boxes.xyxy.cpu().detach().numpy():
            bbox_data.append([bbox[0] / 720, (bbox[1]) / 480, bbox[2] / 720, (bbox[3]) / 480])
        result = {
            "filename": "",
            "id": [i for i in range(len(result.boxes.conf))],
            "bbox": bbox_data,
            "conf": result.boxes.conf.cpu().detach().numpy(),
            "classes": result.boxes.cls
        }
    return result, img1