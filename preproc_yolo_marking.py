import os
import cv2
import numpy as np


def annotate_image(image_path):
    i = 0

    image = cv2.imread(image_path)
    clone = image.copy()
    resized_img = cv2.resize(clone, (720, 480))
    img1 = resized_img[180:, ::]
    img2 = img1.copy()
    cv2.imshow('image', img1)
    rectangles = []
    drawing = False

    def click_and_draw(event, x, y, flags, param):
        nonlocal rectangles, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangles.append((x, y))
            drawing = True
        elif event == cv2.EVENT_LBUTTONUP and drawing:
            rectangles[-1] = (rectangles[-1][0], rectangles[-1][1], x, y, 0)
            drawing = False

        if event == cv2.EVENT_RBUTTONDOWN:
            rectangles.append((x, y))
            drawing = True
        elif event == cv2.EVENT_RBUTTONUP and drawing:
            rectangles[-1] = (rectangles[-1][0], rectangles[-1][1], x, y, 1)
            drawing = False

        color = lambda x: (0, 255, 0) if x else (255, 0, 0)

        for rect in rectangles:
            if drawing == False:
                cv2.rectangle(img1, (rect[0], rect[1]), (rect[2], rect[3]), color(rect[4]), 2)

        cv2.imshow("image", img1)

    cv2.setMouseCallback("image", click_and_draw)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            save_annotations(image_path, rectangles, img2)
            break
        elif key == ord("q"):
            cv2.destroyAllWindows()
            break


def save_annotations(image_path, rectangles, image):
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    annotations_dir = "data_for_yolo/annotations"
    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    img_dir = "data_for_yolo/images"
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    img_path = os.path.join(img_dir, f"{image_name}.jpg")
    cv2.imwrite(img_path, image)

    annotation_file = os.path.join(annotations_dir, f"{image_name}.txt")
    with open(annotation_file, "w") as f:
        for rect in rectangles:
            f.write(f"{rect[4]} {round((((rect[2] - rect[0]) // 2) + rect[0]) / 720, 15)} {round((((rect[3] - rect[1]) // 2) + rect[1]) / 300, 15)} {round(((rect[2] - rect[0])) / 720, 15)} {round(((rect[3] - rect[1])) / 300, 15)}\n")

def main():
    images_dir = r"100GOPRO"

    for filename in os.listdir(images_dir):
        if filename.endswith(".JPG") or filename.endswith(".png"):
            image_path = os.path.join(images_dir, filename)

            annotation_file = os.path.join("data_for_yolo/annotations", f"{os.path.splitext(filename)[0]}.txt")
            if os.path.exists(annotation_file):
                continue
            annotate_image(image_path)


if __name__ == "__main__":
    main()
