import cv2
import numpy as np

image_files = [
    r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016157.jpg',
    r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016158.jpg',
    r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016171.jpg',
    r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016175.jpg'
]


def find_frame_difference(frame1_path, frame2_path):
    frame1 = cv2.imread(frame1_path)
    frame2 = cv2.imread(frame2_path)
    hd_resolution = (1280, 720)
    frame1 = cv2.resize(frame1, hd_resolution)
    frame2 = cv2.resize(frame2, hd_resolution)
    hsv_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
    hsv_frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2HSV)

    difference = cv2.absdiff(hsv_frame1, hsv_frame2)

    _, diff_h = cv2.threshold(difference[:, :, 0], 50, 80, cv2.THRESH_BINARY)
    _, diff_s = cv2.threshold(difference[:, :, 1], 50, 80, cv2.THRESH_BINARY)
    _, diff_v = cv2.threshold(difference[:, :, 2], 50, 80, cv2.THRESH_BINARY)

    total_diff = cv2.bitwise_or(diff_h, diff_s)
    total_diff = cv2.bitwise_or(total_diff, diff_v)

    kernel = np.ones((3, 3), np.uint8)
    total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_OPEN, kernel)
    total_diff = cv2.morphologyEx(total_diff, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Difference", total_diff)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


find_frame_difference(r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016157.jpg', r'C:/Users/druzh/work_flow_python/yolo_data/train/G0016175.jpg')
