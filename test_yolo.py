import cv2
import numpy as np
from analyzer.yolo_detector import get_detector

img = np.zeros((480, 640, 3), dtype=np.uint8)
# We need to make sure the models exist at these paths!
import os
print("Model A exists?", os.path.exists("/home/vision/work/aura/models/hospital_hds/weights/best.pt"))
print("Model B exists?", os.path.exists("/home/vision/work/aura/models/hospital_yolo/weights/best.pt"))

detector = get_detector()
res = detector.detect(img)
print(res["obs"])
