import cv2
import numpy as np

print("OpenCV:", cv2.__version__)
print("NumPy:", np.__version__)
print("VideoCapture доступен:", hasattr(cv2, "VideoCapture"))
