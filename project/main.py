import os
# run ""nvidia-smi" in terminal
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from cv2 import cv2
import numpy as np
import tensorflow as tf
# need yolov3 folder to be in the same folder as main.py
from yolov3.yolov3 import Create_Yolov3
from yolov3.utils import load_yolo_weights, detect_image, detect_video, detect_realtime, CreateCSV
from yolov3.configs import *

### to fix "cuDNN failed to initialize" error
physical_devices = tf.config.experimental.list_physical_devices('GPU') 
for physical_device in physical_devices: 
    tf.config.experimental.set_memory_growth(physical_device, True)


input_size = YOLO_INPUT_SIZE
Darknet_weights = YOLO_DARKNET_WEIGHTS
if TRAIN_YOLO_TINY:
    Darknet_weights = YOLO_DARKNET_TINY_WEIGHTS


yolo = Create_Yolov3(input_size=input_size, CLASSES=TRAIN_CLASSES)
yolo.load_weights("./checkpoints/yolov3_custom") # use keras weights

detect_realtime(yolo, '', input_size=input_size, show=True, CLASSES=TRAIN_CLASSES, rectangle_colors=(255, 0, 0))
#CreateCSV() # call to create CSV
