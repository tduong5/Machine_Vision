from cv2 import cv2
import numpy as np
import os
from PIL import Image
    

dataset_path = "custom_dataset/"

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face.
recognizer = cv2.face.LBPHFaceRecognizer_create()

video_capture = cv2.VideoCapture(0)

if not os.path.exists('./recognizer'):
    os.makedirs('./recognizer')

# function to get the images and label data
def GetFacesWithID(dataset_path):
  imagePaths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path)]
  faces = []
  ids = []
  for imagePath in imagePaths:
    faceImg = Image.open(imagePath).convert('L')
    np_face = np.array(faceImg,'uint8')
    ID = int(os.path.split(imagePath)[-1].split('.')[1])
    faces.append(np_face)
    ids.append(ID)
    cv2.imshow("training", np_face)
    cv2.waitKey(10)
  return np.array(ids), faces


ids, faces = GetFacesWithID(dataset_path)
recognizer.train(faces, ids)
recognizer.save('recognizer/face_training.yml')
cv2.destroyAllWindows()