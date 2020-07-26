# import the necessary packages
from imutils.video import VideoStream
# The argparse module makes it easy to write user-friendly command-line interfaces.
import argparse
# imutils are a series of convenience functions to make basic image processing functions 
import imutils
from cv2 import cv2 # "from cv2" to remove modules not being recognized error
from random import *
import os
from string import digits
import time
import sqlite3

"""
to run: 
$ python face_enrollment.py --output dataset/timmy_duong
windows:
$ .\face_enrollment.py --output dataset/timmy_duong
"""
"""
# cmd lines
argu_parser = argparse.ArgumentParser()
argu_parser.add_argument("-o", "--output", required = True, help = "custom_dataset/face_rec/timmy_duong")
args = vars(argu_parser.parse_args())
"""
# connect to database
connect = sqlite3.connect('user_database.db')
# allow python to execute sql command to a database
cur = connect.cursor()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face.

userName = input("Enter user name: ")

# user name is inserted into the table created, (?) means whatever the input is
cur.execute("INSERT INTO faces (name, balance) VALUES (?, 100)", (userName,))
    # insert initial balance of 100
# userID is returned with the id within the most recent modified row in the table
userID = cur.lastrowid

print("[INFO] Beginning face captures...")

video_capture = cv2.VideoCapture(0)
total = 0 # represent # of face images stored

while (True):
    ret, frame = video_capture.read()

    orig_img = frame.copy()
    frame = imutils.resize(frame, width = 600)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert frame to gray
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image.

    #cv2.namedWindow('Face Capture', cv2.WINDOW_NORMAL) # WINDOW_NORMAL enables you to resize
    #cv2.resizeWindow('Face Capture', 600, 600) # width, height

    key = cv2.waitKey(1) & 0xFF

    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Color a bounding box around the face.
        roi_gray = gray[y:y+h, x:x+w] # Get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image.
        cv2.imshow("Face Capture", frame)

    # if the `s` key was pressed, write original image to dataset folder. then later process it and use it for face recognition
    if key == ord("s"):
        for i in range(0, 50):
            cv2.imwrite("custom_dataset/" + str(userName) + '.' + str(userID) + '.' +  str(total) + ".jpg", gray[y:y+h,x:x+w])
            #filename = os.path.sep.join([args["output"], "{}.png".format(str(total).zfill(5))])
            #cv2.imwrite(filename, orig_img)
            total += 1
            print("[INFO] screen captured.. {0} images so far..".format(total))
            time.sleep(0.03) # 30 ms
    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break

connect.commit() # agree to changes


# print the total faces saved and do a bit of cleanup
print("[INFO] {} face images stored".format(total))
print("[INFO] cleaning up..")
cv2.destroyAllWindows() # destroy the window created for the frames of camera
time.sleep(1) 
connect.close() # close db
video_off = video_capture.release()
