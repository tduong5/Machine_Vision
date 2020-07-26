# Face Detection Model using a webcam

"""
to fix macOS "abort trap: 6" error code
1.type cmd+shift+p 
2.type "shell command: Install code in PATH"
3. Close vscode
4. Use "sudo code" to open vscode 
5. It will give warning not to run as a root user
6. Ignore the warning and run the file , you will not get the "Abort trap: 6" error anymore.
"""

# change from "import cv2" to remove cv2 modules not being recognized error
from cv2 import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') # Load the cascade for the face.

def facedetect(gray, frame): # Create a function that takes the image as input in black and white (gray) and the original image (frame), then return the same image with the detector rectangles.
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) # Apply the detectMultiScale method from the face cascade to locate one or several faces in the image.
    for (x, y, w, h) in faces: # For each detected face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2) # Color a bounding box around the face.
        roi_gray = gray[y:y+h, x:x+w] # Get the region of interest in the black and white image.
        roi_color = frame[y:y+h, x:x+w] # Get the region of interest in the colored image.
    return frame # Return the image with the detector rectangles.

# 0 for single camera
video_capture = cv2.VideoCapture(0) # Turn the webcam on.

# Repeat infinitely until you want to stop the process
while(True):
    # capture frame-by-frame
    ret, frame = video_capture.read() 
        # Grabs, decodes and returns the next video frame.
        # returns false if no frames has been grabbed
    # convert image to gray. usually recommended for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    facedetect(gray, frame)
    # creates a windows referenced by name 
    cv2.namedWindow('OpenCV_Assignment1', cv2.WINDOW_NORMAL) # WINDOW_NORMAL enables you to resize 
    cv2.resizeWindow('OpenCV_Assignment1', 600, 600) # width, height
    # displays image in specificed window
    cv2.imshow('OpenCV_Assignment1', frame)
# The stop condition. 'ESC' key
    if(cv2.waitKey(1) == 27):
        break
# Turn the webcam off.
video_off = video_capture.release()
# to destroy all of HighGUI windows. destroy the window created for the frames of camera
cv2.destroyAllWindows()