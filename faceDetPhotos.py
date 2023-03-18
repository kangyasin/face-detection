"""
python -m pip install --upgrade opencv-contrib-python
pip install Pillow
Algorithm Haar-Cascade untuk mendeteksi wajah
Face Detection xml algorithm = https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
Eye Detection xml algorithm = https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
Step record face detection, training face data, recognition
"""
import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # change width cam
cam.set(4, 488)  # changer height cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eyeDetector = cv2.CascadeClassifier('haarcascade_eye.xml')
while True:
  retV, frame = cam.read()
  grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceDetector.detectMultiScale(grey, 1.3, 5)  # frame, scale factor, min neighbors
  for (x, y, w, h) in faces:
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    roiGrey = grey[x:y+h, x:x+w]
    roiColor = frame[y:y+h, x:x+w]
    eyes = eyeDetector.detectMultiScale(roiGrey)
    for (xe, ye, we, he) in eyes:
      cv2.rectangle(roiColor, (xe, ye), (xe+we, ye+he), (0,0,255), 1)
  cv2.imshow('Webcam', frame)
  k = cv2.waitKey(1) & 0xFF
  if k == 27 or k == ord('q'):
    break
cam.release()
cv2.destroyAllWindows()
