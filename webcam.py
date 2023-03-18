"""
install modul opencv
pip install opencv-contrib-python
or
python -m pip install --upgrade opencv-contrib-python

pip install Pillow
"""
import cv2

cam = cv2.VideoCapture(0)
while True:
  retV, frame = cam.read()
  grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  cv2.imshow('Webcam', frame)
  cv2.imshow('Webcam - Gray', grey)
  k = cv2.waitKey(1) & 0xFF
  if k == 27 or k == ord('q'):
    break
cam.release()
cv2.destroyAllWindows()
