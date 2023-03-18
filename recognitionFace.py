import cv2, os, numpy as np

faceDir = 'facedata'
trainDir = 'trainface'

cam = cv2.VideoCapture(0)
cam.set(3, 640)  # change width cam
cam.set(4, 488)  # changer height cam
faceDetector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faceRecognizer = cv2.face.LBPHFaceRecognizer.create()

faceRecognizer.read(trainDir + '/training.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0
names = ['Tidak Diketahui', 'Kang Yasin', 'Kang Yasin']

minWidth = 0.1 * cam.get(3)
minHeight = 0.1 * cam.get(4)

while True:
  retV, frame = cam.read()
  frame = cv2.flip(frame, 1)  # vertical flip
  grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = faceDetector.detectMultiScale(grey, 1.2, 5, minSize=(round(minWidth), round(minHeight)))
  for (x, y, w, h) in faces:
    # set frame
    frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    faceId, confidence = faceRecognizer.predict(grey[y:y + h, x:x + w])
    if confidence <= 50:
      nameID = names[faceId]
      confidenceText = " {0}%".format(round(100 - confidence))
    else:
      nameID = names[0]
      confidenceText = " {0}%".format(round(100 - confidence))
    cv2.putText(frame, str(nameID), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, str(confidenceText), (x + 5, y + h - 5), font, 1, (255, 255, 0), 2)

  cv2.imshow('Recognition Face', frame)
  k = cv2.waitKey(1) & 0xFF
  if k == 27 or k == ord('q'):
    break
print("EXIT")
cam.release()
cv2.destroyAllWindows()
