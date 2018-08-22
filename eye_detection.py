import numpy as np
import cv2
from matplotlib import pyplot as plt
from imutils.video import VideoStream

eye_cascade = cv2.CascadeClassifier('./haarcascades/haarcascade_eye.xml')

vs = VideoStream(src=0).start()
orgimg =  vs.read()
img = vs.read()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

eyes = eye_cascade.detectMultiScale(img)
for (ex,ey,ew,eh) in eyes:
	cv2.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
		

cv2.imshow('img',np.hstack((img,orgimg)))
cv2.waitKey(0)
cv2.destroyAllWindows()