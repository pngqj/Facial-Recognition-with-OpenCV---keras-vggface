import numpy as np
import cv2
import pickle
from Face_detect import FacialRecognition


face_cascade = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_alt2.xml')
cap = cv2.VideoCapture(0)
face_detect = FacialRecognition()

while(True):
	# Capture frame-by-frame
	ret, frame = cap.read()
	gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray)

	# for (x, y, w, h) in faces:
	if faces != ():
		(x, y, w, h) = faces[0]
		rec = cv2.rectangle(frame,(x,y),(x+w,y+h),(127,0,255),2)
		name, prob = face_detect.predict_face(frame)
		print(name, prob)
		if prob != None:
			prob = str(int(prob*100)) + "%" 
			cv2.putText(rec, name + " " + prob, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
	# Display the resulting frame
	cv2.imshow('frame',frame)
	if cv2.waitKey(20) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()