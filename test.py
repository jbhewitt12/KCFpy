import numpy as np 
import cv2
import sys
from time import time

import kcftracker



if __name__ == '__main__':
	if(len(sys.argv)==1):
		img = cv2.imread('C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/VOT2015/bag/00000001.jpg')
		cv2.rectangle(img, (291,120), (442, 270), (255,0,0), 2)
		# print img

		# tracker = kcftracker.KCFTracker(True, True, True)

		cv2.imshow('image',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

		# cap = cv2.VideoCapture('C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/VOT2015/bag/%08d.jpg')
		# ret, frame = cap.read()
		# cv2.imshow('tracking', frame)
		# c = cv2.waitKey(0) & 0xFF
		# cv2.destroyAllWindows()




