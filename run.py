import numpy as np 
import cv2
import sys
from time import time

import kcftracker

selectingObject = False
initTracking = False
onTracking = False
start = True
ix, iy, cx, cy = -1, -1, -1, -1
w, h = 0, 0

inteval = 25
duration = 0.01

# mouse callback function
def draw_boundingbox(event, x, y, flags, param):
	global selectingObject, initTracking, onTracking, ix, iy, cx,cy, w, h
	
	if event == cv2.EVENT_LBUTTONDOWN:
		selectingObject = True
		onTracking = False
		ix, iy = x, y
		cx, cy = x, y
	
	elif event == cv2.EVENT_MOUSEMOVE:
		cx, cy = x, y
	
	elif event == cv2.EVENT_LBUTTONUP:
		selectingObject = False
		if(abs(x-ix)>10 and abs(y-iy)>10):
			w, h = abs(x - ix), abs(y - iy)
			ix, iy = min(x, ix), min(y, iy)
			initTracking = True
		else:
			onTracking = False
	
	elif event == cv2.EVENT_RBUTTONDOWN:
		onTracking = False
		if(w>0):
			ix, iy = x-w/2, y-h/2
			initTracking = True



if __name__ == '__main__':
	
	base_path = 'C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/'
	sequence = 'VOT2013/bicycle/'
	images = '%08d.jpg'
	groundtruth_path = base_path+sequence+'groundtruth.txt'
	full_path = base_path+sequence+images

	gt = np.loadtxt(groundtruth_path,delimiter=',')
	initial_gt = gt[0]
	print 'gt: '
	print gt
	row1 = gt[0]
	print row1
	print row1[0]


	if(len(sys.argv)==1):
		# cap = cv2.VideoCapture('C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/VOT2015/bag/%08d.jpg')
		cap = cv2.VideoCapture(full_path)
		# cap = cv2.VideoCapture(0)
	elif(len(sys.argv)==2):
		if(sys.argv[1].isdigit()):  # True if sys.argv[1] is str of a nonnegative integer
			cap = cv2.VideoCapture(int(sys.argv[1]))
		else:
			cap = cv2.VideoCapture(sys.argv[1])
			inteval = 30
	else:  assert(0), "too many arguments"

	tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	cv2.namedWindow('tracking')
	cv2.setMouseCallback('tracking',draw_boundingbox)

	while(cap.isOpened()):
		ret, frame = cap.read()
		# print "frame: "
		# print frame
		if not ret:
			break

		if start:
			# w = abs(291 - 442)	
			# h = abs(120 - 270)	
			# tracker.init([291,120,w,h], frame)
			w = initial_gt[2]	
			h = initial_gt[3]	
			tracker.init([initial_gt[0],initial_gt[1],w,h], frame)
			start = False

		initTracking = False
		onTracking = True
		if(onTracking):
			t0 = time()
			boundingbox = tracker.update(frame) #Get the new bounding box from the tracker, given the new frame
			t1 = time()

			boundingbox = map(int, boundingbox)
			cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 1)
			
			duration = 0.8*duration + 0.2*(t1-t0)
			#duration = t1-t0
			cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		cv2.imshow('tracking', frame)
		c = cv2.waitKey(inteval) & 0xFF
		if c==27 or c==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()
