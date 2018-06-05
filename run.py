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

def overlap_ratio(rect1, rect2):
    '''
   	Takes 2 numpy arrays 
    Compute overlap ratio between two rects
    - rect: 1d array of [x,y,w,h] or 
            2d array of N x [x,y,w,h]
    '''

    if rect1.ndim==1:
        rect1 = rect1[None,:]
    if rect2.ndim==1:
        rect2 = rect2[None,:]

    left = np.maximum(rect1[:,0], rect2[:,0])
    right = np.minimum(rect1[:,0]+rect1[:,2], rect2[:,0]+rect2[:,2])
    top = np.maximum(rect1[:,1], rect2[:,1])
    bottom = np.minimum(rect1[:,1]+rect1[:,3], rect2[:,1]+rect2[:,3])

    intersect = np.maximum(0,right - left) * np.maximum(0,bottom - top)
    union = rect1[:,2]*rect1[:,3] + rect2[:,2]*rect2[:,3] - intersect
    iou = np.clip(intersect / union, 0, 1)
    return iou


if __name__ == '__main__':
	
	base_path = 'C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/'
	sequence = 'VOT2013/bicycle/'
	images = '%08d.jpg'
	groundtruth_path = base_path+sequence+'groundtruth.txt'
	full_path = base_path+sequence+images


	
	# print row1
	# print row1[0]


	if(len(sys.argv)==1):
		# cap = cv2.VideoCapture('C:/Users/Josh/Desktop/Uni/Capstone A/VOT dataset/VOT2015/bag/%08d.jpg')
		cap = cv2.VideoCapture(full_path)
		# cap = cv2.VideoCapture(0)
	elif(len(sys.argv)==2):
		folder_p = base_path+'VOT2013/'+sys.argv[1]+'/groundtruth.txt'
		cap = cv2.VideoCapture(base_path+'VOT2013/'+sys.argv[1]+'/'+images)
		
	else:  assert(0), "too many arguments"

	gt = np.loadtxt(folder_p,delimiter=',')
	initial_gt = gt[0]
	# print 'gt: '
	# print gt
	row1 = gt[0]

	tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	cv2.namedWindow('tracking')
	cv2.setMouseCallback('tracking',draw_boundingbox)
	folder = 'saved'
	count = 0
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
			estimate = np.array([boundingbox[0],boundingbox[1],boundingbox[2],boundingbox[3]])
			# estimate = np.array([boundingbox[0],boundingbox[1],boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]])
			cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 2)
			
			duration = 0.8*duration + 0.2*(t1-t0)
			#duration = t1-t0
			# print 'gt:'
			# print gt[count]
			# print 'estimate:'
			# print estimate
			overlap = overlap_ratio(gt[count], estimate)
			# print overlap[0]
			# if(count > 4):
			# 	break
			# cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
			cv2.putText(frame, 'Accuracy: '+ str(overlap[0]), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		cv2.imshow('tracking', frame)
		# cv2.imwrite('%s/%s.JPEG' % (folder,count),frame)
		count += 1
		c = cv2.waitKey(inteval) & 0xFF
		if c==27 or c==ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

