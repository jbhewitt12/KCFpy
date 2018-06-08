import numpy as np 
import cv2
import sys
import os
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

def convert_groundtruth(gt):
    truths = []
    
    for row in gt:
        xvals = [row[0],row[2],row[4],row[6]]
        yvals = [row[1],row[3],row[5],row[7]]
        top = max(yvals)
        bottom = min(yvals)
        left = min(xvals)
        right = max(xvals)
        box = []
        box.append(left)
        box.append(bottom)
        box.append(abs(right - left))
        box.append(abs(top - bottom))
        truths.append(box)

    truths = np.asarray(truths)
    return truths


if __name__ == '__main__':
	
	base_path = os.path.dirname(os.path.abspath(__file__)) + '/VOTdataset/'
	images = '%08d.jpg'

	if(len(sys.argv)==1):
		print 'ERROR: you must enter the path from the "VOTdataset" folder to the sequence folder you wish to run.'	
		print 'eg: "python run.py VOT2016/matrix"'
		sys.exit()
	
	elif(len(sys.argv)==2):
		folder_p = base_path+sys.argv[1]+'/groundtruth.txt'
		cap = cv2.VideoCapture(base_path+sys.argv[1]+'/'+images)
		
	else:  assert(0), "too many arguments"

	gt = np.loadtxt(folder_p,delimiter=',')
	if(len(gt[0]) == 8): #VOT2014 and beyond uses 8 values for the ground truth
		gt = convert_groundtruth(gt)

	initial_gt = gt[0]
	row1 = gt[0]

	tracker = kcftracker.KCFTracker(True, True, True)  # hog, fixed_window, multiscale
	#if you use hog feature, there will be a short pause after you draw a first boundingbox, that is due to the use of Numba.

	cv2.namedWindow('tracking')
	cv2.setMouseCallback('tracking',draw_boundingbox)
	folder = 'saved'
	count = 0 #corresponds to correct ground truth index
	overlap_total = 0 
	reinit = False
	skip_count = False
	reinitialize_count = 0 #number of times reanitialized
	initialized = 0 #number of frames since being reinitialized (We do not count towards average overlap within 10 frames of reinitialization) 
	accuracy_count = 0
	fps_total = 0
	fps_count = 0
	while(cap.isOpened()):
		if reinit: 
			k = 0
			while(k<10): # Skip 10 frames and reinitialize tracker
				ret, frame = cap.read()
				k += 1
				count += 1

			start = True
			reinit = False
			skip_count = True
			initialized = 0
			
		else:
			ret, frame = cap.read()
		
		if not ret:
			break

		if(count < len(gt)):
			current_gt = gt[count]
		
		if start:
			#format: x1, y1, width, height ----- x1, y1 is the bottom left corner of the rectangle
			w = current_gt[2]	
			h = current_gt[3]	
			tracker.init([current_gt[0],current_gt[1],w,h], frame)
			start = False

		initTracking = False
		
		t0 = time()
		boundingbox = tracker.update(frame) #Get the new bounding box from the tracker, given the new frame
		t1 = time()

		boundingbox = map(int, boundingbox)
		
		estimate = np.array([boundingbox[0],boundingbox[1],boundingbox[2],boundingbox[3]])
		
		cv2.rectangle(frame,(boundingbox[0],boundingbox[1]), (boundingbox[0]+boundingbox[2],boundingbox[1]+boundingbox[3]), (0,255,255), 2)
		cv2.rectangle(frame,(int(current_gt[0]),int(current_gt[1])), (int(current_gt[0])+int(current_gt[2]),int(current_gt[1])+int(current_gt[3])), (255,0,0), 2)

		duration = 0.8*duration + 0.2*(t1-t0)
		fps = 1/duration
		
		fps_total += fps
		fps_count += 1
		
		if(count < len(gt)):
			overlap = overlap_ratio(gt[count], estimate)
		if overlap[0] == 0:
			reinit = True
			reinitialize_count += 1
		if(initialized >= 10):
			overlap_total += overlap[0]
			accuracy_count += 1

		# cv2.putText(frame, 'FPS: '+str(1/duration)[:4].strip('.'), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
		cv2.putText(frame, 'Accuracy: '+ str(overlap[0]), (8,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

		cv2.imshow('tracking', frame)

		cv2.imwrite('%s/%s.JPEG' % (folder,count),frame)
		if skip_count:
			skip_count = False
		else:
			count += 1
		initialized += 1
		c = cv2.waitKey(inteval) & 0xFF
		if c==27 or c==ord('q'):
			break


	print 'average fps:'
	print fps_total/fps

	print 'average overlap:'
	print overlap_total/accuracy_count

	print 'Number of reinitializations:'
	print reinitialize_count
	cap.release()
	cv2.destroyAllWindows()

