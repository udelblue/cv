from __future__ import division
import cv2
#to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin
import argparse

if __name__=="__main__":
	argparser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
	argparser.add_argument('i',help = '')
	args = argparser.parse_args()
	i = args.i
	#read the image
	image = cv2.imread(i)
	gwashBW = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	ret,thresh1 = cv2.threshold(gwashBW,15,255,cv2.THRESH_BINARY) #the value of 15 is chosen by trial-and-error to produce the best outline of the skull
	kernel = np.ones((5,5),np.uint8) #square image kernel used for erosion
	erosion = cv2.erode(thresh1, kernel,iterations = 1) #refines all edges in the binary image

	opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
	closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel) #this is for further removing small noises and holes in the image

	plt.imshow(closing, 'gray') #Figure 2
	plt.xticks([]), plt.yticks([])
	plt.show()

	contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #find contours with simple approximation
	areas = [] #list to hold all areas

	for contour in contours:
	  ar = cv2.contourArea(contour)
	  areas.append(ar)

	max_area = max(areas)
	max_area_index = areas.index(max_area) #index of the list element with largest area

	cnt = contours[max_area_index] #largest area contour

	cv2.drawContours(closing, [cnt], 0, (255, 255, 255), 3, maxLevel = 0)
	# show the images
	cv2.imshow("Result", np.hstack([image, closing]))
	cv2.waitKey(0)