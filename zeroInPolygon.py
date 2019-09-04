#! /usr/local/bin/python3
import imageio as iio
import numpy as np
from numba import jit

# imageDims is a number (for a square image) or a tuple (for a rectangular image) specifying the desired image size
# polygon is a list of vertices in x, y coordinates
def zeroInPolygon(imageDims, polygon):
	try:
		# We got passed a number
		returnImage = np.ones((imageDims, imageDims), dtype=np.bool)
	except TypeError:
		# It looks like we got passed a tuple
		returnImage = np.ones(imageDims, dtype=np.bool)
	npPolygon = np.asarray(polygon)
	return zIPNumba(returnImage,npPolygon)

# A function breaking out the core code for NUMBA acceleration
@jit(nopython=True)
def zIPNumba(imageIn, polygon):
	for x in range(0, imageIn.shape[0]):
		for y in range(0, imageIn.shape[1]):
			# Check if the pixel coordinates are within the defined polygon
			for lineI in range(polygon.shape[0]-1):
				imageIn[x,y] = (imageIn[x,y] != xRayCross((x,y), polygon[lineI:lineI+1,:]))
	return imageIn

# Quickly calculates if a point to the right of a line intercept
@jit(nopython=True)
def xRayCross(point, line):
	if line[1, 1] == line[0, 1]:
		return False
	if (line[1, 1] > line[0, 1]):
		if (point[1] < line[1, 1]) != (point[1] >= line[0, 1]):
			return False
	else:
		if (point[1] < line[0, 1]) != (point[1] >= line[1, 1]):
			return False
	intercept = (point[1]*line[1, 0]+line[1, 1]*line[0, 0]-point[1]*line[0, 0]-line[0, 1]*line[1, 0])/(line[1, 1]-line[0, 1])
	if point[0] < intercept:
		return True
	else:
		return False

