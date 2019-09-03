#! /usr/local/bin/python
import imageio as iio
import numpy as np
import scipy.signal as scs
import math
import time
import matplotlib.pyplot as plt
import integrateImage as iI
from numba import double
from numba import jit

######################################
#     Integration of 2D Data         #
######################################

def smoothInt(imageIn, params, maskIn):
    # Calculate the point of nearest incidence
	poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
	poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']

	if   (params['axis'] == 'TT'):
		deltaTwoTheta = (params['highTwoTheta']-params['lowTwoTheta'])/(params['numBins'])
	elif (params['axis'] == 'Q'):
		deltaQ = (params['highQ']-params['lowQ'])/(params['numBins'])
	# Bin centers = l + (h-l)*(i+0.5)/(n)
	# Bin upper limits = l + (h-l)*(i+1)/(n)

	# Calculate some distance corrections
	flXs = np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))     - params['centerX']
	flYs = (np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))).T - params['centerY']
	myXs = math.cos(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	myYs = (-math.sin(params['rotation'])*flXs+math.cos(params['rotation'])*flYs)
	myZs = -math.sin(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	# Single-pixel correction!
	distCorr = ((myXs)**2 + (myYs)**2 + (myZs-params['dist'])**2)
	# Angle of incidence correction
	angleCorr = (distCorr**0.5)/(-myXs*math.sin(params['tilt']) + (params['dist'] - myZs)*math.cos(params['tilt']))
	# Speed up inverse calculations
	binLimits = np.linspace(0,params['numBins']-1, params['numBins'])
	if   (params['axis'] == 'TT'):
		binLimits = np.tan((params['lowTwoTheta'] + deltaTwoTheta*(binLimits+1)) * math.pi/180)
	elif (params['axis'] == 'Q'):
		binLimits = np.tan(2 * np.arcsin(params['lambda']*(params['lowQ'] + deltaQ*(binLimits+1))/ (4*math.pi)))

	# Calculate alpha values for the coming loop
	myXs = np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))
	myYs = (np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))).T
	aA = ( ( (math.cos(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.sin(params['tilt'])*params['dist'])**2 +
					(math.cos(params['rotation'])*(myYs-poniY) - math.sin(params['rotation'])*(myXs-poniX))**2 ) /
					(-math.sin(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.cos(params['tilt'])*params['dist'])**2 )**(0.5)
	aA = aA.T

	myAzi = ((np.arctan2((myYs-poniY), -(myXs-poniX)) + np.pi) - params['aziStart']) % (2*np.pi)
	aziStep = 2.0 * np.pi / params['aziCount']

	# Run the loop in NUMBA-accelerated sub-functions
	if (params['axis'] == 'TT'):
		return smoothIntNumbaTT(params['numBins'], imageIn, distCorr, angleCorr, maskIn, aA, params['pixNum'], deltaTwoTheta, params['lowTwoTheta'], binLimits, params['aziCount'], aziStep, myAzi)
	elif (params['axis'] == 'Q'):
		return smoothIntNumbaQ(params['numBins'], imageIn, distCorr, angleCorr, maskIn, aA, params['pixNum'], deltaQ, params['lowQ'], binLimits, params['lambda'], params['aziCount'], aziStep, myAzi)

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothIntNumbaTT(numBins, imageIn, distCorr, angleCorr, maskIn, aA, pixNum, deltaTwoTheta, lowTwoTheta, binLimits, aziCount, aziStep, myAzi):
	# For each bin
	intValues = np.zeros((numBins, aziCount))
	intCounts = np.zeros((numBins, aziCount))
	corrected = imageIn*distCorr*angleCorr
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Calculate the azimuthal bin
			aziBin = int(myAzi[pX, pY] / aziStep)
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((math.atan(laA00)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				upper = int(math.ceil((math.atan(laA11)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						intValues[lower, aziBin] += corrected[pX, pY]
						intCounts[lower, aziBin] += 1
				else:
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						intValues[lower, aziBin] += lowWeight * corrected[pX, pY]
						intCounts[lower, aziBin] += lowWeight
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								intValues[binN, aziBin] += (highWeight - lowWeight) * corrected[pX, pY]
								intCounts[binN, aziBin] += (highWeight - lowWeight)
							lowWeight = highWeight
	intBins = intValues/(intCounts+1E-15)
	return np.concatenate((np.linspace(0.5*deltaTwoTheta+lowTwoTheta, (numBins-0.5)*deltaTwoTheta+lowTwoTheta, numBins)[:,np.newaxis], intBins), axis=1)

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothIntNumbaQ(numBins, imageIn, distCorr, angleCorr, maskIn, aA, pixNum, deltaQ, lowQ, binLimits, lamda, aziCount, aziStep, myAzi):
	# For each bin
	intValues = np.zeros((numBins, aziCount))
	intCounts = np.zeros((numBins, aziCount))
	corrected = imageIn*distCorr*angleCorr
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Calculate the azimuthal bin
			aziBin = int(myAzi[pX, pY] / aziStep)
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((4*math.pi*math.sin(math.atan(laA00)/2.0)/lamda - lowQ) / deltaQ))
				upper = int(math.floor((4*math.pi*math.sin(math.atan(laA11)/2.0)/lamda - lowQ) / deltaQ))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						intValues[lower, aziBin] += corrected[pX, pY]
						intCounts[lower, aziBin] += 1
				else:
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						intValues[lower, aziBin] += lowWeight * corrected[pX, pY]
						intCounts[lower, aziBin] += lowWeight
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								intValues[binN, aziBin] += (highWeight - lowWeight) * corrected[pX, pY]
								intCounts[binN, aziBin] += (highWeight - lowWeight)
							lowWeight = highWeight
	intBins = intValues/(intCounts+1E-15)
	# test1 = np.expand_dims(np.linspace(0.5*deltaQ+lowQ, (numBins-0.5)*deltaQ+lowQ, numBins), 1)
	# test2 = np.expand_dims(intBins, 1)
	return np.concatenate((np.expand_dims(np.linspace(0.5*deltaQ+lowQ, (numBins-0.5)*deltaQ+lowQ, numBins), 1), intBins), axis=1)

######################################
#     Projection of 1D Data          #
######################################

def smoothProj(dataIn, params, maskIn):
    # Calculate the point of nearest incidence
	poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
	poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']

	if   (params['axis'] == 'TT'):
		deltaTwoTheta = (params['highTwoTheta']-params['lowTwoTheta'])/(params['numBins'])
	elif (params['axis'] == 'Q'):
		deltaQ = (params['highQ']-params['lowQ'])/(params['numBins'])
	# Bin centers = l + (h-l)*(i+0.5)/(n)
	# Bin upper limits = l + (h-l)*(i+1)/(n)

	# Calculate some distance corrections
	flXs = np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))     - params['centerX']
	flYs = (np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))).T - params['centerY']
	myXs = math.cos(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	myYs = (-math.sin(params['rotation'])*flXs+math.cos(params['rotation'])*flYs)
	myZs = -math.sin(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	# Single-pixel correction!
	distCorr = ((myXs)**2 + (myYs)**2 + (myZs-params['dist'])**2)
	# Angle of incidence correction
	angleCorr = (distCorr**0.5)/(-myXs*math.sin(params['tilt']) + (params['dist'] - myZs)*math.cos(params['tilt']))
	# Speed up inverse calculations
	binLimits = np.linspace(0,params['numBins']-1, params['numBins'])
	if   (params['axis'] == 'TT'):
		binLimits = np.tan((params['lowTwoTheta'] + deltaTwoTheta*(binLimits+1)) * math.pi/180)
	elif (params['axis'] == 'Q'):
		binLimits = np.tan(2 * np.arcsin(params['lambda']*(params['lowQ'] + deltaQ*(binLimits+1))/ (4*math.pi)))

	# Calculate alpha values for the coming loop
	myXs = np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))
	myYs = (np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))).T
	aA = ( ( (math.cos(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.sin(params['tilt'])*params['dist'])**2 +
					(math.cos(params['rotation'])*(myYs-poniY) - math.sin(params['rotation'])*(myXs-poniX))**2 ) /
					(-math.sin(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.cos(params['tilt'])*params['dist'])**2 )**(0.5)
	aA = aA.T

	# Run the loop in NUMBA-accelerated sub-functions
	if (params['axis'] == 'TT'):
		return smoothProjNumbaTT(params['numBins'], dataIn, distCorr, angleCorr, maskIn, aA, params['pixNum'], deltaTwoTheta, params['lowTwoTheta'], binLimits)
	elif (params['axis'] == 'Q'):
		return smoothProjNumbaQ(params['numBins'], dataIn, distCorr, angleCorr, maskIn, aA, params['pixNum'], deltaQ, params['lowQ'], binLimits, params['lambda'])

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothProjNumbaTT(numBins, dataIn, distCorr, angleCorr, maskIn, aA, pixNum, deltaTwoTheta, lowTwoTheta, binLimits):
	# For each bin
	correction = 1/(distCorr*angleCorr)
	returnImage = np.zeros((pixNum, pixNum), dtype=np.double)
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((math.atan(laA00)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				upper = int(math.ceil((math.atan(laA11)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						returnImage[pX, pY] += correction[pX, pY]*dataIn[lower]
				else:
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						returnImage[pX, pY] += lowWeight*correction[pX, pY]*dataIn[lower]
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								returnImage[pX, pY] += (highWeight-lowWeight)*correction[pX, pY]*dataIn[binN]
							lowWeight = highWeight
	return returnImage

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothProjNumbaQ(numBins, dataIn, distCorr, angleCorr, maskIn, aA, pixNum, deltaQ, lowQ, binLimits, lamda):
	# For each bin
	returnImage = np.zeros((pixNum, pixNum), dtype=np.double)
	correction = 1/(distCorr*angleCorr+1E-15)
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((4*math.pi*math.sin(math.atan(laA00)/2.0)/lamda - lowQ) / deltaQ))
				upper = int(math.floor((4*math.pi*math.sin(math.atan(laA11)/2.0)/lamda - lowQ) / deltaQ))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						returnImage[pX, pY] = returnImage[pX, pY] + correction[pX, pY] * dataIn[lower]
				else:
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						returnImage[pX, pY] = returnImage[pX, pY] + lowWeight * correction[pX, pY] * dataIn[lower]
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								returnImage[pX, pY] = returnImage[pX, pY] + (highWeight - lowWeight) * correction[pX, pY] * dataIn[binN]
							lowWeight = highWeight
	return returnImage

######################################
#   A helper integration function    #
######################################

@jit(nopython=True)
def calcPixWeight(pX, pY, alpha, laA00, laA01, laA10, laA11):
	if alpha > laA10:
		if alpha > laA01:
			a = (laA11 - alpha) / (laA11 - laA10)
			b = (laA11 - alpha) / (laA11 - laA01)
			return 1 - a*b/2.0
		else:
			a = (alpha - laA10) / (laA11 - laA10)
			b = (alpha - laA00) / (laA01 - laA00)
			return (a + b)/2.0
	else:
		if alpha > laA01:
			a = (alpha - laA00) / (laA10 - laA00)
			b = (alpha - laA01) / (laA11 - laA01)
			return (a + b)/2.0
		else:
			a = (alpha - laA00) / (laA10 - laA00)
			b = (alpha - laA00) / (laA01 - laA00)
			return a*b/2.0

######################################
#  A function to Reverse Broadening  #
######################################

def smoothOverlap(params, maskIn):
    # Calculate the point of nearest incidence
	poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
	poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']

	if   (params['axis'] == 'TT'):
		deltaTwoTheta = (params['highTwoTheta']-params['lowTwoTheta'])/(params['numBins'])
	elif (params['axis'] == 'Q'):
		deltaQ = (params['highQ']-params['lowQ'])/(params['numBins'])
	# Bin centers = l + (h-l)*(i+0.5)/(n)
	# Bin upper limits = l + (h-l)*(i+1)/(n)

	# Speed up inverse calculations
	binLimits = np.linspace(0,params['numBins']-1, params['numBins'])
	if   (params['axis'] == 'TT'):
		binLimits = np.tan((params['lowTwoTheta'] + deltaTwoTheta*(binLimits+1)) * math.pi/180)
	elif (params['axis'] == 'Q'):
		binLimits = np.tan(2 * np.arcsin(params['lambda']*(params['lowQ'] + deltaQ*(binLimits+1))/ (4*math.pi)))

	# Calculate alpha values for the coming loop
	myXs = np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))
	myYs = (np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))).T
	aA = ( ( (math.cos(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.sin(params['tilt'])*params['dist'])**2 +
					(math.cos(params['rotation'])*(myYs-poniY) - math.sin(params['rotation'])*(myXs-poniX))**2 ) /
					(-math.sin(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.cos(params['tilt'])*params['dist'])**2 )**(0.5)
	aA = aA.T

	myAzi = ((np.arctan2((myYs-poniY), -(myXs-poniX)) + np.pi) - params['aziStart']) % (2*np.pi)
	aziStep = 2.0 * np.pi / params['aziCount']

	# Run the loop in NUMBA-accelerated sub-functions
	if (params['axis'] == 'TT'):
		return smoothOverlapNumbaTT(params['numBins'], maskIn, aA, params['pixNum'], deltaTwoTheta, params['lowTwoTheta'], binLimits, params['aziCount'], aziStep, myAzi)
	elif (params['axis'] == 'Q'):
		return smoothOverlapNumbaQ(params['numBins'], maskIn, aA, params['pixNum'], deltaQ, params['lowQ'], binLimits, params['lambda'], params['aziCount'], aziStep, myAzi)

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothOverlapNumbaTT(numBins, maskIn, aA, pixNum, deltaTwoTheta, lowTwoTheta, binLimits, aziCount, aziStep, myAzi):
	# For each bin
	intCounts = np.zeros((numBins, aziCount))
	intSqCounts = np.zeros((numBins, numBins, aziCount))
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Calculate the azimuthal bin
			aziBin = int(myAzi[pX, pY] / aziStep)
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((math.atan(laA00)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				upper = int(math.ceil((math.atan(laA11)*180/math.pi - lowTwoTheta) / deltaTwoTheta))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						intCounts[lower, aziBin] += 1
						intSqCounts[lower, lower, aziBin] += 1
				else:
					# Used to calcualte square counts
					weightList = []
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						intCounts[lower, aziBin] += lowWeight
						weightList += [(lower, lowWeight)]
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								intCounts[binN, aziBin] += (highWeight - lowWeight)
								weightList += [(binN, (highWeight - lowWeight))]
							lowWeight = highWeight
					for weight1 in weightList:
						for weight2 in weightList:
							intSqCounts[weight1[0], weight2[1], aziBin] += weight1[1]*weight2[1]
	intSqCounts = intSqCounts / np.expand_dims(intCounts, 2)
	return intSqCounts

# Break out the core calculation loop for NUMBA acceleration
@jit(nopython=True)
def smoothOverlapNumbaQ(numBins, maskIn, aA, pixNum, deltaQ, lowQ, binLimits, lamda, aziCount, aziStep, myAzi):
	# For each bin
	intCounts = np.zeros((numBins, aziCount), dtype=np.double)
	intSqCounts = np.zeros((numBins, numBins, aziCount), dtype=np.double)
	for pX in range(pixNum):
		for pY in range(pixNum):
			# Calculate the azimuthal bin
			aziBin = int(myAzi[pX, pY] / aziStep)
			# Flip the ordering so the lowest value is [0,0] and the largest value is [1,1]
			if maskIn[pX, pY] > 0:
				if aA[pX, pY] < aA[pX+1, pY]:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX  , pY  ]
						laA01 = aA[pX+1, pY  ]
						laA10 = aA[pX  , pY+1]
						laA11 = aA[pX+1, pY+1]
					else:
						laA00 = aA[pX  , pY+1]
						laA01 = aA[pX+1, pY+1]
						laA10 = aA[pX  , pY  ]
						laA11 = aA[pX+1, pY  ]
				else:
					if aA[pX, pY] < aA[pX, pY+1]:
						laA00 = aA[pX+1, pY  ]
						laA01 = aA[pX  , pY  ]
						laA10 = aA[pX+1, pY+1]
						laA11 = aA[pX  , pY+1]
					else:
						laA00 = aA[pX+1, pY+1]
						laA01 = aA[pX  , pY+1]
						laA10 = aA[pX+1, pY  ]
						laA11 = aA[pX  , pY  ]
				lower = int(math.floor((4*math.pi*math.sin(math.atan(laA00)/2.0)/lamda - lowQ) / deltaQ))
				upper = int(math.floor((4*math.pi*math.sin(math.atan(laA11)/2.0)/lamda - lowQ) / deltaQ))
				nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < numBins and lower > 0):
						intCounts[lower, aziBin] += 1
						intSqCounts[lower, lower, aziBin] += 1
				else:
					# Used to calcualte square counts
					weightList = np.zeros((upper+1-lower), dtype=np.double)
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < numBins and lower > 0):
						intCounts[lower, aziBin] += lowWeight
						weightList[0] = lowWeight
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < numBins and binN > 0):
								intCounts[binN, aziBin] += (highWeight - lowWeight)
								weightList[binN-lower] = (highWeight - lowWeight)
							lowWeight = highWeight
					for w1 in range(upper+1-lower):
						for w2 in range(upper+1-lower):
							intSqCounts[lower+w1, lower+w2, aziBin] += weightList[w1]*weightList[w2]
	for aziBin in range(aziCount):
		for binN in range(numBins):
			intSqCounts[binN, :, aziBin] = intSqCounts[binN, :, aziBin] / (intCounts[binN, aziBin]+(1E-15))
	return intSqCounts

######################################
#        Testing my functions        #
######################################

def getAzis(params):
	poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
	poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']
	myXs = np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))
	myYs = (np.linspace(0-params['pixSize']/2,(params['pixNum']+0.5)*params['pixSize'],params['pixNum']+1)*np.ones((params['pixNum']+1,1))).T
	myAzi = ((np.arctan2((myYs-poniY), -(myXs-poniX)) + np.pi) - params['aziStart']) % (2*np.pi)
	aziStep = 2.0 * np.pi / params['aziCount']
	return np.floor(myAzi / aziStep)

# An ignored pixels mask
myMask = iio.imread('cutLRMask.tiff')/255.0
# A "mask" to average neighbors
myAve = iio.imread('lrPix.tiff')/255.0
# The raw image
myImage = iio.imread('Ni_20190527-120854_18095b_0001_dark_corrected_img.tiff')
# Average over neighbors
lineFix = scs.convolve2d(myImage, np.asarray([[1.0,0.0,1.0]])/2.0)
# The corrected image
myImage = (1-myAve)*myImage[16:2032,16:2032] + myAve*lineFix[16:2032,17:2033]

params = {
	'axis': 'Q',
	'aziCount' : 1,
	'aziStart' : (-45*np.pi/180) }
	'centerX': 213.0054,
	'centerY': 206.3082,
	'dist': 332.556,
	'errorBase': 1.0,
	'errorScale': 0.003,
	'highQ': 33.0,
	'lambda': 0.18208,
	'lowQ': 0.06,
	'numBins': 2500,
	'pixNum': 2048,
	'pixSize': 0.200,
	'rotation': -2.692013291568574,
	'tilt': 0.006178,


poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']

deltaQ = (params['highQ']-params['lowQ'])/(params['numBins'])

myT = time.time()
looseVals = smoothInt(myImage, params, myMask)
print(time.time() - myT)
myT = time.time()
newImage = smoothProj(looseVals[:,1], params, myMask)
print(time.time() - myT)
