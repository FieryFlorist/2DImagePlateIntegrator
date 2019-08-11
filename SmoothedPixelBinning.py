#! /usr/local/bin/python
import scipy.misc as scm
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import time

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
	# Calculate the weights between each bin and each pixel
	# For each bin
	intValues = np.zeros(params['numBins'])
	intCounts = np.zeros(params['numBins'])
	corrected = imageIn*distCorr*angleCorr
	for pX in range(params['pixNum']):
		for pY in range(params['pixNum']):
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
				if   (params['axis'] == 'TT'):
					lower = int(math.floor((math.atan(laA00)*180/math.pi - params['lowTwoTheta']) / deltaTwoTheta))
					upper = int(math.ceil((math.atan(laA11)*180/math.pi - params['lowTwoTheta']) / deltaTwoTheta))
					nextAlpha = binLimits[lower]
				elif (params['axis'] == 'Q'):
					lower = int(math.floor((4*math.pi*math.sin(math.atan(laA00)/2.0)/params['lambda'] - params['lowQ']) / deltaQ))
					upper = int(math.floor((4*math.pi*math.sin(math.atan(laA11)/2.0)/params['lambda'] - params['lowQ']) / deltaQ))
					nextAlpha = binLimits[lower]
				if nextAlpha > laA11:
					#Pure Pixel
					if(lower < params['numBins'] and lower > 0):
						intValues[lower] += corrected[pX, pY]
						intCounts[lower] += 1
				else:
					# Mixed Pixel
					lowWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
					if(lower < params['numBins'] and lower > 0):
						intValues[lower] += lowWeight * corrected[pX, pY]
						intCounts[lower] += lowWeight
					if upper > lower:
						for binN in range(lower+1, upper+1):
							nextAlpha = binLimits[binN]
							if nextAlpha > laA11:
								highWeight = 1
							else:
								highWeight = calcPixWeight(pX, pY, nextAlpha, laA00, laA01, laA10, laA11)
							if(binN < params['numBins'] and binN > 0):
								intValues[binN] += (highWeight - lowWeight) * corrected[pX, pY]
								intCounts[binN] += (highWeight - lowWeight)
							lowWeight = highWeight
	intBins = intValues/(intCounts+1E-15)
	if (params['axis'] == 'TT'):
		return [[(x+0.5)*deltaTwoTheta+params['lowTwoTheta'] for x in range(params['numBins'])], intBins, intValues]
	elif (params['axis'] == 'Q'):
		return [[(x+0.5)*deltaQ+params['lowQ'] for x in range(params['numBins'])], intBins, intValues, intValues]
