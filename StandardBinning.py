#! /usr/local/bin/python
import scipy.misc as scm
import matplotlib.pyplot as plt
import numpy as np
import math
import logging
import time

def sharpInt(imageIn, params, maskIn):
	poniX = params['centerX'] - math.sin(params['tilt']) * math.cos(params['rotation']) * params['dist']
	poniY = params['centerY'] - math.sin(params['tilt']) * math.sin(params['rotation']) * params['dist']
	if   (params['axis'] == 'TT'):
		deltaTwoTheta = (params['highTwoTheta']-params['lowTwoTheta'])/(params['numBins'])
	elif (params['axis'] == 'Q'):
		deltaQ = (params['highQ']-params['lowQ'])/(params['numBins'])
	# myArray = np.zeros((pixNum,pixNum))
	myXs = np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))
	myYs = (np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))).T
	myArray = np.arctan(( ( (math.cos(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.sin(params['tilt'])*params['dist'])**2 +
				  (math.cos(params['rotation'])*(myYs-poniY) - math.sin(params['rotation'])*(myXs-poniX))**2 ) /
				  (-math.sin(params['tilt'])*(math.cos(params['rotation'])*(myXs-poniX)+math.sin(params['rotation'])*(myYs-poniY)) + math.cos(params['tilt'])*params['dist'])**2 )**(0.5))*180/math.pi
	# calculate distance correction
	flXs = np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1)) - params['centerX']
	flYs = (np.linspace(0,params['pixNum']*params['pixSize'],params['pixNum'])*np.ones((params['pixNum'],1))).T - params['centerY']
	myXs = math.cos(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	myYs = (-math.sin(params['rotation'])*flXs+math.cos(params['rotation'])*flYs)
	myZs = -math.sin(params['tilt'])*(math.cos(params['rotation'])*flXs+math.sin(params['rotation'])*flYs)
	# Single-pixel correction!
	distCorr = ((myXs)**2 + (myYs)**2 + (myZs-params['dist'])**2)
	# Angle of incidence correction
	angleCorr = (distCorr**0.5)/(-myXs*math.sin(params['tilt']) + (params['dist'] - myZs)*math.cos(params['tilt']))
	corrected = imageIn*distCorr*angleCorr
	# Polarization Correction
	# polCorr = (1 + (np.cos(myArray*math.pi/180))**2 - params['polFactor']*np.cos(np.arctan2(flYs, flXs)+params['polPhase'])*((np.sin(myArray*math.pi/180))**2))
	# polCorr = np.ones((params['pixNum'],params['pixNum']))
	intValues = np.zeros(params['numBins'])
	intCounts = np.zeros(params['numBins'])
	squareValues = np.zeros(params['numBins'])
	if   (params['axis'] == 'TT'):
		for x in range(params['pixNum']):
			for y in range(params['pixNum']):
				if (maskIn[y,x] > 0):
					ttBin = int((myArray[x,y]-params['lowTwoTheta'])/deltaTwoTheta)
					if (ttBin < params['numBins']):
						# intValues[ttBin] += polCorr[y,x] * angleCorr[y,x] * distCorr[y,x] * imageIn[y,x]
						# intValues[ttBin] += distCorr[y,x] * imageIn[y,x]
						intValues[ttBin] += corrected[y,x]
						intCounts[ttBin] += 1
						# squareValues[ttBin] += (polCorr[y,x] * angleCorr[y,x] * distCorr[y,x] * ( imageIn[y,x] * params['errorScale'] + params['errorBase'] ))**2
						squareValues[ttBin] += (distCorr[y,x] * ( imageIn[y,x] * params['errorScale'] + params['errorBase'] ))**2
	elif (params['axis'] == 'Q'):
		for x in range(params['pixNum']):
			for y in range(params['pixNum']):
				if (maskIn[y,x] > 0):
					nextQ = 4*math.pi*math.sin(myArray[x,y]*math.pi/360) / params['lambda']
					qBin = int((nextQ-params['lowQ'])/deltaQ)
					if (qBin < params['numBins']):
						# intValues[qBin] += polCorr[y,x] * angleCorr[y,x] * distCorr[y,x] * imageIn[y,x]
						intValues[qBin] += corrected[y, x]
						intCounts[qBin] += 1
						# squareValues[qBin] += (polCorr[y,x] * angleCorr[y,x] * distCorr[y,x] * ( imageIn[y,x] * params['errorScale'] + params['errorBase'] ))**2
						squareValues[qBin] += (distCorr[y,x] * ( imageIn[y,x] * params['errorScale'] + params['errorBase'] ))**2
	intBins = intValues/(intCounts+1E-15)
	intErrs = ( squareValues / (intCounts+1E-15)**2 )**0.5
	if (params['axis'] == 'TT'):
		return [[(x+0.5)*deltaTwoTheta+params['lowTwoTheta'] for x in range(params['numBins'])], intBins, intErrs]
	elif (params['axis'] == 'Q'):
		return [[(x+0.5)*deltaQ+params['lowQ'] for x in range(params['numBins'])], intBins, intErrs]
