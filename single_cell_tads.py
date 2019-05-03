#!/share/software/user/open/python/3.6.1/bin/python3

import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform
import math
#from scipy import ndimage
#from scipy import interpolate

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ORCAutils import *
from NormFuncs import *

def nan_gaussian_filter(mat,sigma,keep_nan=True):
	from scipy.ndimage import gaussian_filter
	U=np.array(mat)
	Unan = np.isnan(U)
	V=U.copy()
	V[U!=U]=0
	VV=gaussian_filter(V,sigma=sigma)

	W=0*U.copy()+1
	W[U!=U]=0
	WW=gaussian_filter(W,sigma=sigma)

	Z=VV/WW
	if keep_nan:
		Z[Unan]=np.nan
	return Z

def get_cor_gradient(scIdx, scMaps, sigma, tag, plotFig = False):
	sc_norm_mat = get_single_cell_norm_mat(scIdx, scMaps, tag + "_sc_norm_" + str(scIdx), plotFig)
	sc_cor_mat = get_single_cell_cor_mat(scIdx, scMaps, tag + "_sc_norm_cor_" + str(scIdx), plotFig)
	gaussMapCor = nan_gaussian_filter(sc_cor_mat, sigma=sigma)

	if plotFig == True:
		plot_median_dist_map(gaussMapCor, tag + "_single_cell_gauss_cor_" + str(scIdx))

	gradientX, gradientY = get_XY_matrix_gradient(gaussMapCor, tag, plotFig)
	return gradientX, gradientY, sc_cor_mat

def get_TAD_boundaries_attempt_2(gradX, gradY, tag, plotFig = False):
	gradCollapse = np.nansum(np.abs(gradX), axis=1)

	from scipy.interpolate import UnivariateSpline
	
	spln = UnivariateSpline(range(len(gradCollapse)), gradCollapse, k=4)
	splnDdx = spln.derivative()
	roots = spln.derivative().roots()

	#print("DEBUG: roots " + str(roots))
	
	firstRoot = roots[0]
	xvals = range(0, int(round(firstRoot)))
	yddx = np.sum(splnDdx(xvals))
	
	maxima = list()
	if yddx > 0:
		maxima = [int(round(roots[x])) for x in range(0,len(roots), 2)]
	else:
		maxima = [int(round(roots[x])) for x in range(1,len(roots), 2)]

	#print("DEBUG: maxima: " + str(maxima))

	if plotFig == True:
		fig = plt.figure()
		plt.plot(range(len(gradCollapse)), gradCollapse, 'ro', ms=5)
		xs = np.linspace(0,len(gradCollapse), 1000)
		plt.plot(xs, spln(xs), 'g', lw=3)
		plt.plot(xs, splnDdx(xs), 'b', lw=3)
		plt.plot(range(len(gradCollapse)), np.zeros(len(gradCollapse)), 'r-', lw=1)
		plt.plot(maxima, spln(maxima), 'o', lw=1)
		plt.show()
		fig.savefig(tag + "_debug_spline_fit_to_gradient.pdf", bbox_inches='tight')	

	return maxima	

def get_TAD_boundaries_from_maxes(gradX, gradY, maxima, numToCheck, plotFig = False):
	exits = list()
	entrances = list()

	numBarcodes = gradX.shape[0]

	for i in maxima:
		leftCoord = max(i - numToCheck - 1,0)
		diagLeft = max(i - 1, 0)

		if leftCoord == diagLeft:
			continue

		upCoord = leftCoord
		diagUp = diagLeft

		exitAvg = np.sum(gradX[i,leftCoord:diagLeft]) + np.sum(gradY[upCoord:diagUp,i])
		
		if exitAvg < 0:
			exits.append(i)

		rightCoord = min(i + numToCheck + 1, numBarcodes - 1)
		diagRight = min(i + 1, numBarcodes - 1)
		
		if rightCoord == diagRight:
			continue
	
		downCoord = rightCoord
		diagDown = diagRight

		entranceAvg = np.sum(gradX[i,diagRight:rightCoord]) + np.sum(gradY[diagDown:downCoord,i])
		if entranceAvg > 0:
			entrances.append(i)			


	exits.sort()
	entrances.sort()

	if len(exits) == 0 and len(entrances) == 0:
		print("DEBUG: No TADs in this cell!")

		return [], entrances, exits
	elif len(entrances) == 0:
		return [(0,exits[0])], entrances, exits

	elif len(exits) == 0:
		return [(entrances[0],numBarcodes - 1)], entrances, exits
	else:
		tads = list()
		exitPtr = 0
		entrancePtr = 0

		if exits[0] < entrances[0]:
			tads.append((0,exits[0]))
			if exitPtr + 1 == len(exits):
				return tads, entrances, exits
			else:
				exitPtr += 1

		while True:
			if entrances[entrancePtr] < exits[exitPtr]:
				tads.append((entrances[entrancePtr], exits[exitPtr]))
				entrancePtr += 1
				exitPtr += 1
			else:
				exitPtr += 1

			
			if exitPtr == len(exits):
				if entrancePtr < len(entrances):
					tads.append((entrances[entrancePtr],numBarcodes-1))

				return tads, entrances, exits

			elif entrancePtr == len(entrances):
				return tads, entrances, exits
	
def get_best_TADs(entrances, exits, im_cor, numBarcodes, plotFig = False):
	#print("DEBUG in get_best_TADs()")
	if len(exits) == 0 and len(entrances) == 0:
		#print("DEBUG: No TADs in this cell!")

		return []
	elif len(entrances) == 0:
		#print("DEBUG: len(entrances) == 0!")
		return [(0,exits[0])]

	elif len(exits) == 0:
		#print("DEBUG: len(exits) == 0!")
		return [(entrances[0],numBarcodes - 1)]
	else:
		#print("DEBUG: in else")
		tads = list()
		exitPtr = 0
		entrancePtr = 0

		#if exits[0] < entrances[0]:
		#	tads.append((0,exits[0]))
		#	exitPtr += 1


		ent = entrances.copy()
		ent.insert(0,0)
		#ex = exits[exitPtr:].copy()
		ex = exits.copy()
		ex.append(numBarcodes-1)
		used = list()

		for start in ent:
			opts = list()
			stopOpts = list()
			#print("DEBUG start is " + str(start))
			for stop in ex:
				#print("DEBUG stop is " + str(stop))
				if stop <= start or stop in used:
					#print("DEBUG CONTINUE")
					continue
		
				corMat = im_cor[start:stop, start:stop]
				#print("DEBUG cor mat\n" + str(corMat))
				percAboveZero = (corMat > 0)
				#print("DEBUG mask mat\n" + str(percAboveZero))
				percAboveZero = np.sum(percAboveZero) / corMat.shape[0]**2

				if percAboveZero < 0.80:
					continue

				#print("DEBUG percAboveZero: " + str(percAboveZero))
				scoreCor = corMat.mean() * (stop - start) * percAboveZero
				#print("DEBUG meanCor is " + str(meanCor))
				opts.append(scoreCor)
				stopOpts.append(stop)

			if len(opts) == 0:
				#print("DEBUG CONTINUE")
				continue

			maxIdx = np.argmax(opts)
			#print("DEBUG: max cor score : " + str(opts[maxIdx]))
			bestStop = stopOpts[maxIdx]
			tads.append((start,bestStop))
			used.append(bestStop)
									
		return tads	
	
def get_TAD_boundaries_attempt_1(gradX, gradY, numToCheck):
	# overcalls
	numBarcodes = gradX.shape[0]
	
	exits = list()
	entrances = list()

	for i in range(numBarcodes):
		leftCoord = max(i - numToCheck - 1,0)
		diagLeft = max(i - 1, 0)

		if leftCoord == diagLeft:
			continue

		upCoord = leftCoord
		diagUp = diagLeft

		exitAvg = np.sum(gradX[i,leftCoord:diagLeft]) + np.sum(gradY[upCoord:diagUp,i])
		
		if exitAvg < 0:
			exits.append(i)

		rightCoord = min(i + numToCheck + 1, numBarcodes - 1)
		diagRight = min(i + 1, numBarcodes - 1)
		
		if rightCoord == diagRight:
			continue
	
		downCoord = rightCoord
		diagDown = diagRight

		entranceAvg = np.sum(gradX[i,diagRight:rightCoord]) + np.sum(gradY[diagDown:downCoord,i])
		if entranceAvg > 0:
			entrances.append(i)			


	exits.sort()
	entrances.sort()

	if len(exits) == 0 and len(entrances) == 0:
		print("DEBUG: No TADs in this cell!")

		return []
	elif len(entrances) == 0:
		return [(0,exits[0])]

	elif len(exits) == 0:
		return [(entrances[0],numBarcodes - 1)]
	else:
		tads = list()
		exitPtr = 0
		entrancePtr = 0

		if exits[0] < entrances[0]:
			tads.append((0,exits[0]))
			if exitPtr + 1 == len(exits):
				return tads
			else:
				exitPtr += 1

		while True:
			if entrances[entrancePtr] < exits[exitPtr]:
				tads.append((entrances[entrancePtr], exits[exitPtr]))
				entrancePtr += 1
				exitPtr += 1
			else:
				exitPtr += 1

			
			if exitPtr == len(exits):
				if entrancePtr < len(entrances):
					tads.append((entrances[entrancePtr],numBarcodes-1))

				return tads

			elif entrancePtr == len(entrances):
				return tads
		
def get_single_cell_tads(distmaps, sc_idx, tag, plotFig=False):
	#print("distmaps shape :", str(distmaps.shape))

	#plot_median_dist_map(distmaps[sc_idx,:,:], tag)
	gradX, gradY, im_cor = get_cor_gradient(sc_idx, distmaps, 1.5, tag, plotFig)

	#boundaries = get_TAD_boundaries_attempt_1(gradX, gradY, 8)
	gradMaxima = get_TAD_boundaries_attempt_2(gradX, gradY, tag, plotFig)

	boundaries, entrances, exits = get_TAD_boundaries_from_maxes(gradX, gradY, gradMaxima, 4, plotFig)
	#print("DEBUG: TAD boundaries count: " + str(len(boundaries)))
	#print("DEBUG: TAD boundaries:\n" + str(boundaries))

	tads = get_best_TADs(entrances, exits, im_cor, distmaps.shape[1], plotFig)

	#print("DEBUG: best TADs count: " + str(len(tads)))
	#print("DEBUG: best TADs:\n" + str(tads))

	if plotFig == True:
		plot_map_with_squares(im_cor, tads, tag)

	return tads

def get_avg_tad_corner(distmaps, tag, halfSqWidth):

	avgTadSq = np.zeros([halfSqWidth*2, halfSqWidth*2])
	avgTadCount = 0

	for d in range(distmaps.shape[0]):
		currTads = get_single_cell_tads(distmaps, d, tag)

		for tadx, tady in currTads:
			rowTop = tadx - halfSqWidth
			rowBottom = tadx + halfSqWidth
		
			colLeft = tady - halfSqWidth
			colRight = tady + halfSqWidth

			if rowTop < 0 or colLeft < 0 or rowTop >= distmaps.shape[1] or colRight >= distmaps.shape[1]:
				continue

			subsetMap = distmaps[d, rowTop:rowBottom, colLeft:colRight]

			avgTadSq += subsetMap
			avgTadCount += 1

	avgTadSq /= avgTadCount

	plot_map(avgTadSq, tag)

	return avgTadSq			 

def main():
	if len(sys.argv) < 3:
		print("usage: single_cell_tads.py <tsv file> <pdf tag>\n")
		sys.exit(1)

	filename = sys.argv[1]
	tag = sys.argv[2]
	dat = read_data(filename)
	distmap, distmaps = get_dist_map_aggregate(dat)
	plot_median_dist_map(distmap, tag)
	#im_cor = get_corrected_cor_mat(distmap, distmaps, tag)

	


	sc_idx = 10
	get_single_cell_tads(distmaps, sc_idx, tag + "_single_cell_" + str(sc_idx), plotFig=True)

	# this doesn't work
	#avgCorner = get_avg_tad_corner(distmaps, tag, 4)
	
	# gaussian filter the map
#	gaussMap = ndimage.gaussian_filter(distmaps[sc_idx,:,:], sigma=0.5)
	#gaussMap = nan_gaussian_filter(distmaps[sc_idx,:,:], sigma=1.5)
	#plot_median_dist_map(gaussMap, tag + "_single_cell_gauss_" + str(sc_idx))

	#sc_norm_mat = get_single_cell_norm_mat(sc_idx, distmaps, tag + "_sc_norm_" + str(sc_idx))
	#sc_cor_mat = get_single_cell_cor_mat(sc_idx, distmaps, tag + "_sc_norm_cor_" + str(sc_idx))
	#gaussMapCor = nan_gaussian_filter(sc_cor_mat, sigma=1.5)
	#plot_median_dist_map(gaussMap, tag + "_single_cell_gauss_cor_" + str(sc_idx))

	#gradient = get_sorted_matrix_gradient(gaussMapCor, tag)
	
	
main()
