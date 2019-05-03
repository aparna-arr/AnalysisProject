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


def get_dist_of_contact(scTraces, loopCoords, zoom, loopContact, loopNoContact, tag, vmin=-850, vmax=0):
	inContact = list()
	outOfContact = list()

	closeNotContact = list()

	distValues = list()

	for c in range(scTraces.shape[0]):
		if scTraces[c][loopCoords[0]][loopCoords[1]] < loopContact:
			inContact.append(c)
		elif scTraces[c][loopCoords[0]][loopCoords[1]] >= loopNoContact:
			outOfContact.append(c)
		else:
			closeNotContact.append(c)

		distValues.append(scTraces[c][loopCoords[0]][loopCoords[1]])

	print("Len inContact:" + str(len(inContact)))	
	print("Len outOfContact:" + str(len(outOfContact)))	

	inContact = np.array(inContact)
	outOfContact = np.array(outOfContact)

	medMatContact = np.nanmedian(scTraces[inContact,:,:], axis=0)
	medMatNoContactAll = np.nanmedian(scTraces[outOfContact,:,:], axis=0)

	np.random.seed(1)

	randomIdxs = np.random.choice(len(outOfContact), len(inContact), replace=False)
	medMatNoContactDownsample = np.nanmedian(scTraces[outOfContact[randomIdxs],:,:], axis=0)

	distValues = np.array(distValues)
	distValues = distValues[~np.isnan(distValues)]
	randPlotIdx = np.random.choice(len(distValues),1000, replace=False)

	# color bar scaling
	#vmin = np.nanmin(distValues)
	#vmax = np.nanmax(distValues)

	# full maps
	plot_median_dist_map(medMatContact, tag + "_Contact", vmin=vmin, vmax=vmax)	
	plot_median_dist_map(medMatNoContactAll, tag + "_NoContactAll", vmin=vmin, vmax=vmax)	
	plot_median_dist_map(medMatNoContactDownsample, tag + "_NoContactDownsample", vmin=vmin, vmax=vmax)	

	# zoom in maps
	plot_median_dist_map(medMatContact[zoom[0]:zoom[1], zoom[0]:zoom[1]], tag + "_Contact_Zoom", vmin=vmin, vmax=vmax)	
	plot_median_dist_map(medMatNoContactAll[zoom[0]:zoom[1],zoom[0]:zoom[1]], tag + "_NoContactAll_Zoom", vmin=vmin, vmax=vmax)	
	plot_median_dist_map(medMatNoContactDownsample[zoom[0]:zoom[1],zoom[0]:zoom[1]], tag + "_NoContactDownsample_Zoom", vmin=vmin, vmax=vmax)	


	diffMapDownsample = medMatNoContactDownsample - medMatContact
	print("diffMapDownsample value at loop point: " + str(diffMapDownsample[loopCoords[0],loopCoords[1]]))

	plot_median_dist_map(-diffMapDownsample, tag + "_NoLoop_minus_Loop_Downsample", norm=0)

	diffMapAll = medMatNoContactAll - medMatContact
	print("diffMapAll value at loop point: " + str(diffMapAll[loopCoords[0],loopCoords[1]]))

	plot_median_dist_map(-diffMapAll, tag + "_NoLoop_minus_Loop_All", norm=0)


	# zoom in difference maps
	plot_median_dist_map(-diffMapDownsample[zoom[0]:zoom[1], zoom[0]:zoom[1]], tag + "_NoLoop_minus_Loop_Downsample_Zoom", norm=0)
	plot_median_dist_map(-diffMapAll[zoom[0]:zoom[1], zoom[0]:zoom[1]], tag + "_NoLoop_minus_Loop_All_Zoom", norm=0)
	

	f=plt.figure()
	y = distValues[randPlotIdx]
	x = np.random.normal(1, 0.04, size=len(y))
	plt.boxplot(distValues)
	plt.plot(x, y, '.', alpha=0.2)
	plt.title('distance values for loop coordinates')
	plt.show()
	f.savefig(tag + "_boxplot.pdf")

def plotLoops(distmap, scTraces, coords, loopCoords, tag, vmin, vmax):
	plot_median_dist_map(distmap,tag, vmin=vmin, vmax=vmax)
	plot_median_dist_map(distmap[coords[0]:coords[1], coords[0]:coords[1]], tag + "_coords_" + str(coords[0]) + "_" + str(coords[1]), vmin=vmin, vmax=vmax)


	loopContact = 150 # nm
	loopNoContact = 150 # nm

	get_dist_of_contact(scTraces, loopCoords, coords, loopContact, loopNoContact, tag + "_LoopContactVal_" + str(loopContact) + "_NoLoopVal_" + str(loopNoContact), vmin=vmin, vmax=vmax)

	loopContact = 150 # nm
	loopNoContact = 300 # nm

	get_dist_of_contact(scTraces, loopCoords, coords, loopContact, loopNoContact, tag + "_LoopContactVal_" + str(loopContact) + "_NoLoopVal_" + str(loopNoContact), vmin=vmin, vmax=vmax)

	loopContact = 150 # nm
	loopNoContact = 450 # nm

	get_dist_of_contact(scTraces, loopCoords, coords, loopContact, loopNoContact, tag + "_LoopContactVal_" + str(loopContact) + "_NoLoopVal_" + str(loopNoContact), vmin=vmin, vmax=vmax)

def main():
	if len(sys.argv) < 3:
		print("usage: loops.py <tsv file> <pdf tag>\n")
		sys.exit(1)

	filename = sys.argv[1]
	tag = sys.argv[2]
	dat = read_data(filename)
	distmap, scTraces = get_dist_map_aggregate(dat)

	coords = (6,29)
	loopCoords = (11,17)
	plotLoops(distmap, scTraces, coords, loopCoords, tag + "_loop1", vmin=-850, vmax=0)

	coords = (6,29)
	loopCoords = (17,24)
	plotLoops(distmap, scTraces, coords, loopCoords, tag + "_loop2", vmin=-850, vmax=0)

	coords = (6,29)
	loopCoords = (11,24)
	plotLoops(distmap, scTraces, coords, loopCoords, tag + "_loop3", vmin=-850, vmax=0)
	
main()	
