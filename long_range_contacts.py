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

def get_long_range_stats(distmap, e, k):
	maxCoord = distmap.shape[0]

	rowVals = list()
	colVals = list()

	for i in range(0,maxCoord - e):
		for j in range(e + k):
			rowVals.append(i)
			colVals.append(j)		

	vals = distmap[rowVals,colVals]

	meanVal = np.nanmean(vals)
	maxVal = np.nanmax(vals)
	medianVal = np.nanmedian(vals)
	top90Val = np.percentile(np.sort(vals), 90)

	return {'meanVal' : meanVal, 'maxVal' : maxVal, 'medianVal': medianVal, 'top90Val' : top90Val}

def get_long_range_stats_sliding(distmap, k):
	maxCoord = distmap.shape[0]

	rowVals = list()
	colVals = list()
	dataVals = list()

	for e in range(0, maxCoord - k, k):
		rowVals.append(list())
		colVals.append(list())
		#print("DEBUG length of rowVals is " + str(len(rowVals)) + " e is " + str(e))
		for i in range(0,maxCoord - e):
			for j in range(e + k):
				rowVals[e//k].append(i)
				colVals[e//k].append(j)		

		vals = distmap[rowVals[e//k],colVals[e//k]]

		meanVal = np.nanmean(vals)
		maxVal = np.nanmax(vals)
		medianVal = np.nanmedian(vals)
		top90Val = np.percentile(np.sort(vals), 90)
		dataVals.append(list())
		#print("DEBUG vals length is " + str(len(vals)))
		dataVals[e//k].extend(vals.tolist())

	#dataVals = np.array(dataVals)
	#print("DEBUG dataVals len is " + str(len(dataVals)))	
	#print("DEBUG dataVals[0] len is " + str(len(dataVals[0])))	
	
	return dataVals
	#return {'meanVal' : meanVal, 'maxVal' : maxVal, 'medianVal': medianVal, 'top90Val' : top90Val}

def get_upper_triangle(distmap):
	# this weird data structure is required for plotting
	result = list()
	result.append(list())

	idx = np.triu_indices(distmap.shape[0])
	result[0] = distmap[idx].tolist()
	return result

def draw_plot(data, offset,edge_color, fill_color, ax):

	#print("len data " + str(len(data)))
	#print("len data[0] " + str(len(data[0])))

	max_dat = 0
	for i in range(len(data)):
		if len(data[i]) > max_dat:
			max_dat = len(data[i])

	#print("max_dat is " + str(max_dat))

	plotDat = np.zeros([max_dat, len(data)])

	plotDat[:] = np.nan

	for i in range(len(data)):
		#print("len data[i] " + str(len(data[i])) + " i " + str(i))
		plotDat[0:len(data[i]),i] = np.array(data[i])

	#print("debug: shape plotDat : " + str(plotDat.shape))

	mask = ~np.isnan(plotDat)
	filtered_data = [d[m] for d, m in zip(plotDat.T, mask.T)]


	pos = np.arange(len(data))+offset 
	bp = ax.boxplot(filtered_data, positions= pos, widths=0.3, patch_artist=True, manage_xticks=False)
	for element in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
		plt.setp(bp[element], color=edge_color)
	for patch in bp['boxes']:
		patch.set(facecolor=fill_color)

	return bp

def make_paired_boxplots(dat1, dat2, labels, tag):
	fig, ax = plt.subplots()
	
	fig, ax = plt.subplots()
	p1 = draw_plot(dat1, -0.2, "red", "white", ax)
	p2 = draw_plot(dat2, +0.2,"blue", "white", ax )
	ax.legend([p1["boxes"][0], p2["boxes"][0]], [labels[0], labels[1]], loc='upper right')

	maxTicks = max(len(dat1), len(dat2))

	plt.xticks(np.arange(maxTicks))
	plt.show()
	fig.savefig(tag + "_boxplots.pdf")

def B_A_compartments(datAuxin, datUntreat, tag):
	auxinb1 = (0,23)
	auxina1 = (23,53)
	auxinb2 = (53,65)
		
	untreatb1 = (0,28)
	untreata1 = (28,50)
	untreatb2 = (50,65)

	# upper triangle

	resultsB1AuxinUpperTri = get_upper_triangle(datAuxin[auxinb1[0]:auxinb1[1], auxinb1[0]:auxinb1[1]])
	resultsA1AuxinUpperTri = get_upper_triangle(datAuxin[auxina1[0]:auxina1[1], auxina1[0]:auxina1[1]])
	resultsB2AuxinUpperTri = get_upper_triangle(datAuxin[auxinb2[0]:auxinb2[1], auxinb2[0]:auxinb2[1]])

	resultsB1UntreatUpperTri = get_upper_triangle(datUntreat[untreatb1[0]:untreatb1[1], untreatb1[0]:untreatb1[1]])
	resultsA1UntreatUpperTri = get_upper_triangle(datUntreat[untreata1[0]:untreata1[1], untreata1[0]:untreata1[1]])
	resultsB2UntreatUpperTri = get_upper_triangle(datUntreat[untreatb2[0]:untreatb2[1], untreatb2[0]:untreatb2[1]])

	make_paired_boxplots(resultsB1AuxinUpperTri, resultsB1UntreatUpperTri, ['Auxin', 'Untreated'], tag + "_B1_AuxinUpperTrivsUntreatUpperTri")
	make_paired_boxplots(resultsA1AuxinUpperTri, resultsA1UntreatUpperTri, ['Auxin', 'Untreated'], tag + "_A1_AuxinUpperTrivsUntreatUpperTri")
	make_paired_boxplots(resultsB2AuxinUpperTri, resultsB2UntreatUpperTri, ['Auxin', 'Untreated'], tag + "_B2_AuxinUpperTrivsUntreatUpperTri")
	
	make_paired_boxplots(resultsB1AuxinUpperTri, resultsA1AuxinUpperTri, ['B1', 'A1'], tag + "_AuxinUpperTri_B1vsA1")
	make_paired_boxplots(resultsB2AuxinUpperTri, resultsA1AuxinUpperTri, ['B2', 'A1'], tag + "_AuxinUpperTri_B2vsA1")
	
	make_paired_boxplots(resultsB1UntreatUpperTri, resultsA1UntreatUpperTri, ['B1', 'A1'], tag + "_UntreatUpperTri_B1vsA1")
	make_paired_boxplots(resultsB2UntreatUpperTri, resultsA1UntreatUpperTri, ['B2', 'A1'], tag + "_UntreatUpperTri_B2vsA1")

	# sliding

	k=5

	tag = tag + "_k_" + str(k)

	resultsB1Auxin = get_long_range_stats_sliding(datAuxin[auxinb1[0]:auxinb1[1], auxinb1[0]:auxinb1[1]], k)
	resultsA1Auxin = get_long_range_stats_sliding(datAuxin[auxina1[0]:auxina1[1], auxina1[0]:auxina1[1]], k)
	resultsB2Auxin = get_long_range_stats_sliding(datAuxin[auxinb2[0]:auxinb2[1], auxinb2[0]:auxinb2[1]], k)

	resultsB1Untreat = get_long_range_stats_sliding(datUntreat[untreatb1[0]:untreatb1[1], untreatb1[0]:untreatb1[1]], k)
	resultsA1Untreat = get_long_range_stats_sliding(datUntreat[untreata1[0]:untreata1[1], untreata1[0]:untreata1[1]], k)
	resultsB2Untreat = get_long_range_stats_sliding(datUntreat[untreatb2[0]:untreatb2[1], untreatb2[0]:untreatb2[1]], k)
	
	make_paired_boxplots(resultsB1Auxin, resultsB1Untreat, ['Auxin', 'Untreated'], tag + "_B1_AuxinvsUntreat")
	make_paired_boxplots(resultsA1Auxin, resultsA1Untreat, ['Auxin', 'Untreated'], tag + "_A1_AuxinvsUntreat")
	make_paired_boxplots(resultsB2Auxin, resultsB2Untreat, ['Auxin', 'Untreated'], tag + "_B2_AuxinvsUntreat")
	
	make_paired_boxplots(resultsB1Auxin, resultsA1Auxin, ['B1', 'A1'], tag + "_Auxin_B1vsA1")
	make_paired_boxplots(resultsB2Auxin, resultsA1Auxin, ['B2', 'A1'], tag + "_Auxin_B2vsA1")
	
	make_paired_boxplots(resultsB1Untreat, resultsA1Untreat, ['B1', 'A1'], tag + "_Untreat_B1vsA1")
	make_paired_boxplots(resultsB2Untreat, resultsA1Untreat, ['B2', 'A1'], tag + "_Untreat_B2vsA1")



def main():
	if len(sys.argv) < 3:
		print("usage: loops.py <Auxin tsv file> <Untreatd tsv file> <pdf tag>\n")
		sys.exit(1)
	filenameAuxin = sys.argv[1]
	filenameUntreated = sys.argv[2]
	tag = sys.argv[3]

	datAuxin = read_data(filenameAuxin)
	distmapAuxin, scTracesAuxin = get_dist_map_aggregate(datAuxin)

	datUntreated = read_data(filenameUntreated)
	distmapUntreated, scTracesUntreated = get_dist_map_aggregate(datUntreated)

	#e = 50
	#k = 1

	#resultsAuxin = get_long_range_stats(distmapAuxin, e, k)
	#resultsUntreated = get_long_range_stats(distmapUntreated, e, k)
	#print("Auxin:\n", resultsAuxin)
	#print("Untreated:\n", resultsUntreated)

	# sliding box
	#k = 5
	#print("DEBUG starting sliding box Auxin")
	#resultsAuxinSlide = get_long_range_stats_sliding(distmapAuxin, k)

	#print("DEBUG starting sliding box Untreated")
	#resultsUntreatedSlide = get_long_range_stats_sliding(distmapUntreated, k)

	#make_paired_boxplots(resultsAuxinSlide, resultsUntreatedSlide, ['Auxin', 'Untreated'], tag + "_k_" + str(k))

	resultsAuxinUpperTri = get_upper_triangle(distmapAuxin)
	resultsUntreatedUpperTri = get_upper_triangle(distmapUntreated)

	make_paired_boxplots(resultsAuxinUpperTri, resultsUntreatedUpperTri, ['Auxin', 'Untreated'], tag + "UpperTri")



	
	# and make for individual B / A compartments ones
	#B_A_compartments(distmapAuxin, distmapUntreated, tag)

	# make whole upper triangle boxplots

main()
