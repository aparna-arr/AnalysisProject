#!/share/software/user/open/python/3.6.1/bin/python3

import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ORCAutils import *
from NormFuncs import *


def get_asphericity(dat, compartments):
	asp_compartments = np.zeros((dat.shape[0],3))
	for c in range(dat.shape[0]):
		compartNum = 0
		for i,j in compartments:
			smallDat = dat[c,i:j,:]
			asp = calculate_asphericity(smallDat)	
			asp_compartments[c,compartNum] = asp
			compartNum += 1

	
	#asp_compartments = asp_compartments[~np.isnan(asp_compartments).any(axis=1)]

	avg_asp_comp = np.nanmean(asp_compartments,axis=0)
	stderr = stats.sem(asp_compartments, axis=0, nan_policy='omit')
	#print("asphericity compartments (avg):" + str(avg_asp_comp))	
	#print("stderr: " + str(stderr))
	return avg_asp_comp, stderr

def get_asphericity_compartments(dat1, compartments1, dat2, compartments2, tag):

	avgAsp1, stderr1 = get_asphericity(dat1, compartments1)
	avgAsp2, stderr2 = get_asphericity(dat2, compartments2)

	width = 0.4
	xvals1 = [0,1,2]
	xvals2 = [0+width,1+width,2+width]

	fig = plt.figure()
	plt.title('Asphericity')
	p1 = plt.bar(xvals1, avgAsp1, color='r', width=width, yerr=stderr1)
	p2 = plt.bar(xvals2, avgAsp2, color='b', width=width, yerr=stderr2)
	plt.xticks([0+width/2,1+width/2, 2+width/2],['B1', 'A1', 'B2'])
	plt.ylim(0,0.5)
	plt.legend((p1[0], p2[0]), ('Auxin', 'Untreated'))
	plt.show()
	fig.savefig(tag + "_asphericity_across_compartments.pdf", bbox_inches='tight')

def get_end_to_end_dist_compartments(dat1, compartments1, dat2, compartments2, tag):
	compIdx = 0
	for c in range(3):

		i1, j1 = compartments1[c]	
		smallDat1 = dat1[:,i1:j1,:]
		endToEnds1 = calc_endToEndDist(smallDat1)

		i2, j2 = compartments2[c]	
		smallDat2 = dat2[:,i2:j2,:]
		endToEnds2 = calc_endToEndDist(smallDat2)

		fig = plt.figure()
		plt.title("End-To-End Distance over Subchains")
		p1 = plt.plot(np.nanmedian(endToEnds1, axis=0),'ro-')
		p2 = plt.plot(np.nanmedian(endToEnds2, axis=0),'bo-')

		plt.legend((p1[0], p2[0]), ('Auxin', 'Untreated'))

		plt.show()
		fig.savefig(tag + "_Compartment_" + str(compIdx) + "_end_to_end_dist.pdf", bbox_inches='tight')
		compIdx += 1


def get_mean_dist(distmaps, compartments):
	mean_mat = np.nanmean(distmaps,axis=0)
	#np.fill_diagonal(mean_mat, np.nan)
	mat_size = mean_mat.shape[0]
	lower_tri_idx = np.tril_indices(mat_size) # includes diagonal
	mean_mat[lower_tri_idx] = np.nan
	# get avg dist from points in one compartment to points in other compartment
	b1 = compartments[0]
	a1 = compartments[1]
	b2 = compartments[2]

	b1a1_mat = mean_mat[b1[0]:b1[1],a1[0]:a1[1]]
	#print("b1a1_mat\n" + str(b1a1_mat))
	if np.isnan(b1a1_mat).all():
		b1a1_mat = mean_mat[a1[0]:a1[1],b1[0]:b1[1]]
		
	
	b2a1_mat = mean_mat[b2[0]:b2[1],a1[0]:a1[1]]

	if np.isnan(b2a1_mat).all():
		b2a1_mat = mean_mat[a1[0]:a1[1],b2[0]:b2[1]]
	#print("b2a1_mat\n" + str(b2a1_mat))
	
	b1a1_avg = np.nanmean(b1a1_mat.flatten())
	b1a1_stderr = stats.sem(b1a1_mat.flatten(), axis=0, nan_policy='omit')
	b2a1_avg = np.nanmean(b2a1_mat.flatten())
	b2a1_stderr = stats.sem(b2a1_mat.flatten(), axis=0, nan_policy='omit')

	return [b1a1_avg, b2a1_avg], [b1a1_stderr, b2a1_stderr]

def get_mean_dist_B_to_A(distmaps1, compartments1, distmaps2, compartments2, tag):

	avgs1, stderrs1 = get_mean_dist(distmaps1, compartments1)
	avgs2, stderrs2 = get_mean_dist(distmaps2, compartments2)

	width = 0.4
	xvals1 = [0,1]
	xvals2 = [0+width,1+width]

	#print("b1a1 mean dist:", str(b1a1_avg), "se:", str(b1a1_stderr))
	#print("b2a1 mean dist:", str(b2a1_avg), "se:", str(b2a1_stderr))
	fig = plt.figure()
	plt.title('Mean distance between points in B compartment to points in A compartment')
	p1 = plt.bar(xvals1, avgs1, yerr=stderrs1, color="r", width=width)
	p2 = plt.bar(xvals2, avgs2, yerr=stderrs2, color="b", width=width)
	plt.ylim(0,900)
	plt.xticks([0+width/2,1+width/2],['B1 to A1', 'B2 to A1'])
	plt.legend((p1[0], p2[0]), ('Auxin', 'Untreated'))
	plt.show()
	fig.savefig(tag + "_mean_distance_between_compartments.pdf", bbox_inches='tight')

def main():
	if len(sys.argv) < 3:
		print("usage: compartments.py <tsv file 1 Auxin> <tsv file 2 Untreated> <pdf tag>\n")
		sys.exit(1)

	filename1 = sys.argv[1]
	filename2 = sys.argv[2]
	tag = sys.argv[3]

	dat1 = read_data(filename1)
	dat2 = read_data(filename2)

	distmap1, distmaps1 = get_dist_map_aggregate(dat1)
	distmap2, distmaps2 = get_dist_map_aggregate(dat2)

	plot_median_dist_map(distmap1, tag + "file1")
	plot_median_dist_map(distmap2, tag + "file2")

	im_cor1 = get_corrected_cor_mat(distmap1, distmaps1, tag + "file1")
	im_cor2 = get_corrected_cor_mat(distmap2, distmaps2, tag + "file2")

	gradient1 = get_sorted_matrix_gradient(im_cor1, tag + "file1")
	gradient2 = get_sorted_matrix_gradient(im_cor2, tag + "file2")

	compartments1 = get_top_two_boundaries(gradient1, tag + "file1")
	compartments2 = get_top_two_boundaries(gradient2, tag + "file2")

	get_asphericity_compartments(dat1, compartments1, dat2, compartments2, tag)
	get_end_to_end_dist_compartments(dat1, compartments1, dat2, compartments2, tag)
	get_mean_dist_B_to_A(distmaps1, compartments1, distmaps2, compartments2, tag)

main()
