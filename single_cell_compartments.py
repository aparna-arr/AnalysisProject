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

def plot_median_dist_map(distmap, tag):
	#Plot the median distance map
	f = plt.figure()
	plt.imshow(-distmap,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_med_dist.pdf", bbox_inches='tight')

def plot_gradient(grad, tag):
	f = plt.figure()
	plt.imshow(grad,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_gradient.pdf", bbox_inches='tight')

def plot_gradient_collpase_xy(gradXcollapse, gradYcollapse, tag):
	fig = plt.figure()
	plt.title('Gradient Collapse')
	plt.plot(gradXcollapse,'ro-')
	plt.plot(gradYcollapse,'bo-')
	plt.show()
	fig.savefig(tag + "_XY_gradient_collapse_debug.pdf", bbox_inches='tight')

def plot_gradient_collpase(gradCollapse, tag):
	fig = plt.figure()
	plt.title('Gradient Collapse')
	plt.plot(gradCollapse,'ro-')
	plt.show()
	fig.savefig(tag + "_gradient_collapse_debug.pdf", bbox_inches='tight')

def get_sorted_matrix_gradient(im_cor, tag):
	gradient_x = np.array(np.gradient(im_cor,axis=0))

	plot_gradient(gradient_x, tag + "_X")

	gradient_y = np.array(np.gradient(im_cor,axis=1))
	
	plot_gradient(gradient_y, tag + "_Y")

	gradient_x_collapse = np.sum(np.abs(gradient_x),axis=1)
	gradient_y_collapse = np.sum(np.abs(gradient_y),axis=0)

	gradient_with_labels = np.hstack((gradient_x_collapse.reshape((len(gradient_x_collapse),1)),np.arange(len(gradient_x_collapse)).reshape((len(gradient_x_collapse),1))))

	a = gradient_with_labels
	sorted_ar = a[a[:,0].argsort()[::-1]]

	print("DEBUG: sorted gradient\n" + str(sorted_ar[:2, :]))

	return sorted_ar

def get_top_two_boundaries(sortedGradient, tag):
	if sortedGradient[1][1] > sortedGradient[0][1]:
		compartments = [(0,int(sortedGradient[0][1])), (int(sortedGradient[0][1]),int(sortedGradient[1][1])), (int(sortedGradient[1][1]), len(sortedGradient))]
	else:
		compartments = [(0,int(sortedGradient[1][1])), (int(sortedGradient[1][1]),int(sortedGradient[0][1])), (int(sortedGradient[0][1]), len(sortedGradient))]

	print("compartments\n" + str(compartments))
	return compartments

def get_asphericity_compartments(dat, compartments, tag):
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
	print("asphericity compartments (avg):" + str(avg_asp_comp))	
	print("stderr: " + str(stderr))

	fig = plt.figure()
	plt.title('Asphericity')
	plt.bar([0,1,2], avg_asp_comp, tick_label=['Compartment1', 'Compartment2', 'Compartment3'], yerr=stderr)
	plt.ylim(0,0.5)
	plt.show()
	fig.savefig(tag + "_asphericity_across_compartments.pdf", bbox_inches='tight')

def get_end_to_end_dist_compartments(dat, compartments, tag):
	compIdx = 0
	for i,j in compartments:
		smallDat = dat[:,i:j,:]
		endToEnds = calc_endToEndDist(smallDat)

		fig = plt.figure()
		plt.title("End-To-End Distance over Subchains")
		plt.plot(np.nanmedian(endToEnds, axis=0),'ro-')
		plt.show()
		fig.savefig(tag + "_Compartment_" + str(compIdx) + "_end_to_end_dist.pdf", bbox_inches='tight')
		compIdx += 1

def get_mean_dist_B_to_A(distmaps, compartments, tag):
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

	print("b1a1 mean dist:", str(b1a1_avg), "se:", str(b1a1_stderr))
	print("b2a1 mean dist:", str(b2a1_avg), "se:", str(b2a1_stderr))
	fig = plt.figure()
	plt.title('Mean distance between points in B compartment to points in A compartment')
	plt.bar([0,1], [b1a1_avg, b2a1_avg], tick_label=['B1 to A1', 'B2 to A1'], yerr=[b1a1_stderr, b2a1_stderr])
	plt.ylim(0,900)
	plt.show()
	fig.savefig(tag + "_mean_distance_between_compartments.pdf", bbox_inches='tight')

def main():
	if len(sys.argv) < 3:
		print("usage: single_cell_compartments.py <tsv file> <pdf tag>\n")
		sys.exit(1)

	filename = sys.argv[1]
	tag = sys.argv[2]
	dat = read_data(filename)
	distmap, distmaps = get_dist_map_aggregate(dat)
	plot_median_dist_map(distmap, tag)
	im_cor = get_corrected_cor_mat(distmap, distmaps, tag)
	gradient = get_sorted_matrix_gradient(im_cor, tag)
	compartments = get_top_two_boundaries(gradient, tag)
	get_asphericity_compartments(dat, compartments, tag)
	get_end_to_end_dist_compartments(dat, compartments, tag)
	get_mean_dist_B_to_A(distmaps, compartments, tag)

main()
