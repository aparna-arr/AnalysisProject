#!/share/software/user/open/python/3.6.1/bin/python3

import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


## compartment functions ##
## from Bogdan's code

def get_norm(im_mean,func=np.nanmean):
	"""Go across off diagonal elements of the matrix <im_mean> and average with <func> across them"""
	dic_norm={}
	for i in range(len(im_mean)):
		for j in range(i+1):
			key = np.abs(i-j)
			if key not in dic_norm: dic_norm[key]=[]
			dic_norm[key].append(im_mean[i,j])
	dic_norm2 = {key:func(dic_norm[key]) for key in dic_norm.keys()}
	x = np.sort([x for x in dic_norm2.keys()])
	y = [dic_norm2[key] for key in x]
	return np.array(x),np.array(y)

def corr_coef(x_,y_,print_err=False):
	x=np.ravel(x_)
	y=np.ravel(y_)
	keep=(np.abs(x)!=np.inf)&(np.abs(y)!=np.inf)&(np.isnan(x)==False)&(np.isnan(y)==False)
	x=x[keep]
	y=y[keep]
	A = np.vstack([x, np.ones(len(x))]).T
	m, c = np.linalg.lstsq(A, y)[0]
	if print_err:
		model = sm.OLS(y,A)
		result = model.fit()
		return np.corrcoef([x,y])[0,1],c,m,result.bse
	return np.corrcoef([x,y])[0,1],c,m


def get_gen_pos_gen_vals(mats):
	"""Given list of single cell distance matrices, find the population-average median, then group based on genomic distance and compute medians across groups.
	Perform fit in log space 
	"""
	im_dist = np.nanmedian(mats,0)
	gen_pos,gen_vals = get_norm(im_dist,func=np.nanmedian)
	ro,c,m=corr_coef(np.log(gen_pos),np.log(gen_vals))
	gen_vals = np.exp(c)*gen_pos**m

	print("gen_vals:[" + str(gen_vals) + "]")
	print("gen_vals.shape:[" + str(gen_vals.shape) + "]")
	print("gen_pos:[" + str(gen_pos) + "]")
	return gen_pos,gen_vals

def nan_corrcoef(x,y):
	x_ = np.array(x)
	y_ = np.array(y)
	keep = (np.isinf(x_)==False)&(np.isinf(y_)==False)&(np.isnan(x_)==False)&(np.isnan(y_)==False)
	if np.sum(keep)>2:
		return np.corrcoef(x_[keep],y_[keep])[0,1]
	return 0

def cor_mat(im_log):
	im_log = np.array(im_log)
	im_cor = np.zeros(im_log.shape)
	for i in range(len(im_cor)):
		for j in range(i+1):
			im_cor[i,j]=nan_corrcoef(im_log[i],im_log[j])
			im_cor[j,i]=im_cor[i,j]
	return im_cor

# Bogdan's code did not include this so I wrote what I thought it was
def perform_norm(mat, gen_pos, gen_vals):
	norm_mat = np.zeros(mat.shape)
	for i in range(mat.shape[0]):
		for j in range(mat.shape[1]):
			gen_dist = int(abs(i - j))
			exp_dist = gen_vals[gen_dist]
			real_dist = mat[i][j]
			norm_dist = real_dist / exp_dist
			norm_mat[i][j] = norm_dist

	return norm_mat

def get_cor_matrix(mat, tag, gen_pos=None,gen_vals=None,plt_val=True):
	mat_ = np.array(mat)
	
	if plt_val:
		plt.figure()
		plt.title('original distance matrix')
		plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
		plt.colorbar()
		plt.show()
	mat_[range(len(mat_)),range(len(mat_))]=np.nan
	
	##normalize for polymer effect
	if gen_pos is not None:
		mat_ = perform_norm(mat_,gen_pos,gen_vals)
		#mat_ = np.log(mat_)

	mat_[np.isinf(mat_)]=np.nan
	if plt_val:
		f=plt.figure()
		plt.title('distance normalized matrix')
		plt.imshow(-mat_,interpolation='nearest',cmap='seismic')
		plt.colorbar()
		plt.show()  
		f.savefig(tag + "_distance_norm_mat.pdf", bbox_inches='tight')

	#compute correlation matrix
	mat_ = cor_mat(mat_)

	if plt_val:
		f=plt.figure()
		plt.title('correlation matrix')
		plt.imshow(mat_,interpolation='nearest',cmap='seismic')
		plt.colorbar()
		plt.show()
		f.savefig(tag + "_cormat_debug.pdf", bbox_inches='tight')
	return mat_

def get_dist_map_aggregate(dat):
	distance_mats=np.array(list(map(squareform,list(map(pdist,dat)))))
	med_mat = np.nanmedian(distance_mats,axis=0)

	return med_mat, distance_mats

## end of borrowed functions

def calculate_asphericity(datWNA):
	dat = datWNA[~np.isnan(datWNA).any(axis=1)]

	if dat.shape[0] < 3:
		return np.nan
	
	allCm = np.mean(dat,axis=0)

	tensor = np.zeros([3, 3])
	for coord in range(np.size(dat,0)):
		currPos = dat[coord,:]
		for i in range(3):
			for j in range(3):
				elem1 = currPos[i] - allCm[i]
				elem2 = currPos[j] - allCm[j]
				value = elem1 * elem2
				tensor[i,j] += value


	tensorNorm = tensor / np.size(dat,0)
	w, trash = LA.eig(tensorNorm)

	w2 = sorted(w,reverse=True)
	sphericity = 1 - 3 * (w2[0]*w2[1] +  w2[0]*w2[2] + w2[1]*w2[2]) / (w[0] + w[1] + w[2]) ** 2

	return sphericity



def read_data(filename):
	fh = open(filename, "r")
	dat = list() # initially read it into a list

	firstLine = True
	NUM_COORDS = 3
	maxChr = 0
	maxBarcode = 0

	for line in fh:
		if firstLine == True:
			firstLine = False
			continue

		elem = line.rstrip().split('\t')

		elemConv = [int(x) if (x != 'nan') else np.nan for x in elem]
		dat.append(elemConv)	

	fh.close()

	datm = np.array(dat)

	max_chr = int(datm[:,0].max())
	max_bar = int(datm[:,1].max())

	datm = datm[:,2:5].reshape([max_chr, max_bar, NUM_COORDS])
	
	print("DEBUG: matrix shape post reshape: " + str(datm.shape))

	return datm

def calc_endToEndDist(dat):	
	endToEnd = np.zeros((dat.shape[0], dat.shape[1] - 3))
	for c in range(dat.shape[0]):
		lenIdx = 0
		for length in range(3,dat.shape[1]):
			currEndToEnd = list()
			i = 0
			while (i + length <= dat.shape[1]):
				start = i
				end = i + length
				i += length
				subDat = dat[c,start:end,:]
			
				x1 = subDat[0][0]
				y1 = subDat[0][1]
				z1 = subDat[0][2]
				x2 = subDat[length-1][0]
				y2 = subDat[length-1][1]
				z2 = subDat[length-1][2]
	
				if not np.isnan(x1) and not np.isnan(x2):
					dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
					currEndToEnd.append(dist)
			endToEnd[c][lenIdx] = np.nanmedian(currEndToEnd) if len(currEndToEnd) > 0 else np.nan 
			lenIdx += 1

	return endToEnd

def main():
	if len(sys.argv) < 3:
		print("usage: single_cell_compartments.py <tsv file> <pdf tag>\n")
		sys.exit(1)

	filename = sys.argv[1]
	tag = sys.argv[2]

	dat = read_data(filename)
	distmap, distmaps = get_dist_map_aggregate(dat)
	
	#Plot the median distance map
	f = plt.figure()
	plt.imshow(-distmap,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_med_dist.pdf", bbox_inches='tight')

	gen_pos, gen_vals = get_gen_pos_gen_vals(distmaps)
	im_cor = get_cor_matrix(distmap, tag, gen_pos, gen_vals)

	gradient_x = np.array(np.gradient(im_cor,axis=0))

	#Plot the median distance map
	f = plt.figure()
	plt.imshow(gradient_x,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_gradient_x.pdf", bbox_inches='tight')

	gradient_y = np.array(np.gradient(im_cor,axis=1))

	#Plot the median distance map
	f = plt.figure()
	plt.imshow(gradient_y,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_gradient_y.pdf", bbox_inches='tight')

	gradient_x_collapse = np.sum(np.abs(gradient_x),axis=1)
	gradient_y_collapse = np.sum(np.abs(gradient_y),axis=0)

	fig = plt.figure()
	plt.title('Gradient Collapse')
	plt.plot(gradient_x_collapse,'ro-')
	plt.plot(gradient_y_collapse,'bo-')
	plt.show()
	fig.savefig(tag + "_gradient_collapse_debug.pdf", bbox_inches='tight')

	gradient_with_labels = np.hstack((gradient_x_collapse.reshape((len(gradient_x_collapse),1)),np.arange(len(gradient_x_collapse)).reshape((len(gradient_x_collapse),1))))

	#print("gradient with labels:" + str(gradient_with_labels))
	#print("gradient with labels shape:" + str(gradient_with_labels.shape))

	a = gradient_with_labels
	sorted_ar = a[a[:,0].argsort()[::-1]]

	print("sorted gradient\n" + str(sorted_ar[:2, :]))

	if sorted_ar[1][1] > sorted_ar[0][1]:
		compartments = [(0,int(sorted_ar[0][1])), (int(sorted_ar[0][1]),int(sorted_ar[1][1])), (int(sorted_ar[1][1]), len(sorted_ar))]
	else:
		compartments = [(0,int(sorted_ar[1][1])), (int(sorted_ar[1][1]),int(sorted_ar[0][1])), (int(sorted_ar[0][1]), len(sorted_ar))]

	print("compartments\n" + str(compartments))

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
main()
