#!/share/software/user/open/python/3.6.1/bin/python3
import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def get_corrected_cor_mat(distmap, distmaps, tag):
	gen_pos, gen_vals = get_gen_pos_gen_vals(distmaps)
	im_cor = get_cor_matrix(distmap, tag, gen_pos, gen_vals)
	return im_cor

## From Bogdan's code ##
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

	#print("gen_vals:[" + str(gen_vals) + "]")
	#print("gen_vals.shape:[" + str(gen_vals.shape) + "]")
	#print("gen_pos:[" + str(gen_pos) + "]")
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
