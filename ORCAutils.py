#!/share/software/user/open/python/3.6.1/bin/python3
import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from matplotlib.colors import Normalize

class MidpointNormalize(Normalize):
        def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
                self.midpoint = midpoint
                Normalize.__init__(self, vmin, vmax, clip)

        def __call__(self, value, clip=None):
                # I'm ignoring masked values and all kinds of edge cases to make a
                # simple example...
                x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
                return np.ma.masked_array(np.interp(value, x, y))

def plot_map(mat, tag):
	f = plt.figure()
	plt.imshow(mat,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_map.pdf", bbox_inches='tight')

def plot_gradient(grad, tag):
	f = plt.figure()
	plt.imshow(grad,interpolation='nearest',cmap='seismic')
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_gradient.pdf", bbox_inches='tight')

def plot_map_with_squares(mat, coords, tag):
	import matplotlib.patches as patches

	norm = MidpointNormalize(midpoint=0)

	f = plt.figure()
	plt.imshow(mat,interpolation='nearest',cmap='seismic', norm=norm)

	for c in coords:
		sqsize = c[1] - c[0]
		sqpt = c[0]

		rect = patches.Rectangle((sqpt,sqpt),sqsize,sqsize,linewidth=2,edgecolor='g',facecolor='none')
		plt.gca().add_patch(rect)
	plt.colorbar()
	plt.show()
	f.savefig(tag + "_debug_mat_w_square.pdf", bbox_inches='tight')

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

def get_XY_matrix_gradient(im_cor, tag, plotFig=False):
	gradient_x = np.array(np.gradient(im_cor,axis=0))

	if plotFig == True:
		plot_gradient(gradient_x, tag + "_X")

	gradient_y = np.array(np.gradient(im_cor,axis=1))
	
	if plotFig == True:
		plot_gradient(gradient_y, tag + "_Y")

	return gradient_x, gradient_y

def get_sorted_matrix_gradient(im_cor, tag):
	gradient_x = np.array(np.gradient(im_cor,axis=0))

	plot_gradient(gradient_x, tag + "_X")

	gradient_y = np.array(np.gradient(im_cor,axis=1))
	
	plot_gradient(gradient_y, tag + "_Y")

	gradient_x_collapse = np.nansum(np.abs(gradient_x),axis=1)
	gradient_y_collapse = np.nansum(np.abs(gradient_y),axis=0)


	plot_gradient_collpase_xy(gradient_x_collapse, gradient_y_collapse, tag)

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

def plot_median_dist_map(distmap, tag, vmin=None, vmax=None, norm=None):

	if norm != None:
		norm = MidpointNormalize(midpoint=0)

	if vmin == None and vmax == None:
		#Plot the median distance map
		f = plt.figure()
		plt.imshow(-distmap,interpolation='nearest',cmap='seismic', norm=norm)
		plt.colorbar()
		plt.show()
		f.savefig(tag + "_med_dist.pdf", bbox_inches='tight')
	else:
		#Plot the median distance map
		f = plt.figure()
		plt.imshow(-distmap,interpolation='nearest',cmap='seismic', vmin=vmin, vmax=vmax, norm=norm)
		plt.colorbar()
		plt.show()
		f.savefig(tag + "_med_dist.pdf", bbox_inches='tight')

def get_dist_map_aggregate(dat):
        distance_mats=np.array(list(map(squareform,list(map(pdist,dat)))))
        med_mat = np.nanmedian(distance_mats,axis=0)

        return med_mat, distance_mats


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

