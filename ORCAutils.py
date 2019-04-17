#!/share/software/user/open/python/3.6.1/bin/python3
import sys
import numpy as np
from numpy import linalg as LA
from scipy.spatial.distance import pdist,squareform

from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

