import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
from scipy import optimize
import scipy.ndimage as ndi
import imgPage_float as imgPage
import sys, argparse #--- To handle console arguments 
import matplotlib.patches as patches
#import bbox
import time
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
import subprocess as shell

def minFun(x, GMM, p0, p1):
    if (x[2] <= x[0] or x[3] <= x[1]):
        return np.inf
    else:
        sumP0 = imgPage.getIIsum(p0, x)
        sumP1 = imgPage.getIIsum(p1, x)
        valGMM = imgPage.getGMMlog(GMM, x)
        return -sumP0 - sumP1 - valGMM

def findBboxBF(GMM, P0, P1):
    #--- compute II
    p0II = imgPage.computeII(P0)
    p1II = imgPage.computeII(P1)
    r, c = P0.shape
    #--- using r/2 in order to reduce grid size, but Bootm point cant be reduced
    #--- in general case, since bbox could be pretty small 
    #--- Use small set for testing
    rranges = (slice(5,r-5,3), slice(139,141,1), slice(983,985,1), slice(570,572,1) )
    params = (GMM, p0II, p1II)
    resBrute = optimize.brute(minFun, rranges, args=params, full_output=False, finish=None)#, finish=optimize.fmin)
    print resBrute
    return resBrute


def main():
    #--- processing arguments 
    parser = argparse.ArgumentParser(description='Layout Analysis')
    parser.add_argument('-imgData', action="store", help="Pointer to images Data pickle file")
    parser.add_argument('-gmmData', action="store", help="Pointer to GMM Data pickle file")
    parser.add_argument('-t', '--testDir', action="store", help="Pointer to CRFs model file")
    parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
    parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
    args = parser.parse_args()
    if (args.debug): print args 
    if(args.statistics): init = time.clock() 

    #--- Read imgData 
    fh = open(args.imgData, 'r')
    imgData = pickle.load(fh)
    fh.close()
    #--- Read GMM model
    fh = open(args.gmmData, 'r')
    GMMmodel = pickle.load(fh)
    fh.close()
    #--- use only first image in order to test code 
    for bla, img in enumerate(imgData):
        if(bla > 0): break
        #--- read img
        if(args.statistics): Decinit = time.clock() 
        print "Working on {}...".format(img.name)
        img.readImage(zoom=img.zoom)
        #--- window and granularity should be extracted from model, but isnt on test model yet
        img.window = 16
        img.granularity = 3
        #--- Results Format: Label\smarginalProb
        #--- Since we are working on 2 class problem, P(0) = 1- P(1) 
        rData = np.loadtxt(args.testDir + '/' + img.name + '.results')
        labels = rData[:,0]
        P0 = rData[:,1].copy()
        P1 = rData[:,1].copy()
        P0[rData[:,0]==1] = 1 - P0[rData[:,0]==1]
        P1[rData[:,0]==1] = 1 - P1[rData[:,0]==1]
        rows = np.arange(0,img.imgShape[0],img.granularity)
        rows = rows[np.where((rows>(img.window/2)) & (rows<=(img.imgShape[0]-(img.window/2))))]
        colums = np.arange(0,img.imgShape[1],img.granularity)
        colums = colums[np.where((colums>(img.window/2)) & (colums<=(img.imgShape[1]-(img.window/2))))]
        zm = np.zeros(img.imgShape)
        z0 = np.zeros(img.imgShape)
        z1 = np.zeros(img.imgShape)
        labels = labels.reshape(rows.size, colums.size)
        P0 = P0.reshape(rows.size, colums.size)
        P1 = P1.reshape(rows.size, colums.size)
        if(args.statistics): Rinit = time.clock() 
        for r, row in enumerate(rows):
            for c, col in enumerate(colums):
                winIndex = np.ix_(np.arange(row-(img.granularity/2),row+(img.granularity/2)+1,1,dtype=int),
                        np.arange(col-(img.granularity/2),col+(img.granularity/2)+1,1,dtype=int))
                zm[winIndex] = labels[r,c]
                z0[winIndex] = P0[r,c]
                z1[ winIndex] = P1[r,c]
        zm[zm==0] = 255
        zm[zm==1] = 0
        if (args.statistics): print 'Expand: {0:.5f} seconds'.format(time.clock() - Rinit)
        if (args.statistics): print 'Decoding: {0:.5f} seconds'.format(time.clock() - Decinit)
        #--- Find Main Paragraph using PAWS
        if(args.statistics): PAWSinit = time.clock() 
        data_slices = imgPage.find_paws(zm, smooth_radius = 20, threshold = 22)
        bboxes = imgPage.slice_to_bbox(data_slices)
        maxArea = 0
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.imshow(img.img, cmap='gray')
        #--- get real bbox
        Up = img.getUpperPoints()
        Bp = img.getBottomPoints()
        rPatch = patches.Rectangle((Up[0],Up[1]), Bp[0]-Up[0], Bp[1]-Up[1],
                fc = 'none', ec = 'blue')
        ax.add_patch(rPatch)
        #--- Adding PAWS result
        ax.imshow(zm, alpha=.4, cmap='viridis')
        for box in bboxes:
            xwidth = box.x2 - box.x1
            ywidth = box.y2 - box.y1
            area = xwidth * ywidth
            if (area > maxArea):
                xW = xwidth
                yW = ywidth
                x1 = box.x1
                y1 = box.y1
                maxArea = area
        p = patches.Rectangle((x1, y1), xW, yW,
                                fc = 'none', ec = 'red')
        ax.add_patch(p)
        if (args.statistics): print 'PAWS Time: {0:.5f} seconds'.format(time.clock() - PAWSinit)
        #--- Find Main Paragraph using Brute Force 
        print "Working on Brute Force alg..."
        if(args.statistics): BFinit = time.clock() 
        Br = findBboxBF(GMMmodel, z0, z1) 
        #bruteBbox = bruteResults[0]
        #bruteLogScore = bruteResults[1]
        bfPatch = patches.Rectangle((Br[1],Br[0]), Br[3]-Br[1], Br[2]-Br[0],
                fc = 'none', ec = 'green')
        ax.add_patch(bfPatch)
        if (args.statistics): print 'Brute Force: {0:.5f} seconds'.format(time.clock() - BFinit)
        #--- Save Results
        fig.savefig(args.testDir + '/' + img.name + '_small.png')
        plt.close(fig)
        if (args.statistics): print 'Total Time: {0:.5f} seconds'.format(time.clock() - init)

    print "Done..."


if __name__ == '__main__':
    main()
