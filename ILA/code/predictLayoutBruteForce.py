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
import bz2
#import bbox
import time
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
import subprocess as shell

#class SearchBounds(object):
#    def __init__(self, xmax=[1.1,1.1], xmin=[1.1,1.1]):
#        self.rmax = 365
#        self.rmin = 0
#        self.cmax = 230
#        self.cmin = 0
#    def __call__(self, **kwargs):
#        x = kwargs["x_new"]
#        d = x[2] <= x[0] or x[3] <= x[1] or \
#                self.rmax >= x[0] >= self.rmin or \
#                self.cmax >= x[1] >= self.cmin or \
#                self.rmax >= x[2] >= self.rmin or \
#                self.cmax >= x[3] >= self.cmin 
#        return d


def minFun(x, uZ, bZ, p0, p1):
    if (x[2] <= x[0] or x[3] <= x[1]):
        return np.inf
    else:
        sumP0 = p0[-1,-1] - imgPage.getIIsum(p0, x)
        sumP1 = imgPage.getIIsum(p1, x)
        #valGMM = imgPage.getGMMlog(uZ, x)
        valGMM = uZ[x[0],x[1]] + bZ[x[2],x[3]]
        return -sumP0 - sumP1 - valGMM

#def findBboxBF(GMM, P0, P1, x):
#    #--- compute II
#    p0II = imgPage.computeII(P0)
#    p1II = imgPage.computeII(P1)
#    r, c = P0.shape
#    #--- using r/2 in order to reduce grid size, but Bootm point cant be reduced
#    #--- in general case, since bbox could be pretty small 
#    #--- Use small set for testing
#    Urmin = 0 if x[0]-50 < 0 else x[0]-50
#    Ucmin = 0 if x[1]-50 < 0 else x[1]-50
#    Brmin = 0 if x[2]-50 < 0 else x[2]-50
#    Bcmin = 0 if x[3]-50 < 0 else x[3]-50
#    Urmax = r if x[0]+50 > r else x[0]+50
#    Ucmax = c if x[1]+50 < c else x[1]+50
#    Brmax = r if x[2]+50 < r else x[2]+50
#    Bcmax = c if x[3]+50 < c else x[3]+50
#    rranges = (slice(Urmin,Urmax,3), slice(Ucmin,Ucmax,3), slice(Brmin,Brmax,3), slice(Bcmin,Bcmax,3) )
#    params = (GMM, p0II, p1II)
#    resBrute = optimize.brute(minFun, rranges, args=params, full_output=False, finish=None)#, finish=optimize.fmin)
#    print resBrute
#    return resBrute
#    #--- Use basinhopping
#    #minimizer_kwargs = {"method": "BFGS"}
#    #resBH = optimize.basinhopping(minFunc, x0, niter= 1000, T=3, stepsize=3, 
#     #       minimizer_kwargs= minKw, disp=True, niter_success=50)
#

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
        #if(bla > 0): break
        #--- read img
        if(args.statistics): Decinit = time.clock() 
        print "Working on {}...".format(img.name)
        img.readImage(zoom=img.zoom)
        #--- window and granularity should be extracted from model, but isnt on test model yet
        #print img.window
        #print img.granularity
        #x = np.linspace(0, img.imgShape[1])
        #y = np.linspace(0, img.imgShape[0])
        x = np.arange(0, img.imgShape[1])
        y = np.arange(0, img.imgShape[0])
        X, Y = np.meshgrid(x, y)
        XX = np.array([X.ravel(), Y.ravel()]).T

        uZ = GMMmodel['Upper'].score_samples(XX)[0].reshape(img.imgShape)
        bZ = GMMmodel['Bottom'].score_samples(XX)[0].reshape(img.imgShape)

        #--- Results Format: Label\smarginalProb
        #--- Since we are working on 2 class problem, P(0) = 1- P(1) 
        rData = np.loadtxt(args.testDir + '/' + img.name + '.results')
        labels = rData[:,0]
        P0 = rData[:,1].copy()
        P1 = rData[:,1].copy()
        P0[rData[:,0]==1] = 1 - P0[rData[:,0]==1]
        P1[rData[:,0]==0] = 1 - P1[rData[:,0]==0]
        rows = np.arange(0,img.imgShape[0],img.granularity)
        rows = rows[np.where((rows>(img.window/2)) & (rows<=(img.imgShape[0]-(img.window/2))))]
        colums = np.arange(0,img.imgShape[1],img.granularity)
        colums = colums[np.where((colums>(img.window/2)) & (colums<(img.imgShape[1]-(img.window/2))))]
        zm = np.zeros(img.imgShape)
        z0 = np.ones(img.imgShape)
        z1 = np.ones(img.imgShape)
        #print labels.shape
        #print rows.shape
        #print colums.shape
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
        img.zm = zm
        img.z0 = z0
        img.z1 = z1
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
        plt.axis('off')
        ax.imshow(img.img, cmap='gray')
        #--- get real bbox
        Up = img.getUpperPoints()
        Bp = img.getBottomPoints()
        rPatch = patches.Rectangle((Up[0],Up[1]), Bp[0]-Up[0], Bp[1]-Up[1],
                fc = 'none', ec = 'blue')
        ax.add_patch(rPatch)
        img.gtbbox = np.array([Up[1], Up[0], Bp[1], Bp[0]])
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
        img.pgbbox = np.array([y1, x1, yW-y1, xW-x1])
        if (args.statistics): print 'PAWS Time: {0:.5f} seconds'.format(time.clock() - PAWSinit)
        #--- Find Main Paragraph using Brute Force does not work search spage is pretty big 
        #print "Working on Brute Force alg..."
        #x0 = np.array([y1, x1, yW+y1,xW+x1 ])
        if(args.statistics): BFinit = time.clock() 
        #Br = findBboxBF(GMMmodel, z0, z1,x0) 
        #bruteBbox = bruteResults[0]
        #bruteLogScore = bruteResults[1]
        #bfPatch = patches.Rectangle((Br[1],Br[0]), Br[3]-Br[1], Br[2]-Br[0],
        #        fc = 'none', ec = 'green')
        #ax.add_patch(bfPatch)
        #--- gen data 
        #print img.imgShape
        #print Up
        #print Bp
        #P1 = 0.3 * np.random.random(img.imgShape)
        #P1[Up[1]:Bp[1],Up[0]:Bp[0]] = ((0.9-0.7)*np.random.random(P1[Up[1]:Bp[1],Up[0]:Bp[0]].shape)) + 0.7
        #P0 = 1 - P1
        z0L = z0.copy()
        z1L = z1.copy()
        z0L[z0 != 0] = np.log(z0[z0!=0])
        z1L[z1 != 0] = np.log(z1[z1!=0])
        P0II = imgPage.computeII(z0L)
        P1II = imgPage.computeII(z1L)
        Usum = np.ones(img.imgShape) * np.inf
        Bsum = np.ones(img.imgShape) * np.inf
        #uZ = GMMmodel
        #bZ = 0
        for r in np.arange(0,img.imgShape[0]):
            for c in np.arange(0,img.imgShape[1]):
                Usum[r,c]= minFun(np.array([r, c,img.imgShape[0]-1, img.imgShape[1]-1]), uZ, bZ, P0II, P1II)
                Bsum[r,c]= minFun(np.array([0,0,r,c]), uZ, bZ, P0II, P1II)
        if (args.statistics): print 'Brute Force: {0:.5f} seconds'.format(time.clock() - BFinit)
        #--- Save Results
        UsC = np.unravel_index(Usum.argmin(),img.imgShape)
        BsC = np.unravel_index(Bsum.argmin(),img.imgShape)
        pp = patches.Rectangle((UsC[1], UsC[0]), BsC[1]-UsC[1], BsC[0]-UsC[0],
                                fc = 'none', ec = 'green')
        fig2, ax2 = plt.subplots( nrows=1, ncols=1 )
        ax2.imshow(img.img, cmap='gray')
        ax2.imshow(Usum + Bsum, alpha=0.4)
        #ax2[1].imshow(img.img, cmap='gray')
        #ax2[1].imshow(Bsum, alpha=0.4)
        ax.add_patch(pp)
        img.bfbbox = np.array([UsC[0], UsC[1], BsC[0], BsC[1]])
        fig.savefig(args.testDir + '/' + img.name + '.png', bbox_inches='tight', pad_inches=0, frameon=False)
        plt.close(fig)
        fig2.savefig(args.testDir + '/' + img.name + '_sums.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig2)
        fig3, ax3 = plt.subplots(nrows=1, ncols=1)
        ax3.imshow(np.hstack((z0L, z1L)))
        fig3.savefig(args.testDir + '/' + img.name + '_II.png', bbox_inches='tight', pad_inches=0)
        plt.close(fig3)
        #fig4, ax4 = plt.subplots(nrows=1, ncols=1)
        #ax4.imshow(np.hstack((np.log(P0), np.log(P1))))
        #fig4.savefig(args.testDir + '/' + img.name + 'sII.png')
        #plt.close(fig4)
        fh = bz2.BZ2File(args.testDir + '/' + img.name + '_data.pickle.bz2', 'w')
        pickle.dump(img, fh)
        fh.close()
        if (args.statistics): print 'Total Time: {0:.5f} seconds'.format(time.clock() - init)

    print "Done..."


if __name__ == '__main__':
    main()
