from __future__ import division
import sys, argparse #--- To handle console arguments 
import numpy as np #--- To handle math processing
import scipy.ndimage as ndi #--- To handle image processing
from scipy import misc
import glob, os #--- To handle OS callbacks 
import utils
from sklearn import mixture
import time
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
try:
   import cPickle as pickle
except:
   import pickle


def main():
   """
   #---------------------------------------------------------------------------#
   #---                            main                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      main module
   Inputs:
      #--- To be updated
   Outputs:
      #--- To be updated
   Author:
      Quiros Diaz, Lorenzo
   Date:
      Jun/20/2016
   #------------------------------------------------------------------------------#
   """
   #--- processing arguments 
   parser = argparse.ArgumentParser(description='K-NN classifier')
   parser.add_argument('-trDir', required=True, action="store", help="Pointer to Training images folder")
   parser.add_argument('-o', '--out', required=True, default=".", action="store", help="Folder to save Out files")
   parser.add_argument('-nU', '--nUpper', type=int, default=2, action="store", help="Number of Mixtures for Upper Model [Default=2]")
   parser.add_argument('-nB', '--nBottom', type=int, default=3, action="store", help="Number of Mixtures for Bottom Model [Default=3]")
   parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
   parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
   args = parser.parse_args()
   if (args.debug): print args 
   if (args.statistics): init = time.clock()
   #--- Validate arguments 
   if (not os.path.isdir(args.trDir)):
      print "Folder: %s does not exists\n" %args.trDir
      parser.print_help()
      sys.exit(2)
   if (not os.path.isdir(args.out)):
      print "Folder: %s does not exists\n" %args.out
      parser.print_help()
      sys.exit(2)

   #--- Read images 
   allImgs = glob.glob(args.trDir + "/*.jpg")
   nImgs = len(allImgs)
   if  nImgs <= 0:
      print "Folder: %s contains no images\n" %args.trDir
      parser.print_help()
      sys.exit(2)

   if (args.statistics): GPinit = time.clock()
   #--- keep all image data, just to check memory usage
   #--- TODO: remove unnecessary data on each iteration  
   imgData = np.empty(nImgs, dtype=object)
   #--- Array of Upper corners 
   U = np.zeros((nImgs, 2), dtype=np.int)
   #--- Array of Bottom corners
   B = np.zeros((nImgs, 2), dtype=np.int)
   #--- get U & B corners from all TR dataSet 
   for i, file in enumerate(allImgs):
      imgData[i] = utils.imgPage(file)
      #imgData[i].readImage()
      imgData[i].parseXML()
      U[i] = imgData[i].getUpperPoints()
      B[i] = imgData[i].getBottomPoints()

   if (args.statistics): print 'Getting Data Points: {0:.5f} seconds'.format(time.clock() - GPinit)
   if (args.statistics): TGinit = time.clock()
   #--- Train GMM Models
   #--- Upper GMM
   uGMM = mixture.GMM(n_components = args.nUpper)
   uGMM.fit(U)
   #--- Bottom GMM
   bGMM = mixture.GMM(n_components = args.nBottom,  covariance_type='diag')
   bGMM.fit(B)

   GMM_models = {'Upper': uGMM, 'Bottom': bGMM}
   #--- Save Models to file
   #--- Out File Name 
   outFile = args.out + 'GMM_tr' + str(nImgs) + '_u' + str(args.nUpper) + '_b' + str(args.nBottom)
   fh = open(outFile + '.model', 'w')
   pickle.dump(GMM_models, fh)
   fh.close()
   if (args.statistics): print 'Training GMM: {0:.5f} seconds'.format(time.clock() - TGinit)

   #--- Plot Mixtures and Data 
   m=9
   imgData[m].readImage(full=True)
   fig, axs = plt.subplots(1,1)
   axs.scatter(U[:, 0], U[:, 1], .8, color='red')
   axs.scatter(B[:, 0], B[:, 1], .8, color='blue')

   x = np.linspace(0, imgData[m].imgShape[1])
   y = np.linspace(0, imgData[m].imgShape[0])
   X, Y = np.meshgrid(x, y)
   XX = np.array([X.ravel(), Y.ravel()]).T

   uZ = -uGMM.score_samples(XX)[0]
   uZ = uZ.reshape(X.shape)
   bZ = -bGMM.score_samples(XX)[0]
   bZ = bZ.reshape(X.shape)

   CSu = axs.contour(X, Y, uZ, norm=LogNorm(vmin=np.min(uZ), vmax=np.max(uZ)),
                 levels=np.logspace(0, 3, 20))
   CSb = axs.contour(X, Y, bZ, norm=LogNorm(vmin=np.min(bZ), vmax=np.max(bZ)),
                 levels=np.logspace(0, 3, 20))
   #axs.clabel(CS, inline=1, fontsize=10)
   CB = plt.colorbar(CSu, ax=axs, extend='both')

   axs.imshow(imgData[m].img, cmap='gray')
   plt.axis('off')
   fig.savefig(outFile + '.png', bbox_inches='tight')
   if (args.statistics): print 'Total Time: {0:.5f} seconds'.format(time.clock() - init)
   plt.show()  

if __name__ == '__main__':
   main()

