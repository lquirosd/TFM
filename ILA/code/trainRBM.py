#!/usr/bin/python
from __future__ import division
import numpy as np #--- To handle math processing
#import scipy.integrate as sci
import scipy.ndimage as ndi
from skimage import exposure
from scipy import linalg as LA
#from scipy.ndimage.interpolation import zoom
#import mahotas #--- to handle image manipulation <numpy underlining>
import matplotlib.pyplot as plt #--- To handle plotting
#import matplotlib.patches as patches
import sys, argparse #--- To handle console arguments 
import os #--- to handle OS callbacks 
import xml.etree.ElementTree as ET
import time
import pickle

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.cross_validation import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline


def plotResults(imgData, window, granularity, labels, realLabels):
   """
   #---------------------------------------------------------------------------#
   #---                            plotResults                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      plot results over original image 
   Inputs:
      InputsDescription
   Outputs:
      OutputsDescription
   Author:
      Quiros Diaz, Lorenzo
   Date:
      Date
   #------------------------------------------------------------------------------#
   """
   iters = imgData.shape
   rows = np.arange(0,iters[0],granularity)
   rows = rows[np.where((rows>(window/2)) & (rows<=(iters[0]-(window/2))))]
   colums = np.arange(0,iters[1],granularity)
   colums = colums[np.where((colums>(window/2)) & (colums<=(iters[1]-(window/2))))]
   index = np.ix_(rows,colums)
   newImg = np.zeros(iters)
   #newImg[index] = np.multiply(imgData[index], labels.reshape((rows.shape[0],colums.shape[0])))
   for r, row in enumerate(rows):
      for c, col in enumerate(colums):
         #--- Get gray values around it
         #--- build index array
         winIndex = np.ix_(np.arange(row-(window/2),row+(window/2),1,dtype=int),np.arange(col-(window/2),col+(window/2),1,dtype=int))
         #newImg[winIndex] = np.ones((window,window)) * labels[r*colums.shape[0]+c]
         newImg[winIndex] = (labels[r*colums.shape[0]+c] + realLabels[r*colums.shape[0]+c] )/2
         #newImg[winIndex,0] = labels[r*colums.shape[0]+c]
         #newImg[winIndex,1] = realLabels[r*colums.shape[0]+c]
         #newImg[winIndex,2] = 0.5
   plt.figure(2)
   plt.subplot(1,2,1)
   plt.imshow(imgData, cmap=plt.cm.gray_r)
   plt.subplot(1,2,2)
   plt.imshow(newImg, cmap=plt.cm.gray_r)
   plt.show()


def getFeatures(imgData, window, granularity, outDir):
   """
   #---------------------------------------------------------------------------#
   #---                            getFeatures                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      Description
   Inputs:
      InputsDescription
   Outputs:
      OutputsDescription
   Author:
      Quiros Diaz, Lorenzo
   Date:
      Apr/28/2016
   #------------------------------------------------------------------------------#
   """
   imgKeys = imgData.keys()
   iters = imgData[imgKeys[0]]['sImg'].shape
   rows = np.arange(0,iters[0],granularity)
   rows = rows[np.where((rows>(window/2)) & (rows<=(iters[0]-(window/2))))]
   colums = np.arange(0,iters[1],granularity)
   colums = colums[np.where((colums>(window/2)) & (colums<=(iters[1]-(window/2))))]
   index = np.ix_(rows,colums)
   rs = rows.size
   cs = colums.size
   print rs
   print cs
   fullFeatures = rs*cs*len(imgKeys)
   allImages = np.zeros((fullFeatures, window*window), dtype='uint8')
   allLabels = np.zeros(fullFeatures, dtype='uint8')
   print 'Total Features Set: {0:}'.format(fullFeatures)
   for i, imgKey in enumerate(imgKeys):
      for r, row in enumerate(rows):
         #r = rows[i]
         for c, col in enumerate(colums):
            #--- Get gray values around it
            #--- build index array
            winIndex = np.ix_(np.arange(row-(window/2),row+(window/2),1,dtype=int),np.arange(col-(window/2),col+(window/2),1,dtype=int))
            allImages[i*(rs*cs)+(r*cs)+c,:] = ndi.distance_transform_edt(imgData[imgKey]['sImg'][winIndex]).flatten()
            #--- add label (1 = layout, 0 = out)
            if (imgData[imgKey]['sPLay'][1] < row and imgData[imgKey]['sPLay'][3] > row and imgData[imgKey]['sPLay'][0] < col and imgData[imgKey]['sPLay'][2] > col):
               allLabels[i*(rs*cs)+(r*cs)+c] = 1
   return allImages, allLabels


def parseXML(inFile):
   """
   #---------------------------------------------------------------------------#
   #---                            parseXML                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      function to parse XML file and extract layout coordinates
   Inputs:
      inFile: pointer to XML file
   Outputs:
      layout coordinates
   Author:
      Quiros Diaz, Lorenzo
   Date:
      11/Apr/2016
   #------------------------------------------------------------------------------#
   """
   tree = ET.parse(inFile)
   root = tree.getroot()
   mainParag = root[1][0][0].attrib.get('points').split()
   orig = [i.split(',') for i in mainParag]
   return np.array([int(orig[0][0]), int(orig[0][1]), int(orig[2][0]), int(orig[2][1])])

def readImgData(inDir, reduction):
   """
   #---------------------------------------------------------------------------#
   #---                            readImgData                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      function to search images and XML files on inData folder, then data is extracted:
         Image Luminance
         small Image Luminance 
         Main paragraph Layout coordinates
   Inputs:
      inDir: pointer to folder where IMG files and XML are.
   Outputs:
      toReturn: [dictionary] a dictionary with all data:
         toReturn[i]->img[]
         toReturn[i]->sImg[]
         toReturn[i]->pLay[]
   Author:
      Quiros Diaz, Lorenzo
   Date:
      12/Apr/2016
   #------------------------------------------------------------------------------#
   """
   toReturn = {}
   for file in os.listdir(inDir):
      if file.endswith('.jpg'):
         filename = os.path.splitext(os.path.basename(file))[0]
         img = ndi.imread(inDir + file, mode='I')
         #--- use spline interpolation defined on scipy zoom function:
         #--- http://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.zoom.html#scipy.ndimage.zoom
         #--- https://en.wikipedia.org/wiki/Discrete_spline_interpolation
         small_img = ndi.zoom(img, reduction)
         #small_img = exposure.equalize_hist(temp_img)
         #gauss_img = ndi.gaussian_filter(temp_img, 5)
         #small_img = exposure.equalize_adapthist(gauss_img, clip_limit=0.1)
         pData = parseXML(inDir + '/page/' + filename + '.xml')
         sPData = np.ndarray.astype(pData * reduction,int)
         #toReturn[filename] = {'img': img, 'pLay':pData, 'sImg': small_img, 'sPLay': sPData}
         toReturn[filename] = {'sImg': small_img, 'sPLay': sPData}
   return toReturn

def main():
   """
   #---------------------------------------------------------------------------#
   #---                            main                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      Main Function, arguments are processed here, and program structure is defined.
   Inputs:
      argc, argv
   Outputs:
      NONE
   Author:
      Quiros Diaz, Lorenzo
   Date:
      22/03/2016
   #------------------------------------------------------------------------------#
   """
   #--- processing arguments 
   parser = argparse.ArgumentParser(description='K-NN classifier')
   parser.add_argument('-trDir', required=True, action="store", help="Pointer to Training images folder")
   parser.add_argument('-teDir', required=True, action="store", help="Pointer to Test images folder")
   parser.add_argument('-m', '--model', required=False, action="store", help="Previous trained model")
   parser.add_argument('-r', '--reduction', type=float, default=0.5, action="store", help="Image size reduction [default 0.5]")
   parser.add_argument('-w', '--window', type=int, default=5, action="store", help="Window size")
   parser.add_argument('-g', '--granularity', type=int, default=1, action="store", help="Granularity of Filter[default 1]")
   parser.add_argument('-lr', '--learning_rate', type=float, default=0.01, action="store", help="RBM learning Rate [default 0.01]")
   parser.add_argument('-n', '--n_iter', type=int, default=20, action="store", help="RBM Number of iterations [default 20]")
   parser.add_argument('-c', '--components', type=int, default=100, action="store", help="RBM Number of components [default 100]")
   parser.add_argument('-o', '--out', default=".", action="store", help="Folder to save Out files")
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
   if (not os.path.isdir(args.teDir)):
      print "Folder: %s does not exists\n" %args.teDir
      parser.print_help()
      sys.exit(2)

   #--- read training and test images
   if (args.statistics): READinit = time.clock()
   TRdata = readImgData(args.trDir, args.reduction)
   TEdata = readImgData(args.teDir, args.reduction)
   if (args.statistics): print 'READ TPT: {0:.5f} seconds'.format(time.clock() - READinit)

   if (args.statistics): CHinit = time.clock()
   X_train, Y_train  = getFeatures(TRdata, args.window, args.granularity, args.out)
   X_train = np.asarray(X_train, 'float32')
   X_test, Y_test  = getFeatures(TEdata, args.window, args.granularity, args.out)
   X_test = np.asarray(X_test, 'float32')
   #--- Normalize data to [0-1]
   X_min = np.min([np.min(X_test,0),np.min(X_train,0)],0)
   X_max = np.max([np.max(X_test,0),np.max(X_train,0)],0)
   X_train = (X_train - X_min) / (X_max + 0.0001)
   X_test = (X_test - X_min) / (X_max + 0.0001)
   if (args.statistics): print 'BUILD FEATURES TPT: {0:.5f} seconds'.format(time.clock() - CHinit)
   #--- Define regressors 
   if (not args.model):
      logistic = linear_model.LogisticRegression()
      rbm = BernoulliRBM(random_state=0, verbose=True)
      classifier = Pipeline(steps=[('rbm', rbm), ('logistic', logistic)])
      #--- Set parameters 
      rbm.learning_rate = args.learning_rate
      rbm.n_iter = args.n_iter
      rbm.n_components = args.components
      logistic.C = 60.0
      #--- Train model
      classifier.fit(X_train, Y_train)
      #--- Save classifier
      out = open('LA_RBM_' + str(args.learning_rate) +
             '_' + str(args.n_iter) + '_' + str(args.components) +
             '_60_t.classifier', 'w')
      pickle.dump(classifier, out)
      out.close()
   else:
      f = open(args.model, 'r')
      classifier = pickle.load(f)
   Y_labels = classifier.predict(X_test)
   #Y_prob = classifier.predict_log_proba(X_test)
   #print Y_prob
   #to_plot = Y_prob[:,-9000].reshape(4*75,4*120)
   print("Logistic regression using RBM features:\n%s\n" % (metrics.classification_report(Y_test,Y_labels,target_names=['OUT-Layout','IN-Layout'])))
   
   if (args.statistics): print 'TPT: {0:.5f} seconds'.format(time.clock() - init)
   
   plotResults(TEdata['Mss_003357_0944_pag-811[843]']['sImg'], args.window, args.granularity, Y_labels, Y_test)
   #--- plot RBM components
   if not args.model:
      plt.figure(figsize=(8.4, 8))
      for i, comp in enumerate(rbm.components_):
         plt.subplot(10, 10, i + 1)
         plt.imshow(comp.reshape((32, 32)), cmap=plt.cm.gray_r,
               interpolation='nearest')
         plt.xticks(())
         plt.yticks(())
      plt.suptitle('100 components extracted by RBM', fontsize=16)
      plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
   plt.show()

   

   #raw_input('Press Enter to close...')
   #sys.exit()


if __name__ == '__main__':
   main()