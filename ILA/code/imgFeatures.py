from __future__ import division
import numpy as np #--- To handle math processing
#import scipy.integrate as sci
import scipy.ndimage as ndi
from scipy import linalg as LA
#from scipy.ndimage.interpolation import zoom
#import mahotas #--- to handle image manipulation <numpy underlining>
import matplotlib.pyplot as plt #--- To handle plotting
import matplotlib.patches as patches
import sys, argparse #--- To handle console arguments 
import glob, os #--- to handle OS callbacks 
import xml.etree.ElementTree as ET
from pprint import pprint
import time
import imgPage
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export


def main():
    """
    #---------------------------------------------------------------------------#
    #---                            main                            ---#
    #---------------------------------------------------------------------------#
    Description: 
        Main Function
    Inputs:
        See argparse section
    Outputs:
        Features matrix
    Author:
        Quiros Diaz, Lorenzo
    Date:
        18/Jul/2016
    #------------------------------------------------------------------------------#
    """
    #--- processing arguments 
    parser = argparse.ArgumentParser(description='K-NN classifier')
    parser.add_argument('-imgDir', required=True, action="store", help="Pointer to XML's folder")
    parser.add_argument('-z', '--zoom', type=float, default=0.5, action="store", help="Image size zoom [default 0.5]; 1.0= no zoom")
    parser.add_argument('-w', '--window', type=int, default=5, action="store", help="Window size")
    parser.add_argument('-g', '--granularity', type=int, default=1, action="store", help="Granularity of Filter[default 1]")
    parser.add_argument('-if', '--imgFeatures', action="store", help="Pre-Computed Image Features")
    parser.add_argument('-o', '--out', default=".", action="store", help="Folder to save Out files")
    parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
    parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
    args = parser.parse_args()
    if (args.debug): print args 
    if (args.statistics): init = time.clock()
    #--- Validate arguments 
    if (not os.path.isdir(args.imgDir)):
        print "Folder: %s does not exists\n" %args.imgDir
        parser.print_help()
        sys.exit(2)

    #--- Read images and build features maps
    allImgs = glob.glob(args.imgDir + "/*.jpg")
    nImgs = len(allImgs)
    #--- define out files name codec
    subName = str(nImgs) + "_z" + str(args.zoom) + "_w" + str(args.window) + "_g" + str(args.granularity)

    if  nImgs <= 0:
        print "Folder: %s contains no images\n" %args.imgDir
        parser.print_help()
        sys.exit(2)
    
    if (args.statistics): PIinit = time.clock()
    if (arfs.imgFeatures):
        fh = open(args.imgFeatures, 'r')
        imgData = pickle.load(fh)
        fh.close()
    else:
        imgData = np.empty(nImgs, dtype=object)
        for i, file in enumerate(allImgs):
            imgData[i] = imgPage.imgPage(file)
            print "Processing: {}".format(file)
            imgData[i].readImage(zoom = args.zoom)
            imgData[i].parseXML()
            imgData[i].getFeatures(window=args.window, granularuty=args.granularity, pca = True)
            #--- remove img data in order to reduce memory usage 
            imgData[i].delimg()
        print "saving data..."
        fh = open(args.out + "features_" + subName + ".pickle", 'w')
        pickle.dump(imgData, fh)
        fh.close()
    if (args.statistics): print 'Parsing Image Data: {0:.5f} seconds'.format(time.clock() - PIinit) 
    #--- train CRF model

if __name__ == '__main__':
   main()
