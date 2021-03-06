from __future__ import division
import numpy as np #--- To handle math processing
import sys, argparse #--- To handle console arguments 
import glob, os #--- to handle OS callbacks 
import time
import pycrfsuite as crfsuite #--- to handle CRF models
from sklearn import mixture
import imgPage #--- to handle imga manipulations

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
    parser.add_argument('-gm', '--gmmModel', action="store", help="Pre-Computed IGMM Model")
    parser.add_argument('-nU', '--nUpper', type=int, default=2, action="store", help="Number of Mixtures for Upper Model [Default=2]")
    parser.add_argument('-nB', '--nBottom', type=int, default=3, action="store", help="Number of Mixtures for Bottom Model [Default=3]")
    parser.add_argument('-o', '--out', default=".", action="store", help="Folder to save Out files")
    parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
    parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
    args = parser.parse_args()
    if (args.debug): print args 
    if (args.statistics): init = time.clock()
    #--- Validate arguments 
    if (not os.path.isdir(args.imgDir)):
        raise ValueError("Folder: %s does not exists\n" %args.imgDir)
        parser.print_help()
        sys.exit(2)

    if (args.imgFeatures):
        if (not os.path.isfile(args.imgFeatures)):
            raise ValueError("File: %s does not exists\n" %args.imgFeatures)
            parser.print_help()
            sys.exit(2)
    else:
        if(not args.zoom and not args.window and not args.granularity):
            raise UserWarning('No feature inputs detected, using default values...')
    
    if (args.gmmModel):
        if (not os.path.isfile(args.gmmModel)):
            raise ValueError("File: %s does not exists\n" %args.gmmModel)
            parser.print_help()
            sys.exit(2)
    else:
        if(not args.nUpper and not args.nBottom):
            raise UserWarning('No GMM inputs detected, using default values...')
    
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
    if (args.imgFeatures):
        fh = open(args.imgFeatures, 'r')
        imgData = pickle.load(fh)
        fh.close()
    else:
        imgData = np.empty(nImgs, dtype=object)
        #--- Array of Upper corners 
        U = np.zeros((nImgs, 2), dtype=np.int)
        #--- Array of Bottom corners
        B = np.zeros((nImgs, 2), dtype=np.int)
        for i, file in enumerate(allImgs):
            imgData[i] = imgPage.imgPage(file)
            print "Processing: {}".format(file)
            imgData[i].readImage(zoom = args.zoom)
            imgData[i].parseXML()
            U[i] = imgData[i].getUpperPoints()
            B[i] = imgData[i].getBottomPoints()
            imgData[i].getFeatures(window=args.window, granularity=args.granularity, pca = True)
            #--- remove img data in order to reduce memory usage 
            imgData[i].delimg()
        #--- Normalize data 
        #for i in range(nImgs):
        #
        print "saving data..."
        fh = open(args.out + "features_" + subName + ".pickle", 'w')
        pickle.dump(imgData, fh)
        fh.close()
    if (args.gmmModel):
        fh = open(args.gmmModel, 'r')
        GMM_models = pickle.load(fh)
        fh.close()
    else:
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
        outFile = args.out + 'GMM_' + str(nImgs) + '_u' + str(args.nUpper) + '_b' + str(args.nBottom)
        fh = open(outFile + '.pickle', 'w')
        pickle.dump(GMM_models, fh)
        fh.close()
        
    if (args.statistics): print 'Parsing Image Data: {0:.5f} seconds'.format(time.clock() - PIinit) 
    #--- train CRF modeli
    if (args.statistics): CRFinit = time.clock()
    #--- translate features to CRF suite style
    #@[X_train y_train] = utilCRF.buildFeature(imgData)
    #--- Create trainer
    #@trainer = pycrfsuite.Trainer(verbose=False)
    #--- Load data to the trainer
    #@for xseq, yseq in zip(X_train, y_train):
    #@        trainer.append(xseq, yseq)
    #--- Set trainer params 
    #@trainer.set_params({
    #@        'c1': 1.0,   # coefficient for L1 penalty
    #@        'c2': 1e-3,  # coefficient for L2 penalty
    #@        'max_iterations': 50,  # stop earlier
    #@        # include transitions that are possible, but not observed
    #@        'feature.possible_transitions': True
    #@        })
    #--- Train the model
    #@trainer.train(args.out + 'sitesModel_' + 'c1-1.0_c2-1e-3_i50.crfsuite')
    if (args.statistics): print 'Training CRF Model: {0:.5f} seconds'.format(time.clock() - CRFinit) 
    
    
    
    if (args.statistics): print 'Total time: {0:.5f} seconds'.format(time.clock() - init) 
    

if __name__ == '__main__':
   main()
