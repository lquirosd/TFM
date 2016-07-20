from __future__ import division
import numpy as np #--- To handle math processing
import scipy.ndimage as ndi #--- To handle image processing
from scipy import misc
import os #--- To handle OS callbacks 
import xml.etree.ElementTree as et #--- To handle XML data
import gc #--- To handle garbage collection
import time
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
from scipy import linalg as la

class imgPage(object):
    """imgPage object compiles all data of some page"""
    def __init__(self, filePointer):
        super(imgPage, self).__init__()
        self.imgPointer = filePointer
        self.dir = os.path.dirname(filePointer)
        self.name = os.path.splitext(os.path.basename(filePointer))[0]
        self.xmlPointer = self.dir + '/page/' + self.name + '.xml'
        self.zoom = 1.0

    def readImage(self, full = False, zoom = 1.0):
        """
        #---------------------------------------------------------------------------#
        #---                            readImage                            ---#
        #---------------------------------------------------------------------------#
        Description: 
            Read input image and stores it on numpy array
        Inputs:
            self
        Outputs:
            self + image array
        Author:
            Quiros Diaz, Lorenzo
        Date:
            Jun/19/2016
        #------------------------------------------------------------------------------#
        """
        self.zoom = zoom
        if (full):
            self.img = ndi.zoom(ndi.imread(self.imgPointer), self.zoom)
        else:
            self.img = np.asarray(ndi.zoom(ndi.imread(self.imgPointer, mode='I'), self.zoom), 'float32')
        self.imgShape = self.img.shape

    def parseXML(self):
        """
        #---------------------------------------------------------------------------#
        #---                            parseXML                            ---#
        #---------------------------------------------------------------------------#
        Description: 
            parse XML file related to img
        Inputs:
            self
        Outputs:
            self + XML data
        Author:
            Quiros Diaz, Lorenzo
        Date:
            Jun/19/2016
        #------------------------------------------------------------------------------#
        """
        tree = et.parse(self.xmlPointer)
        self.rootXML = tree.getroot()
        self.baseXML = self.rootXML.tag.rsplit('}',1)[0] + '}'

    def getMainParagraph(self):
        #mainParag = self.rootXML[1][3][0].attrib.get('points').split()
        mainParag = self.rootXML.findall('./' + self.baseXML + 'Page' +
                '/*[@type="paragraph"]')[0].findall('./' + self.baseXML +
                'Coords')[0].attrib.get('points').split()
        return (np.array([i.split(',') for i in mainParag]).astype(np.int) * self.zoom).astype(np.int)

    def getUpperPoints(self):
        orig = self.getMainParagraph()
        return np.array([orig[0][0], orig[0][1]])

    def getBottomPoints(self):
        orig = self.getMainParagraph()
        return np.array([orig[2][0], orig[2][1]])

    def getGroundTrueMask(self):
        to_return = np.zeros(self.imgShape, dtype='uint8')
        Points = self.getMainParagraph()
        parPos = np.ix_(np.arange(Points[0][1],Points[2][1]),
                np.arange(Points[0][0], Points[2][0]))
        to_return[parPos] = 255

        return to_return

    def getFeatures(self, window=32, granularity=3, pca=True):

        rows = np.arange(0,self.imgShape[0],granularity)
        rows = rows[np.where((rows>(window/2)) & (rows<=(self.imgShape[0]-(window/2))))]
        colums = np.arange(0,self.imgShape[1],granularity)
        colums = colums[np.where((colums>(window/2)) & (colums<=(self.imgShape[1]-(window/2))))]
        #index = np.ix_(rows,colums)
        self.rs = rows.size
        self.cs = colums.size
        if (pca):
            self.Xdata = np.zeros((self.rs, self.cs,9), dtype='uint8')
        else:
            self.Xdata = np.zeros((self.rs, self.cs, window*window), dtype='uint8')
        self.labels = np.zeros((self.rs, self.cs), dtype='uint8')
        uCorner = self.getUpperPoints()
        bCorner = self.getBottomPoints()
        init = time.clock()
        for r, row in enumerate(rows):
            for c, col in enumerate(colums):
                #--- Get gray values around it
                #--- build index array
                winIndex = np.ix_(np.arange(row-(window/2),row+(window/2),1,dtype=int),np.arange(col-(window/2),col+(window/2),1,dtype=int))
                if (pca):
                    fstPCA = getPCA(self.img[winIndex],3)
                    sndPCA = getPCA(fstPCA.T,3).flatten()
                    Vmin = np.abs(np.min(sndPCA))
                    Vmax = np.abs(np.max(sndPCA))
                    self.Xdata[r,c,:] = (((sndPCA + Vmin)/(Vmax+Vmin)) * 255).astype('uint8')
                else:
                    self.Xdata[r,c,:] = self.img[winIndex].flatten()
                #--- add label (1 = layout, 0 = out)
                if (uCorner[1] < row and bCorner[1] > row and uCorner[0] < col and uCorner[0] > col):
                    self.labels[r,c] = 1
        print "Features time: {0:.5f}".format(time.clock() - init)
    def delimg(self):
        del self.img
        gc.collect()
        gc.collect()
   
def getPCA(data, dims=2):
    """
    Reduce data by PCA
    """
    m, n = data.shape
    # mean center the data
    data -= data.mean(axis=0)
    # calculate the covariance matrix
    R = np.cov(data, rowvar=False)
    # calculate eigenvectors & eigenvalues of the covariance matrix
    # use 'eigh' rather than 'eig' since R is symmetric, 
    # the performance gain is substantial
    evals, evecs = la.eigh(R)
    # sort eigenvalue in decreasing order
    idx = np.argsort(evals)[::-1]
    evecs = evecs[:,idx]
    # sort eigenvectors according to same index
    evals = evals[idx]
    # select the first n eigenvectors (n is desired dimension
    # of rescaled data array, or dims_rescaled_data)
    evecs = evecs[:, :dims]
    # carry out the transformation on the data using eigenvectors
    # and return the re-scaled data, eigenvalues, and eigenvectors
    return np.dot(evecs.T, data.T).T





def test_module():
   

    #imgP = "/Users/lquirosd/Documents/MsC_MIARFID/MIARFID/TFM/ILA/DataCorpus/test/Mss_003357_0958_pag-825[857].jpg"
    init = time.clock()
    imgP = "/home/lorenzoqd/TFM/ILA/DataCorpus/Plantas-1/Mss_003357_0958_pag-825[857].jpg"
    imgData = imgPage(imgP)
    imgData.readImage(zoom=0.3, )
    imgData.parseXML()
    #mask = imgData.getGroundTrueMask()
    imgData.getFeatures(window=32, granularity=3, pca=True)
    #a = imgData.Xdata.shape
    #print a
    #print imgData.labels.shape
    #np.savetxt("features.txt",imgData.Xdata.reshape(a[0]*a[1],9), fmt='%d')
    #misc.imsave('mask.jpg', mask)
    #import matplotlib.pyplot as plt
    print "Seconds = {0:.5f}".format(time.clock()-init)
    print imgData.img.shape 
    imgData.delimg()
    #plt.imshow(imgData.img, cmap = 'Greys')
    #plt.imshow(mask, cmap = 'Greys')
    #plt.show()
    #---- sudo ln /dev/null /dev/raw1394

if __name__ == '__main__':
    test_module()
