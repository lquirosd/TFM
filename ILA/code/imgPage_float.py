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
    def __init__(self, filePointer, statistics=True):
        super(imgPage, self).__init__()
        self.imgPointer = filePointer
        self.statistics = statistics
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

    def getFeatures(self, window=33, granularity=3, pca=True):

        self.window = window
        self.granularity = granularity
        self.pca = pca
        rows = np.arange(0,self.imgShape[0],granularity)
        rows = rows[np.where((rows>(window/2)) & (rows<=(self.imgShape[0]-(window/2))))]
        colums = np.arange(0,self.imgShape[1],granularity)
        colums = colums[np.where((colums>(window/2)) & (colums<=(self.imgShape[1]-(window/2))))]
        #index = np.ix_(rows,colums)
        self.rs = rows.size
        self.cs = colums.size
        if (pca):
            #self.Xdata = np.zeros((self.rs, self.cs,9), dtype='uint8')
            self.Xdata = np.zeros((self.rs, self.cs,9))
        else:
            #self.Xdata = np.zeros((self.rs, self.cs, window*window), dtype='uint8')
            self.Xdata = np.zeros((self.rs, self.cs, window*window))
        self.labels = np.zeros((self.rs, self.cs), dtype='uint8')
        uCorner = self.getUpperPoints()
        bCorner = self.getBottomPoints()
        if(self.statistics): init = time.clock()
        for r, row in enumerate(rows):
            for c, col in enumerate(colums):
                #--- Get gray values around it
                #--- build index array
                winIndex = np.ix_(np.arange(row-(window/2),row+(window/2)+1,1,dtype=int),np.arange(col-(window/2),col+(window/2)+1,1,dtype=int))
                if (pca):
                    fstPCA = getPCA(self.img[winIndex],3)
                    self.Xdata[r,c,:] = getPCA(fstPCA.T,3).flatten()
                else:
                    self.Xdata[r,c,:] = self.img[winIndex].flatten()
                #--- add label (1 = layout, 0 = out)
                if (uCorner[1] < row and bCorner[1] > row and uCorner[0] < col and bCorner[0] > col):
                    self.labels[r,c] = 1
        if (self.statistics): print "Features time: {0:.5f} seconds".format(time.clock() - init)
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

def computeII(data):
    """
    Computes Integral Image as defined on 
    Lewis, J.P. (1995). Fast template matching. Proc. Vision Interface
    """
    return data.cumsum(axis=0).cumsum(axis=1)

def getIIsum(data, X):
    """
    Compute summed area as:
       A=U          Bi=U[0],B[1]
        +----------+
        |          |
        |          |
        +----------+
       C=B[0],U[1]  D=B

    \sum = I(D) - I(A) + I(Bi) + I(C)
    """
    X = (X[0],X[1],X[2],X[3])
    if (X[0:2] == X[2:4]):
        return data[U]
    else:
        return data[X[2],X[3]] - data[X[0],X[1]] + data[X[0], X[3]] + data[X[2], X[1]]

def getGMMlog(GMM, X):
    return  GMM['Upper'].score(np.array([[X[0], X[1]]])) + \
            GMM['Bottom'].score(np.array([[X[2],X[3]]]))

class BBox(object):
   def __init__(self, x1, y1, x2, y2):
      '''
      (x1, y1) is the upper left corner,
      (x2, y2) is the lower right corner,
      with (0, 0) being in the upper left corner.
      '''
      if x1 > x2: x1, x2 = x2, x1
      if y1 > y2: y1, y2 = y2, y1
      self.x1 = x1
      self.y1 = y1
      self.x2 = x2
      self.y2 = y2
      self.area = (x2 - x1) * (y2 - y1)
   def taxicab_diagonal(self):
      '''
      Return the taxicab distance from (x1,y1) to (x2,y2)
      '''
      return self.x2 - self.x1 + self.y2 - self.y1
   def overlaps(self, other):
      '''
      Return True iff self and other overlap.
      '''
      return not ((self.x1 > other.x2)
                  or (self.x2 < other.x1)
                  or (self.y1 > other.y2)
                  or (self.y2 < other.y1))
   def __eq__(self, other):
      return (self.x1 == other.x1
               and self.y1 == other.y1
               and self.x2 == other.x2
               and self.y2 == other.y2)

def find_paws(data, smooth_radius = 5, threshold = 0.0001):
   """Detects and isolates contiguous regions in the input array"""
   # Blur the input data a bit so the paws have a continous footprint 
   data = ndi.uniform_filter(data, smooth_radius)
   # Threshold the blurred data (this needs to be a bit > 0 due to the blur)
   thresh = data < threshold
   # Fill any interior holes in the paws to get cleaner regions...
   filled = ndi.morphology.binary_fill_holes(thresh)
   # Label each contiguous paw
   coded_paws, num_paws = ndi.label(filled)
   # Isolate the extent of each paw
   # find_objects returns a list of 2-tuples: (slice(...), slice(...))
   # which represents a rectangular box around the object
   data_slices = ndi.find_objects(coded_paws)
   return data_slices

def slice_to_bbox(slices):
   for s in slices:
      dy, dx = s[:2]
      yield BBox(dx.start, dy.start, dx.stop+1, dy.stop+1)

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
    print "Total Seconds = {0:.5f}".format(time.clock()-init)
    print imgData.img.shape 
    imgData.delimg()
    #plt.imshow(imgData.img, cmap = 'Greys')
    #plt.imshow(mask, cmap = 'Greys')
    #plt.show()
    #---- sudo ln /dev/null /dev/raw1394

if __name__ == '__main__':
    test_module()
