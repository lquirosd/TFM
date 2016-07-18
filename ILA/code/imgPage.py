from __future__ import division
import numpy as np #--- To handle math processing
import scipy.ndimage as ndi #--- To handle image processing
from scipy import misc
import os #--- To handle OS callbacks 
import xml.etree.ElementTree as et #--- To handle XML data
import gc


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
         self.img = ndi.zoom(ndi.imread(self.imgPointer, mode='I'), self.zoom)
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

   def getFeatures(self, window=32, granularity=3):

      rows = np.arange(0,self.imgShape[0],granularity)
      rows = rows[np.where((rows>(window/2)) & (rows<=(self.imgShape[0]-(window/2))))]
      colums = np.arange(0,self.imgShape[1],granularity)
      colums = colums[np.where((colums>(window/2)) & (colums<=(self.imgShape[1]-(window/2))))]
      index = np.ix_(rows,colums)
      rs = rows.size
      cs = colums.size
      self.Xdata = np.zeros((rs, cs, window*window), dtype=int)
      self.labels = np.zeros((rs, cs), dtype='uint8')
      print self.imgShape
      print self.labels.shape 
      uCorner = self.getUpperPoints()
      bCorner = self.getBottomPoints()
      for r, row in enumerate(rows):
         for c, col in enumerate(colums):
            #--- Get gray values around it
            #--- build index array
            winIndex = np.ix_(np.arange(row-(window/2),row+(window/2),1,dtype=int),np.arange(col-(window/2),col+(window/2),1,dtype=int))
            self.Xdata[r,c,:] = self.img[winIndex].flatten()
            #--- add label (1 = layout, 0 = out)
            if (uCorner[1] < row and bCorner[1] > row and uCorner[0] < col and uCorner[0] > col):
               self.labels[r,c] = 1

   def delimg(self):
      del self.img
      gc.collect()
      gc.collect()




def test_module():
   imgP = "/Users/lquirosd/Documents/MsC_MIARFID/MIARFID/TFM/ILA/DataCorpus/test/Mss_003357_0958_pag-825[857].jpg"

   imgData = imgPage(imgP)
   imgData.readImage(zoom=0.1)
   imgData.parseXML()
   mask = imgData.getGroundTrueMask()
   imgData.getFeatures()
   print imgData.labels.shape
   #misc.imsave('mask.jpg', mask)
   import matplotlib.pyplot as plt
   print imgData.img.shape 
   imgData.delimg()
   #plt.imshow(imgData.img, cmap = 'Greys')
   plt.imshow(mask, cmap = 'Greys')
   plt.show()
   #---- sudo ln /dev/null /dev/raw1394

if __name__ == '__main__':
   test_module()