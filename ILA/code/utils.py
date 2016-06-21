from __future__ import division
import numpy as np #--- To handle math processing
import scipy.ndimage as ndi #--- To handle image processing
from scipy import misc
import os #--- To handle OS callbacks 
import xml.etree.ElementTree as et #--- To handle XML data


class imgPage(object):
   """imgPage object compiles all data of some page"""
   def __init__(self, filePointer):
      super(imgPage, self).__init__()
      self.imgPointer = filePointer
      self.dir = os.path.dirname(filePointer)
      self.name = os.path.splitext(os.path.basename(filePointer))[0]
      self.xmlPointer = self.dir + '/page/' + self.name + '.xml'

   def readImage(self, full = False):
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
      if (full):
         self.img = ndi.imread(self.imgPointer)
      else:
         self.img = ndi.imread(self.imgPointer, mode='I')
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
      return np.array([i.split(',') for i in mainParag]).astype(np.int)

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


def test_module():
   imgP = "/Users/lquirosd/Documents/MsC_MIARFID/MIARFID/TFM/MILA/DataCorpus/test/Mss_003357_0958_pag-825[857].jpg"

   imgData = imgPage(imgP)
   imgData.readImage()
   imgData.parseXML()
   mask = imgData.getGroundTrueMask()
   misc.imsave('mask.jpg', mask)
   #---- sudo ln /dev/null /dev/raw1394

if __name__ == '__main__':
   test_module()