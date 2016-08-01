#!~/anaconda2/bin/python -u
import numpy as np #--- To handle math processing

class IntegralImage:
   """Class to store Integral Image an their properties"""
   def __init__(self, imData, label):
      #self.imData = imData
      self.label = label
      self.weight = 0
      self.computeII(imData)

   def computeII(self, imData):
      """
      #---------------------------------------------------------------------------#
      #---                            computeII                            ---#
      #---------------------------------------------------------------------------#
      Description: 
         Computes Integral Image
      Inputs:
         file: pointer to image file  
      Outputs:
         outData: II array 
      Author:
         Quiros Diaz, Lorenzo
      Date:
         Apr/20/2016
      #------------------------------------------------------------------------------#
      """
      self.InIm = np.zeros(imData.shape)
      self.InIm = imData.cumsum(axis=0)
      for dim in range(1,imData.ndim):
         self.InIm = self.InIm.cumsum(axis=dim)


   def getSum(self, TL, BR):
      """
      #---------------------------------------------------------------------------#
      #---                            getSum                            ---#
      #---------------------------------------------------------------------------#
      Description: 
         Computes the sum of II in the rectangle specified
      Inputs:
         self: II object
      Outputs:
         aSum: [float]
      Author:
         Quiros Diaz, Lorenzo
      Date:
         Apr/22/2016
      #------------------------------------------------------------------------------#
      """
      #--- transform X,Y space to R,C space
      TL = (TL[1], TL[0])
      BR = (BR[1], BR[0])
      #--- if TL==UR return II value on that point
      if (TL == BR):
         return self.InIm[TL]
      #--- else A = (II[BRr,BRc]+II[TLr,TLc]) - (II[TLr,BRc]+II[BRr,TLc])
      return (self.InIm[BR] + self.InIm[TL]) - (self.InIm[TL[0], BR[1]] + self.InIm[BR[0], TL[1]])

   def setLabel(self, label):
      """
      #---------------------------------------------------------------------------#
      #---                            setLabel                            ---#
      #---------------------------------------------------------------------------#
      Description: 
         Set if self is a face Label=1 or non-Face Label=0
      Inputs:
         label: [int] label to assign to the image
      Outputs:
         NONE
      Author:
         Quiros Diaz, Lorenzo
      Date:
         Apr/22/2016
      #------------------------------------------------------------------------------#
      """
      self.label = label

   def setWeight(self,weight):
      """
      #---------------------------------------------------------------------------#
      #---                            setWeight                            ---#
      #---------------------------------------------------------------------------#
      Description: 
         Set Weight for current image 
      Inputs:
         weight: un-normalized weight
      Outputs:
         NONE
      Author:
         Quiros Diaz, Lorenzo
      Date:
         Apr/22/2016
      #------------------------------------------------------------------------------#
      """
      self.weight = weight


   def getII(self):
      """
      #---------------------------------------------------------------------------#
      #---                            getII                            ---#
      #---------------------------------------------------------------------------#
      Description: 
         return II array
      Inputs:
         NONE
      Outputs:
         II: [array] n-dimensional
      Author:
         Quiros Diaz, Lorenzo
      Date:
         Apr/22/2016
      #------------------------------------------------------------------------------#
      """
      return self.InIm 

def computeII(imData):
   """
   #---------------------------------------------------------------------------#
   #---                            computeII                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      Computes Integral Image
   Inputs:
      imData: image array 
   Outputs:
      outData: II array 
   Author:
      Quiros Diaz, Lorenzo
   Date:
      Apr/20/2016
   #------------------------------------------------------------------------------#
   """
   #--- loop over each image
   outData = np.zeros(imData.shape)
   for i in range(imData.shape[0]):
      outData[i,:,:] = imData[i,:,:].cumsum(axis=0)
      outData[i,:,:] = outData[i,:,:].cumsum(axis=1)
   return outData
   #---- TODO: compute II as cumsum in 0 axis and 1 axis, then pass the function to each array in matrix

def getSum(II, TL, BR):
   """
   #---------------------------------------------------------------------------#
   #---                            getSum                            ---#
   #---------------------------------------------------------------------------#
   Description: 
      Computes the sum of II in the rectangle specified
   Inputs:
      self: II object
   Outputs:
      aSum: [float]
   Author:
      Quiros Diaz, Lorenzo
   Date:
      Apr/22/2016
   #------------------------------------------------------------------------------#
   """
   #--- transform X,Y space to R,C space
   TL = (TL[1], TL[0])
   BR = (BR[1], BR[0])
   #--- if TL==UR return II value on that point
   if (TL == BR):
      return II[TL]
   #--- else A = (II[BRr,BRc]+II[TLr,TLc]) - (II[TLr,BRc]+II[BRr,TLc])
   return (II[BR] + II[TL]) - (II[TL[0], BR[1]] + II[BR[0], TL[1]])

