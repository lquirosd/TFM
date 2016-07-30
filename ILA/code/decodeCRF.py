
from __future__ import division
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
import imgPage
import sys, argparse #--- To handle console arguments 


def decodeCRF(data, img, r, c, g):
    #--- reshape data input
    crfData = data.reshape(r,c)
    #--- chech 

def main():

    parser = argparse.ArgumentParser(description='Page Layout Extraction')
    parser.add_argument('-i', required=True, action="store", help="Pointer to XML's folder")
    parser.add_argument('-out', required=True, action="store", help="Pointer to XML's folder")
    args = parser.parse_args()
