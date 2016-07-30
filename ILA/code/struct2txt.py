from __future__ import division
import numpy as np
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
import imgPage
import sys, argparse #--- To handle console arguments 
 



def main():

    parser = argparse.ArgumentParser(description='Page Layout Extraction')
    parser.add_argument('-i', required=True, action="store", help="Pointer to XML's folder")
    parser.add_argument('-out', required=True, action="store", help="Pointer to XML's folder")
    args = parser.parse_args()
    
    fh = open(args.i, 'r')
    imgData = pickle.load(fh)
    for img in imgData:
        a = np.zeros((365*230,10))
        a[:,0:9] = img.Xdata.reshape(365*230,9)
        a[:,-1] = img.labels.reshape(365*230)
        np.savetxt(args.out + img.name, a, fmt='%d')


if __name__ == '__main__':
    main()
