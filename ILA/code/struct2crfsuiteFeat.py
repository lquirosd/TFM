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
    fh.close()
    for img in imgData:
        (row, col, feat)  = img.Xdata.shape  
        a = np.empty(row*col, dtype='str')
        a = np.apply_along_axis('-'.join, 1, img.Xdata.reshape(row*col,feat).astype('str'))
        l = img.labels.reshape(row*col)
        print "Working on {} ...".format(img.name)
        it = np.nditer(a, flags=['f_index'])
        b = np.empty(row*col, dtype='a600')
        while not it.finished:
            #print it.index
            if (it.index == 0):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[n]={2:s}\tw[b]={3:s}\tw[t]|w[n]={1}|{2}\tw[t]|w[b]={1}|{3}\t__BOS__".format(l[it.index], a[it.index], a[it.index+1], a[it.index+col+1])
            elif (it.index == it.itersize-1):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[u]={3:s}\tw[u]|w[t]={3}|{1}\tw[p]|w[t]={2}|{1}\t__EOS__".format(l[it.index], a[it.index], a[it.index-1], a[it.index-col-1])
            elif (it.index <= col):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[b]={4:s}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}\tw[t]|w[b]={1}|{4}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index+col+1])
            elif (it.index >= it.itersize-col-1):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[u]={4:s}\tw[u]|w[t]={4}|{1}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index-col-1])
            else:
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[u]={4:s}\tw[b]{5:s}\tw[u]|w[t]={4}|{1}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}\tw[t]|w[b]={1}|{5}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index-col-1], a[it.index+col+1])

            it.iternext()
        np.savetxt(args.out + img.name, b, fmt="%s")



if __name__ == '__main__':
    main()
