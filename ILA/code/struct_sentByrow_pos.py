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
        a = np.apply_along_axis('-'.join, 1, img.Xdata.reshape(row*col,feat).astype('int').astype('str'))
        l = img.labels.reshape(row*col)
        print "Working on {} ...".format(img.name)
        it = np.nditer(a, flags=['f_index'])
        b = np.empty(row*col, dtype='a600')
        while not it.finished:
            #print it.index
            r = int(it.index/col)
            c = it.index - (r * col)
            if (it.index == 0):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[n]={2:s}\tw[b]={3:s}\tw[t]|w[n]={1}|{2}\tw[t]|w[b]={1}|{3}\tp[r]={4:d}\tp[c]={5:d}".format(l[it.index], a[it.index], a[it.index+1], a[it.index+col+1],r, c)
            elif (it.index == it.itersize-1):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[u]={3:s}\tw[u]|w[t]={3}|{1}\tw[p]|w[t]={2}|{1}\tp[r]={4:d}\tp[c]={5:d}".format(l[it.index], a[it.index], a[it.index-1], a[it.index-col-1], r, c)
            elif (it.index <= col):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[b]={4:s}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}\tw[t]|w[b]={1}|{4}\tp[r]={5:d}\tp[c]={6:d}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index+col+1], r, c)
            elif (it.index >= it.itersize-col-1):
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[u]={4:s}\tw[u]|w[t]={4}|{1}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}\tp[r]={5:d}\tp[c]={6:d}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index-col-1],r , c)
            else:
                b[it.index] = "{0:d}\tw[t]={1:s}\tw[p]={2:s}\tw[n]={3:s}\tw[u]={4:s}\tw[b]{5:s}\tw[u]|w[t]={4}|{1}\tw[p]|w[t]={2}|{1}\tw[t]|w[n]={1}|{3}\tw[t]|w[b]={1}|{5}\tp[r]={6:d}\tp[c]={7:d}".format(l[it.index], a[it.index], a[it.index-1], a[it.index+1], a[it.index-col-1], a[it.index+col+1], r, c)
            if (it.index != 0 and (it.index + 1) % (col) == 0):
                b[it.index] = b[it.index] + "\t__EOS__\n"
            if ((it.index) % (col) == 0 ):
                b[it.index] = b[it.index] + "\t__BOS__"

            it.iternext()
        np.savetxt(args.out + img.name, b, fmt="%s")



if __name__ == '__main__':
    main()
