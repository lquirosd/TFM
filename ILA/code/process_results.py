import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import misc
import scipy.ndimage as ndi
import imgPage_float as imgPage
import sys, argparse #--- To handle console arguments 
import matplotlib.patches as patches
import bbox
try:
    import cPickle as pickle
except:
    import pickle #--- To handle data export
import subprocess as shell


def main():
    #--- processing arguments 
    parser = argparse.ArgumentParser(description='Layout Analysis')
    parser.add_argument('-imgData', action="store", help="Pointer to images Data pickle file")
    parser.add_argument('-t', '--testDir', action="store", help="Pointer to CRFs model file")
    parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
    parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
    args = parser.parse_args()
    if (args.debug): print args 
    

    #--- Read imgData 
    fh = open(args.imgData, 'r')
    imgData = pickle.load(fh)
    fh.close()
    for img in imgData:
        #--- read img
        print "Working on {}...".format(img.name)
        img.readImage(zoom=img.zoom)
        img.window = 32
        img.granularity = 3
        labels = np.loadtxt(args.testDir + '/' + img.name + '.results')
        rows = np.arange(0,img.imgShape[0],img.granularity)
        rows = rows[np.where((rows>(img.window/2)) & (rows<=(img.imgShape[0]-(img.window/2))))]
        colums = np.arange(0,img.imgShape[1],img.granularity)
        colums = colums[np.where((colums>(img.window/2)) & (colums<=(img.imgShape[1]-(img.window/2))))]
        zm = np.zeros(img.imgShape)
        labels = labels.reshape(rows.size, colums.size)
        for r, row in enumerate(rows):
            for c, col in enumerate(colums):
                winIndex = np.ix_(np.arange(row-(img.granularity/2),row+(img.granularity/2)+1,1,dtype=int),
                        np.arange(col-(img.granularity/2),col+(img.granularity/2)+1,1,dtype=int))
                zm[winIndex] = labels[r,c]
        zm[zm==0] = 255
        zm[zm==1] = 0
        data_slices = bbox.find_paws(zm, smooth_radius = 20, threshold = 22)
        bboxes = bbox.slice_to_bbox(data_slices)
        maxArea = 0
        #for s in data_slices:
        #    dx, dy = s[:2]
        #    area = (dx.stop+1 - dx.start)*(dy.stop+1 - dy.start)
        #    if (area > maxArea):
        #        maxArea = area
        #        x1 = dx.start
        #        y1 = dy.start
        #        xW = (dx.stop+1 - dx.start)
        #        yW = (dy.stop+1 - dy.start)
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        ax.axis('off')
        #p = patches.Rectangle((x1, y1), xW, yW, fc = 'none', ec = 'red')
        ax.imshow(img.img, cmap='gray')
        ax.imshow(zm, alpha=.4, cmap='viridis')
        #ax.add_patch(p)
        for box in bboxes:
            xwidth = box.x2 - box.x1
            ywidth = box.y2 - box.y1
            area = xwidth * ywidth
            if (area > maxArea):
                xW = xwidth
                yW = ywidth
                x1 = box.x1
                y1 = box.y1
                maxArea = area
        p = patches.Rectangle((x1, y1), xW, yW,
                                fc = 'none', ec = 'red')
        ax.add_patch(p)
        #--- get real bbox
        Up = img.getUpperPoints()
        Bp = img.getBottomPoints()
        rPatch = patches.Rectangle((Up[0],Up[1]), Bp[0]-Up[0], Bp[1]-Up[1],
                fc = 'none', ec = 'blue')
        ax.add_patch(rPatch)

        fig.savefig(args.testDir + '/' + img.name + '.png', bbox_inches='tight')
        plt.close(fig)
    print "Done..."


if __name__ == '__main__':
    main()
