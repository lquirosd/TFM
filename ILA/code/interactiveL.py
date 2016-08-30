import numpy as np
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
except:
    import pickle as pickle 

import imgPage_float as imgPage
import os, glob, sys, argparse #--- To handle console arguments 
import matplotlib.patches as patches
import bz2
import time
import gc

Bfix = False
Ufix = False

def minFun (x0, x1, x2, x3, uZ, bZ, p0, p1):
    sumP0 = p0[-1,-1] - imgPage.getIIsum(p0, np.array([x0, x1, x2, x3]))
    sumP1 = imgPage.getIIsum(p1, np.array([x0, x1, x2, x3]))
    valGMM = uZ[x0,x1] + bZ[x2,x3]
    return -sumP0 - sumP1 - valGMM

def decodeFeedback(event, bbox, ax, data):
    """
        Decode user feedback (ie. click over image), then update layout hypothesis
        based on that feedback. Image is updated to present new layout (red line)
    """
    init = time.clock()
    #--- get Fix status
    global Ufix
    global Bfix
    #print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
    #        (event.button, event.x, event.y, event.xdata, event.ydata))
    #--- Only right click is expected, any other is ignored
    if (event.button == 1):
        #--- Compute euclidean distance to each main paragraph corner
        Udist = np.sqrt((bbox[1]-event.xdata)**2+(bbox[0]-event.ydata)**2)
        Bdist = np.sqrt((bbox[3]-event.xdata)**2+(bbox[2]-event.ydata)**2)
        #--- Cast event to fit matrix index 
        event.xdata = int(event.xdata)
        event.ydata = int(event.ydata)
        #--- User click is decoded as the nearest (ie. Minimum distance)
        if (Udist < Bdist):
            if (Bfix == False):
                Bsum = np.ones(data.imgShape) * np.inf
                #for r in np.arange(event.ydata+1,data.imgShape[0]):
                #    for c in np.arange(event.xdata+1, data.imgShape[1]):
                #        Bsum[r,c] = minFun(event.ydata, event.xdata, r, c, data.uZ, data.bZ, data.P0II, data.P1II)
                
                #--------------------------------------------------------------------------
                #--- Use vectorization in order to reduce execution time (for loop is intractable
                #--- for interaction systems)
                #--- TPT reduction: ~25s -> ~0.065s
                #--- Bsum = -(P0[-1,-1]-(D_0+A_0-B_0-C_0))-(D_1+A_1-B_1-C_1)-Ugmm-Bgmm
                #---         |--------------v------------|  |-----v--------| |---v---|
                #---                P0 II sum                 P1 II sum        GMM model
                #--------------------------------------------------------------------------
                Bsum[event.ydata:,event.xdata:] = -(
                    data.P0II[-1,-1] - ( 
                    data.P0II[event.ydata:,event.xdata:] +
                    data.P0II[event.ydata-1,event.xdata-1] -
                    data.P0II[event.ydata-1,event.xdata:] - 
                    np.broadcast_to(data.P0II[event.ydata:,event.xdata-1], (data.imgShape[1]-event.xdata, data.imgShape[0]-event.ydata)).T) 
                    ) - (
                    data.P1II[event.ydata:,event.xdata:] + 
                    data.P1II[event.ydata-1,event.xdata-1] - 
                    data.P1II[event.ydata-1,event.xdata:] - 
                    np.broadcast_to(data.P1II[event.ydata:,event.xdata-1], (data.imgShape[1]-event.xdata, data.imgShape[0]-event.ydata)).T
                    ) - data.uZ[event.ydata,event.xdata] - data.bZ[event.ydata:,event.xdata:]
                #--- Search for minimum in Plog function
                BsC = np.unravel_index(Bsum.argmin(),data.imgShape)
                bbox[2:4] = BsC
            bbox[0:2] = [event.ydata, event.xdata]
            Ufix = True
        if (Udist > Bdist):
            if (Ufix == False):
                Usum = np.ones(data.imgShape) * np.inf
                #for r in np.arange(0,event.ydata-1):
                #    for c in np.arange(0,event.xdata-1):
                #        Usum[r,c] = minFun(r, c, event.ydata, event.xdata, data.uZ, data.bZ, data.P0II, data.P1II)
                
                #--------------------------------------------------------------------------
                #---   USE vectorization as in Bsum case 
                #--------------------------------------------------------------------------
                Usum[1:event.ydata,1:event.xdata] = -(
                    data.P0II[-1,-1] - (
                    data.P0II[event.ydata,event.xdata] +
                    data.P0II[:event.ydata-1,:event.xdata-1] -
                    data.P0II[event.ydata,:event.xdata-1] - 
                    np.broadcast_to(data.P0II[:event.ydata-1,event.xdata], (event.xdata-1, event.ydata-1)).T) 
                    ) - (
                    data.P1II[event.ydata,event.xdata] + 
                    data.P1II[:event.ydata-1,:event.xdata-1] - 
                    data.P1II[event.ydata,:event.xdata-1] - 
                    np.broadcast_to(data.P1II[:event.ydata-1,event.xdata], (event.xdata-1, event.ydata-1)).T
                    ) - data.uZ[1:event.ydata,1:event.xdata] - data.bZ[event.ydata,event.xdata]
                UsC = np.unravel_index(Usum.argmin(),data.imgShape)
                bbox[0:2] = UsC
            bbox[2:4] = [event.ydata, event.xdata]
            Bfix = True
        nPatch = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0], fc = 'none', ec = 'red')
        ax.add_patch(nPatch)
        plt.draw()
        #--- keep a small sleep time to allow system to draw correctly 
        plt.pause(0.01)
        print "Delay: {0:.5f} seconds".format(time.clock() - init)

def handle_close(event, imgFile, imgData, bbox):
    """
        Save data to original file when user closes the Figure window.
        Add last bbox as user defined bbox
    """
    imgData.ubbox = bbox
    oh = open(imgFile, 'w')
    pickle.dump(imgData, oh, 2)
    oh.close()
    print "Image Data Saved..."

def main():
    """
        Main process, reads data from user provided pointers, then shows one by one
        images to the user in order to get his/her feedback.
        Results would be saved to input pickle files
    """
    #--- processing arguments 
    parser = argparse.ArgumentParser(description='Interactive Layout Analysis')
    parser.add_argument('-imgData', action="store", help="Pointer to images Data pickle file")
    parser.add_argument('-t', '--testDir', action="store", help="Pointer to CRFs model file")
    parser.add_argument('-s', '--statistics', action="store_true", help="Print some statistics about script execution")
    parser.add_argument('--debug', action="store_true", help="Run script on Debugging mode")
    args = parser.parse_args()
    if (args.debug): print args 
    if(args.statistics): init = time.clock() 
    #--- Use global fix variables to keep tracking of automatic/user changes
    global Ufix
    global Bfix
    if (args.imgData and not os.path.isdir(args.imgData)): 
        raise ValueError("Folder: %s does not exist\n" %args.imgData)
        parser.print_help()
        sys.exit(2)
    #------------------------------------------------------------------------
    #--- Use binary files in order to reduce reading time
    #--- Improvement: bz2: ~36s per file; bin: ~0.08s per file
    #------------------------------------------------------------------------
    #imgNames = glob.glob(args.imgData + "/*_data.pickle.bz2")
    imgNames = glob.glob(args.imgData + "/*_data_bin.pickle")
    nImgs = len(imgNames)
    if nImgs <= 0:
        raise ValueError("Folder: %s contains no image data\n" %args.imgData)
        parser.print_help()
        sys.exit(2)
    #--- Main loop
    for i, imgFile in enumerate(imgNames):
        #--- Set fix variables to False, then any change is allowed
        Ufix = False
        Bfix = False
        print "Working on data from: {0:}".format(imgFile) 
        Rinit = time.clock()
        fh = open(imgFile, 'rb')
        #fh = bz2.BZ2File(imgFile, "rb")
        Pinit = time.clock()
        gc.disable()
        imgData = pickle.load(fh)
        gc.enable()
        print "Load Delay: {0:.5f} seconds".format(time.clock()-Pinit)
        fh.close()
        print "Read Delay: {0:.5f} seconds".format(time.clock()-Rinit)
        fig, ax = plt.subplots( nrows=1, ncols=1 )
        plt.axis('off')
        ax.imshow(imgData.img, cmap='gray')
        bbox = imgData.bfbbox * 1
        bPatch = patches.Rectangle((bbox[1], bbox[0]), bbox[3]-bbox[1], bbox[2]-bbox[0],
                fc = 'none', ec = 'green')
        ax.add_patch(bPatch)
        #--- Connect button_press and close events to respective functions in order
        #---    to handle user feedback decoding
        #cid = fig.canvas.mpl_connect('button_press_event', decodeFeedback)
        cid = fig.canvas.mpl_connect('button_press_event', lambda event: decodeFeedback(event, bbox, ax, imgData))
        xid = fig.canvas.mpl_connect('close_event', lambda event: handle_close(event, imgFile, imgData, bbox))
        plt.show()
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(xid)

if __name__ == '__main__':
    main()

