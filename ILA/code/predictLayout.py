from __future__ import division
import numpy as np
import scipy.ndimage as ndi
from sklearn import mixture


def twoPointStencil2D(data, h=1):
    """
    Compute two-Pooints stencil on each axis:
             f(x+h)-f(x-h)    1Dconvolve([1, 0, -1])
     f'(x) = ------------- =  ----------------------
                 2h                    2h
    Handle borders using one-sided stencil
             f(x)-f(x-h)    f'(x) = f(x+h)-f(x)
     f'(x) + -----------            -----------
                  h                     h
    """
    der = np.zeros((data.shape[0], data.shape[1],2))
    der[:,:,0] = ndi.convolve1d(data, [1, 0, -1], axis=0, mode= 'nearest')/(2*h)
    der[:,:,1] = ndi.convolve1d(data, [1, 0, -1], axis=1, mode= 'nearest')/(2*h)
    #--- Handle rows border
    der[0,:,0] = (data[1,:] - data[0,:])/h
    der[-1,:,0] = (data[-1,:] - data[-2,:])/h
    #--- handle colums border
    der[:,0,1] = (data[:,1] - data[:,0])/h
    der[:,-1,1] = (data[:,-1] - data[:,-2])/h

    return der


def derGMMmodel(GMMmodel, UB):
    """
    Compute derivates of GMM model, respect to each corner as:
             sum(W*N(x,\mu,\Sigma)*(x - \mu).T inv(\Sigma))
    f'(x) =  -----------------------------------------------
                       sum(W*N(x,\mu,\Sigma))
    """
    outUB = UB
    U = UB[0:2]
    B = UB[2:4]
    #--- Compute deriv respect to Upper corner
    denU = np.exp(GMMmodel['Upper'].score(U.reshape(1,-1)))
    numU = np.sum(
            np.exp(
                mixture.log_multivariate_normal_density(
                    GMMmodel['Upper'].means_,
                    GMMmodel['Upper'].covars_,
                    GMMmodel['Upper'].covariance_type)
                ) 
            * GMMmodel['Upper'].weights_ 
            * (GMMmodel['Upper'].mean_ - U).T
            * np.linalg.inv(GMMmodel['Upper'].covars_),
            axis=0
            )
    outUB[0:2] = numU/denU

    #--- Compute deriv respect to Bottom corner
    denB = np.exp(GMMmodel['Bottom'].score(B.reshape(1,-1)))
    numB = np.sum(
            np.exp(
                mixture.log_multivariate_normal_density(
                    GMMmodel['Bottom'].means_,
                    GMMmodel['Bottom'].covars_,
                    GMMmodel['Bottom'].covariance_type)
                ) 
            * GMMmodel['Bottom'].weights_ 
            * (GMMmodel['Bottom'].mean_ - U).T
            * np.linalg.inv(GMMmodel['Bottom'].covars_),
            axis=0
            )
    outUB[2:4] = numB/denB


    return outUB

def computeII(data):
    """
    Computes Integral Image as defined on 
    Lewis, J.P. (1995). Fast template matching. Proc. Vision Interface
    """
    return data.cumsum(axis=0).cumsum(axis=1)

def getIIsum(data, U, B):
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
    if (U == B):
        return data[U]
    else:
        return (data[B] + data[U]) - (data[U[0], B[1]] + data[B[0], U[1]])

def computeLogProb(P1II, P0II, Qmodel, UB):
    """
    Compute prob as:
    #---
                   __ K   __ |S_k|             __|~S_k|
            P(L) = \      \      log{P(s_d|h)} \     log{P(s_d|h)} + log{P(h)}
                   /__k=1 /__ d=1              /__d=1

            log{P(h)} = log{P(u)P(b)} = log{P(u)} + log{P(b)}
    Where \sum is computed using Inntegral Image 
    """
    U = UB[0:2]
    B = UB[2:4]
    #qProb = Qmodel['Upper'].score(U.reshape(1,-1)) + \
    #        Qmodel['Bottom'].score(B.reshape(1,-1))
    pProb1 = getIIsum(P1II, (U[0], U[1]), (B[0], B[1]))
    pProb0 = P0II[-1,-1] - getIIsum(P0II, (U[0], U[1]), (B[0], B[1]))

    return pProb1 + pProb0 #+ qProb

def derP1(II, UB):
    dUr = (getIIsum(II, (UB[0]+1, UB[1]), (UB[2],UB[3])) - getIIsum(II, (UB[0]-1, UB[1]), (UB[2],UB[3])))/2
    dUc = (getIIsum(II, (UB[0], UB[1]+1), (UB[2],UB[3])) - getIIsum(II, (UB[0], UB[1]-1), (UB[2],UB[3])))/2
    dBr = (getIIsum(II, (UB[0], UB[1]), (UB[2]+1,UB[3])) - getIIsum(II, (UB[0], UB[1]), (UB[2]-1,UB[3])))/2
    dBc = (getIIsum(II, (UB[0], UB[1]), (UB[2],UB[3]+1)) - getIIsum(II, (UB[0], UB[1]), (UB[2],UB[3]-1)))/2
    return np.array([dUr, dUc, dBr, dBc])

def derP0(II, UB):
    all0 = 2*II[-1,-1]
    dUr = (all0 - getIIsum(II, (UB[0]+1, UB[1]), (UB[2],UB[3])) + getIIsum(II, (UB[0]-1, UB[1]), (UB[2],UB[3])))/2
    dUc = (all0 - getIIsum(II, (UB[0], UB[1]+1), (UB[2],UB[3])) + getIIsum(II, (UB[0], UB[1]-1), (UB[2],UB[3])))/2
    dBr = (all0 - getIIsum(II, (UB[0], UB[1]), (UB[2]+1,UB[3])) + getIIsum(II, (UB[0], UB[1]), (UB[2]-1,UB[3])))/2
    dBc = (all0 - getIIsum(II, (UB[0], UB[1]), (UB[2],UB[3]+1)) + getIIsum(II, (UB[0], UB[1]), (UB[2],UB[3]-1)))/2
    return np.array([dUr, dUc, dBr, dBc])

def predictLayout(P1II, P0II, Qmodel, init=np.zeros(4), thr=0.001, T=100, alpha=0.1):
    deltaLogProb = np.Inf
    prevLogProb = 99999999999
    bestUB = init
    #--- Init Step
    thisUB = init
    bestLogProb = computeLogProb(P1II, P0II, Qmodel, thisUB)
    #--- Iterate "T" times or until converge
    for i in np.arange(T):
        #thisUB = thisUB - (alpha * \
        #        (derPmodelII[thisUB[[0,2]],
        #            thisUB[[1,3]],:].flatten() + \
        #        derQmodel(Qmodel, thisUB)))
        thisUB = thisUB - (
                    0.00001 * \
                    (
                        derP1(P1II, thisUB) + derP0(P0II, thisUB) #+ derGMMmodel(Qmodel, thisUB) 
                    )
                ).astype(int)
        print thisUB
        logProb = computeLogProb(P1II, P0II, Qmodel, thisUB)
        print "Iteration: {0:}, LogProb= {1:}".format(i, logProb)
        #deltaLogProb = np.abs(logProb - prevLogProb)
        prevLogProb = logProb
        if (logProb > bestLogProb):
            bestLogProb = logProb
            bestUB = thisUB
        if(deltaLogProb <= thr):
            #--- Alg is converged, the get out of here!!!
            print "hola"
            break

    return bestUB



def _testModule():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import cm

    try:
        import cPickle as pickle
    except:
        import pickle as pickle
    EPS = np.finfo(float).eps

    fh = open("/home/lorenzoqd/TFM/ILA/models/CRFs/_z0.3_w32_g3/GMM_22_z0.3_w32_g3_u2_b3_model.pickle",'r')
    Qmodel =  pickle.load(fh)
    fh.close()
    P = np.loadtxt('/home/lorenzoqd/TFM/ILA/models/CRFs/_z0.3_w32_g3/test_pos/bla.txt')
    P1 = P[:,1].copy()
    P0 = P[:,1].copy()
    P1[P[:,0]==0] = 1 - P1[P[:,0]==0]
    P0[P[:,0]==1] = 1 - P1[P[:,0]==1]
    P1 = np.log(P1 + EPS).reshape(365,230)
    P0 = np.log(P0 + EPS).reshape(365,230)
    #Pmodel = np.log(P1)
    #Pmodel0 = Pmodel.copy()
    #Pmodel1 = Pmodel.copy()
    #Pmodel1[P[:,0]==0] = 0
    #Pmodel1 = Pmodel1.reshape(365,230)
    #Pmodel0[P[:,0]==1] = 0
    #Pmodel0 = Pmodel0.reshape(365,230)
    T = 100
    thr = 0.1 #--- keep hight for test only
    alpha = 0.1
    #--- Test computeII -> OK
    P1II = computeII(P1)
    P0II = computeII(P0)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].axis('off')
    ax[0].imshow(P1, cmap=cm.coolwarm)
    ax[1].axis('off')
    ax[1].imshow(P0, cmap=cm.coolwarm)
    fig.savefig('testP.png', bbox_inches='tight')
    plt.close(fig)
    fig1, ax1 = plt.subplots(nrows=1, ncols=2)
    ax1[0].axis('off')
    ax1[0].imshow(P1II, cmap=cm.coolwarm)
    ax1[1].axis('off')
    ax1[1].imshow(P0II, cmap=cm.coolwarm)
    fig1.savefig('testII.png', bbox_inches='tight')
    plt.close(fig1)
    uc = 0
    br = 364
    bc = 229
    all0 = getIIsum(P0II, (0,0), (364,229))
    der = np.zeros((365,230))
    for r in np.arange(5,360,1):
        for c in np.arange(5,225,1):
            der[r,c] = ((getIIsum(P1II, (r+1, c+1),(br, bc)) - getIIsum(P1II, (r-1, c-1),(br, bc)))/2) + \
                    (((all0 - getIIsum(P0II, (r+1,c-1),(br, bc)))-(all0 - getIIsum(P0II, (r-1,c+1), (br,bc))))/2)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    ax2.axis('off')
    im = ax2.imshow(der, cmap=cm.coolwarm)
    fig2.colorbar(im)
    fig2.savefig('testIIder.png', bbox_inches='tight')
    print computeLogProb(P1II, P0II, Qmodel, np.array([100,80,200,180])) 
    OUT = predictLayout(P1II, P0II, Qmodel, init=np.array([100,80,200,180]), thr=thr, T=T, alpha=alpha)
    #OUT = predictLayout(init=np.array([100, 80, 200, 180]), 
    #        P1II=P1II, P0II=P0II,
    #        Qmodel=Qmodel,
    #        thr=thr, T=T, alpha=alpha)
    print OUT
    print "test"


if __name__ == '__main__':
    _testModule()
