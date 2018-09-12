#
# TODO: Get a permanent solution for the six module.
# TODO: Figure out the way to do this on analytix!
#
import sys
#sys.path.insert(0, "/afs/cern.ch/user/v/vkhriste/.local/lib/python2.7/site-packages")
sys.path.insert(0, "/afs/cern.ch/user/p/pkothuri/.local/lib/python2.7/site-packages")
import six

import numpy as np
import time, os
import matplotlib
from pyspark.sql import Row
from scipy import misc
from skimage import draw

from pyspark.sql import Row
from pyspark.ml.linalg import Vectors, Matrices

#
# common definitions for image build up
#
feature_variables = [
    'Energy', 'Px', 'Py', 'Pz', 'Pt', 'Eta', 'Phi',
    'vtxX', 'vtxY', 'vtxZ','ChPFIso', 'GammaPFIso', 'NeuPFIso',
    'isChHad', 'isNeuHad', 'isGamma', 'isEle',  'isMu',
    #'Charge'
]

colors = {'isMu' : 'green',
        'isEle': 'green',
         'isGamma':'blue',
         'isChHad' : 'red',
         'isNeuHad': 'yellow'}

shapes = {'isMu' : 5,
          'isEle': 5,
          'isGamma':3,
          'isChHad' : 4,
          'isNeuHad': 0}

c_colors = [colors[k] for k in feature_variables[13:]]+['black']
cc_colors = [np.asarray(matplotlib.colors.to_rgb(k)) for k in c_colors]
cc_shapes = [shapes[k] for k in feature_variables[13:]]+[0]


def showImage(image):
    fig = plt.figure(frameon=False)
    plt.imshow(image.swapaxes(0,1))
    plt.axis('off')
    plt.savefig('fig.png', dpi=100, bbox_inches='tight', pad_inches=0)
#    plt.show()

def create3D(data):
    max_eta = 5
    max_phi = np.pi
    res= 100
    neta = int(max_eta*res)
    nphi = int(max_phi*res)
    eeta = 2.*max_eta / float(neta)
    ephi = 2.*max_phi / float(nphi)
    def ieta( eta ): return (eta+max_eta) / eeta
    def iphi(phi) : return (phi+max_phi) / ephi
    blend = 0.3
    image = np.ones((neta,nphi,3), dtype = np.float32)

    before_loop = time.time()
    for ip in range(data.shape[0]):
        p_data = data[ip,:]
        eta = p_data[0]
        phi = p_data[1]
        if eta==0 and phi==0:
            #print ip
            continue
        #pT = p_data[2]
        #lpT = min(max(np.log(pT)/5.,0.001), 10)*res/2.
        lpT = p_data[2]
        ptype = int(p_data[3])
        c = cc_colors[ ptype ]
        s = cc_shapes[ ptype ]
        R = lpT * res/1.5
        iee = ieta(eta)
        ip0 = iphi(phi)
        ip1 = iphi(phi+2*np.pi)
        ip2 = iphi(phi-2*np.pi)

        if s==0:
            xi0,yi0 = draw.circle(  iee, ip0,radius=R, shape=image.shape[:2])
            xi1,yi1 = draw.circle( iee, ip1, radius=R, shape=image.shape[:2])
            xi2,yi2 = draw.circle( iee, ip2, radius=R, shape=image.shape[:2])
            #if ptype == 5:
            #    print "MET",eta,phi
        else:
            nv = s
            vx = [iee + R*np.cos(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy = [ip0 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy1 = [ip1 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            vy2 = [ip2 + R*np.sin(ang) for ang in np.arange(0,2*np.pi, 2*np.pi/nv)]
            xi0,yi0 = draw.polygon( vx, vy , shape=image.shape[:2])
            xi1,yi1 = draw.polygon( vx, vy1 , shape=image.shape[:2])
            xi2,yi2 = draw.polygon( vx, vy2 , shape=image.shape[:2])

        xi = np.concatenate((xi0,xi1,xi2))
        yi = np.concatenate((yi0,yi1,yi2))
        image[xi,yi,:] = (image[xi,yi,:] *(1-blend)) + (c*blend)
    after_loop = time.time()
    print "Time to process the loop inside create3D : %3.3f" % (after_loop - before_loop)
    return image
#    return Vectors.dense(image.reshape((neta*nphi*3)))

def convert2image(row):
    """Assume that a row contains a non-empty 2D matrix of features"""
    start = time.time()
    #
    # this is for avoiding clashes on CC7 and SLC6 of the python versions
    #
    if np.__version__!="1.13.3":
        import sys
        sys.path.insert(0, "/afs/cern.ch/user/p/pkothuri/.local/lib/python2.7/site-packages")
        reload(np)

    lmat = np.asarray(row.lfeatures, dtype=np.float32)
    hmat = np.asarray(row.hfeatures, dtype=np.float32)

    # low level features
    l_reduced = np.asarray(np.zeros((lmat.shape[0], 4)))
    l_reduced[:, 0] = lmat[:, 5]
    l_reduced[:, 1] = lmat[:, 6]
    l_reduced[:, 2] = np.minimum(np.log(np.maximum(lmat[:, 4], 1.001))/5., 10)
    l_reduced[:, 3] = np.argmax(lmat[:, 13:], axis=-1)

    # high level features
    h_reduced = np.zeros( (1, 4))
    h_reduced[0,2] = np.minimum(np.maximum(np.log(hmat[1])/5.,0.001), 10) # MET
    h_reduced[0,1] = hmat[2] # MET-phi
    h_reduced[0,3] = int(5) ## met type

    # concatenate the high and low level features
    reduced = np.concatenate((l_reduced, h_reduced), axis=0)

    # geneate the image (as a 3D matrix)
    before_create3D = time.time()
    img = create3D(reduced)

    before_tolist = time.time()
    l = img
    end = time.time()

    print "Tiem to procss a Row before create3D: %3.3f" % (before_create3D - start)
    print "Time to process a Row: %3.3f" % (end - start)
    print "Time to run tolist: %3.3f" % (end - before_tolist)

    return Row(image=l, label=row.label)
