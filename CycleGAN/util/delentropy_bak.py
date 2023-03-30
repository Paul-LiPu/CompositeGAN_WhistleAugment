#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Trying to reproduce: https://arxiv.org/abs/1609.01117

from urllib.request import urlopen
from scipy import misc,ndimage,signal
import numpy as np
import matplotlib.pyplot as plt
import urllib.request, json

np.random.seed( 5739751 )

images = []

images += [ ( "Racoon", misc.face( gray = True ).astype( int ) ) ]
images += [ ( "Blurred Racoon", ( ndimage.gaussian_filter( misc.face( gray = True ), 100 ) ).astype( int ) ) ]
images += [ ( "Uniform random noise (8 bit)" , (   255 * np.random.rand( 256 , 256  ) ).astype( int ) ) ]
images += [ ( "Uniform random noise (16 bit)", ( 65535 * np.random.rand( 1024, 1024 ) ).astype( int ) ) ]
images += [ ( "Blurred random noise (16 bit)", ( ndimage.gaussian_filter( 65535 * np.random.rand( 1024, 1024 ), 100 ) ) ) ]
# http://www.io.csic.es/PagsPers/JPortilla/content/BLS-GSM/test_images/barbara.png
#images += [ ( "Barbara", misc.imread( "barbara.png" ).astype( int ) ) ]
# varying the smoothing and animating the 2D histogram over that should look interesting

# Physical quantum random data
if True:
    print( "Getting random data online from qrng.anu.edu.au" )
    qrngData = json.load( urllib.request.urlopen( 'https://qrng.anu.edu.au/API/jsonI.php?length=256&type=hex16&size=256' ) )
    uint8Data = np.array( [ np.frombuffer( bytearray.fromhex( line ), dtype = np.uint8 ) for line in qrngData['data'] ] ).astype( int )
    images += [ ( "qrng.anu.edu.au", uint8Data ) ]
    misc.imsave( "qrng.png", uint8Data )
else:
    images += [ ( "qrng.anu.edu.au", misc.imread( "qrng.png" ).astype( int ) ) ]

# Linear Gradients
images += [ ( "Horizontal Gradient", np.outer( np.ones  ( 256 ), np.arange( 256 ) ) ) ]
images += [ ( "Vertical Gradient"  , np.outer( np.arange( 256 ), np.ones  ( 256 ) ) ) ]

# Better physical quantum random data from: https://www.qutools.com/products/quRNG/quRNG_sample_file.bin
width,height = 256,256
images += [ ( "qutools.com", np.frombuffer( bytearray( open('quRNG_sample_file.bin', 'rb').read() ), dtype = np.uint8 )[:width*height].reshape( height, width ).astype( int ) ) ]

def createGaussian( width, height ):
    x = np.outer( np.ones  ( width ), np.arange( height ) ) - height / 2.
    y = np.outer( np.arange( width ), np.ones  ( height ) ) - width  / 2.
    return ( 256 * np.exp( -( x**2 + y**2 ) / ( 2 * ( min( width, height )/4. )**2 ) ) ).astype( float )
images += [ ( "Gaussian (256x256, float)", createGaussian( 256, 256 ) ) ]
images += [ ( "Gaussian (256x256, int)"  , createGaussian( 256, 256 ).astype( int ) ) ]
images += [ ( "Gaussian (128x128, float)", createGaussian( 128, 128 ) ) ]
images += [ ( "Gaussian (128x128, int)"  , createGaussian( 128, 128 ).astype( int ) ) ]

for i,imagePair in zip( range(len(images)), images ):
    label, image = imagePair
    fig = plt.figure( figsize = ( 9,8 ) )
    fig.suptitle( label )
    print( "=== " + label + " ===" )

    # Using a 2x2 difference kernel [[-1,+1],[-1,+1]] results in artifacts!
    # In tests the deldensity seemed to follow a diagonal because of the
    # assymetry introduced by the backward/forward difference
    # the central difference correspond to a convolution kernel of
    # [[-1,0,1],[-1,0,1],[-1,0,1]] and its transposed, produces a symmetric
    # deldensity for random noise.
    if True:
        # see paper eq. (4)
        fx = ( image[:,2:] - image[:,:-2] )[1:-1,:]
        fy = ( image[2:,:] - image[:-2,:] )[:,1:-1]
    else:
        # throw away last row, because it seems to show some artifacts which it shouldn't really
        # Cleaning this up does not seem to work
        kernelDiffY = np.array( [ [-1,-1],[1,1] ] )
        fx = signal.fftconvolve( image, kernelDiffY.T ).astype( image.dtype )[:-1,:-1]
        fy = signal.fftconvolve( image, kernelDiffY   ).astype( image.dtype )[:-1,:-1]
    print( "fx in [{},{}], fy in [{},{}]".format( fx.min(), fx.max(), fy.min(), fy.max() ) )
    diffRange = np.max( [ np.abs( fx.min() ), np.abs( fx.max() ), np.abs( fy.min() ), np.abs( fy.max() ) ] )
    if diffRange >= 200   and diffRange <= 255  : diffRange = 255
    if diffRange >= 60000 and diffRange <= 65535: diffRange = 65535

    # see paper eq. (17)
    # The bin edges must be integers, that's why the number of bins and range depends on each other
    nBins = min( 1024, 2*diffRange+1 )
    if image.dtype == np.float:
        nBins = 1024
    print( "Bins = {}, Range of Diff = {}".format( nBins, diffRange ) )
    # Centering the bins is necessary because else all value will lie on
    # the bin edges thereby leading to assymetric artifacts
    dbin = 0 if image.dtype == np.float else 0.5
    r = diffRange + dbin
    delDensity, xedges, yedges = np.histogram2d( fx.flatten(), fy.flatten(), bins = nBins, range = [ [-r,r], [-r,r] ] )
    if nBins == 2*diffRange+1:
        assert( xedges[1] - xedges[0] == 1.0 )
        assert( yedges[1] - yedges[0] == 1.0 )

    # Normalization for entropy calculation. np.sum( H ) should be ( imageWidth-1 )*( imageHeight-1 )
    # The -1 stems from the lost pixels when calculating the gradients with non-periodic boundary conditions
    #assert( np.product( np.array( image.shape ) - 1 ) == np.sum( delDensity ) )
    delDensity = delDensity / np.sum( delDensity ) # see paper eq. (17)
    delDensity = delDensity.T
    # "The entropy is a sum of terms of the form p log(p). When p=0 you instead use the limiting value (as p approaches 0 from above), which is 0."
    # The 0.5 factor is discussed in the paper chapter "4.3 Papoulis generalized sampling halves the delentropy"
    H = - 0.5 * np.sum( delDensity[ delDensity.nonzero() ] * np.log2( delDensity[ delDensity.nonzero() ] ) ) # see paper eq. (16)
    print( "H =", H )
    # gamma enhancements and inversion for better viewing pleasure
    delDensity = np.max(delDensity) - delDensity
    gamma = 1.
    delDensity = ( delDensity / np.max( delDensity ) )**gamma * np.max( delDensity )

    ax = [
        fig.add_subplot( 221, title = "Example image " + str(i) + ", H=" + str( np.round( H, 3 ) ) ),
        fig.add_subplot( 222, title = "x gradient of image (color range: [" +
                              str( np.round( -diffRange, 3 ) ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
        fig.add_subplot( 223, title = "y gradient of image (color range: [" +
                              str( np.round( -diffRange, 3 ) ) + "," + str( np.round( diffRange, 3 ) ) + "])" ),
        fig.add_subplot( 224, title = "Histogram of gradient (gamma corr. " + str(gamma) + " )" )
    ]
    ax[0].imshow( image, cmap=plt.cm.gray )
    ax[1].imshow( fx , cmap=plt.cm.gray, vmin = -diffRange, vmax = diffRange )
    ax[2].imshow( fy , cmap=plt.cm.gray, vmin = -diffRange, vmax = diffRange )
    ax[3].imshow( delDensity  , cmap=plt.cm.gray, vmin = 0, interpolation='nearest', origin='low',
            extent = [ xedges[0], xedges[-1], yedges[0], yedges[-1] ] )

    fig.tight_layout()
    fig.subplots_adjust( top = 0.92 )
    fig.savefig( label + "-delentropy.png" )

plt.show()
