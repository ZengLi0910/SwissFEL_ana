#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 13:28:31 2018

@author: esposito_v
"""
import numpy as np
import pyFAI
import matplotlib.pyplot as plt
import inspect,dis
import dask.array as da


def expandImage(img):
  nrow = (512+2)*3+2*36
  ncol = 1032
  nimg = np.zeros( (nrow,ncol),dtype=img.dtype )
  for n in range(3):
    row_orig = slice( n*512, (n+1)*512 )
    row_new  = slice( n*514+n*36, (n+1)*514+n*36 )
    nimg[row_new] = expandModule( img[row_orig] )
  return nimg

def expandModule(img):
    assert img.shape == (512,1024)
    nimage = np.zeros( (514,1032),dtype=img.dtype )
    for col in range(4):

      #print(col,"orig",1+256*col,256*(col+1)-1)
      #print(col,"new",2+258*col,258*(col+1)-2)

      col_orig = slice( 1+256*col, 256*(col+1)-1  )
      col_new  = slice( 2+258*col, 258*(col+1)-2  )

      nimage[0:255,col_new] = img[0:255,col_orig]

      nimage[259:514,col_new] = img[257:512,col_orig]

    return nimage



def azInt1d(img, npt_az, poni, rot, wvl, pixel1=75e-6, pixel2=75e-6):
    """
    Perform azimuthal integration of a set of images.
    INPUT:
        npt: number of bins
        poni: point of normal incidence (center of the rings in the detector plane):
                poni[0]: distance sample-plane
                poni[1], [2]: pixel posititon of the center
        rot: rotations (see https://pyfai.readthedocs.io/en/latest/usage/tutorial/Geometry/geometry.html#Conclusion)
    RETURN: 1d integrated curve
    """
    ai = pyFAI.AzimuthalIntegrator(pixel1=pixel1, pixel2=pixel2, dist=poni[0], poni1=poni[1], poni2=poni[2],
                              rot1=rot[0], rot2=rot[1], rot3=rot[2], wavelength=wvl)
    
    if len(img.shape) == 2:
        q, intensity = ai.integrate1d(img, npt_az,  unit='2th_deg', mask=mask_gap(img))
    
    return q, intensity



def azInt2d(img, npt_az, npt_rad, poni, rot, wvl, pixel1=75e-6, pixel2=75e-6):
    """
    Perform azimuthal integration of a set of images.
    INPUT:
        npt: number of bins
        poni: point of normal incidence (center of the rings in the detector plane):
                poni[0]: distance sample-plane
                poni[1], [2]: pixel posititon of the center
        rot: rotations (see https://pyfai.readthedocs.io/en/latest/usage/tutorial/Geometry/geometry.html#Conclusion)
    RETURN: projected image
    """
    ai = pyFAI.AzimuthalIntegrator(pixel1=pixel1, pixel2=pixel2, dist=poni[0], poni1=poni[1], poni2=poni[2],
                              rot1=rot[0], rot2=rot[1], rot3=rot[2], wavelength=wvl)
    
    if len(img.shape) == 2:
        intensity, q, chi = ai.integrate2d(img, npt_az, npt_rad, unit='2th_deg')
        
    return q, chi, intensity


def azInt(img, poni, rot, wvl, plot=1, clim=(0,300), corrImg = None):
    npt_az = 360
    npt_rad = 1000

    if len(img.shape)==3:
        if isinstance(img, da.Array):
            img = da.nanmean(img, axis=0)
            img = img.compute()
        else:
            img = np.nanmean(img, axis=0)
    
    if not (corrImg is None):
        img = corrImg(img)

    q2d, chi, I2d = azInt2d(img, npt_rad, npt_az, poni, rot, wvl)
    q, I = azInt1d(img, npt_rad, poni, rot, wvl)
    
    if plot:
        plotAz(img, q, I, chi, I2d, clim=clim)
    
    if expecting() == 1:
        return dict(q=q, I=I, chi=chi, I2d=I2d)
    else:
        return q, I, chi, I2d


def azInt_single(imgs, poni, rot, wvl, plot=1, clim=None, corrImg = None):
    npt_az = 360
    npt_rad = 1000

    if len(imgs.shape)==3:
        I = []
        for img in imgs:
            if not (corrImg is None):
                img = corrImg(img)
        
#            q2d, chi, I2d = azInt2d(img, npt_rad, npt_az, poni, rot, wvl)
            q, Itemp = azInt1d(img, npt_rad, poni, rot, wvl)
            I.append(Itemp)
            
        I = np.asarray(I)
        if expecting() == 1:
            return dict(q=q, I=I)
        else:
            return q, I

    elif len(imgs.shape)==2:
        if not (corrImg is None):
            img = corrImg(img)
    
        q2d, chi, I2d = azInt2d(img, npt_rad, npt_az, poni, rot, wvl)
        q, I = azInt1d(img, npt_rad, poni, rot, wvl)
        
        if plot:
            plotAz(img, q, I, chi, I2d, clim=clim)
        
        if expecting() == 1:
            return dict(q=q, I=I, chi=chi, I2d=I2d)
        else:
            return q, I, chi, I2d





def expecting():
    """Return how many values the caller is expecting"""
    f = inspect.currentframe()
    f = f.f_back.f_back
    c = f.f_code
    i = f.f_lasti
    bytecode = c.co_code
    instruction = bytecode[i+3]
    if instruction == dis.opmap['UNPACK_SEQUENCE']:
        howmany = bytecode[i+4]
        return howmany
    elif instruction == dis.opmap['POP_TOP']:
        return 0
    return 1


def plotAz(img, q, I, chi, I2d, clim=(0,10)):
    plt.figure(figsize=(14,10))
    plt.suptitle('AZIMUTHAL INTEGRATION: OVERVIEW')
    ax1 = plt.subplot2grid((2,2),(0,0),rowspan=2)
    ax1.imshow(img, origin="lower", aspect="auto",interpolation='None', clim=clim)
    
    ax2 = plt.subplot2grid((2,2),(0,1))
    ax2.imshow(I2d, origin="lower", extent=[q.min(), q.max(), chi.min(), chi.max()], 
                                          aspect="auto",interpolation='None', clim=clim)
    ax2.set_xlabel('2-theta')
    ax2.set_ylabel('chi (deg)')
    
    ax3 = plt.subplot2grid((2,2),(1,1))
    ax3.plot(q,I)
    ax3.set_xlabel('2-theta')
    ax3.set_ylabel('Intensity')
#    plt.tight_layout()


def mask_gap(img):
  return img == 0



def maximize_int(img, params, wvl=1e-10):
    d = 0.17
    npt_az = 360
    poni = [d, params[0], params[1]]
    rot = [params[2], params[3], params[4]]

    q, I = azInt1d(img, npt_az, poni, rot, wvl, pixel1=75e-6, pixel2=75e-6)
    q_peak = 38.4
    idx = np.argmin(np.abs(q-q_peak))
    I = I[idx]
    I = 1/I
    return I
    

























