import csv
import math

from tkinter import filedialog

import pandas as pd
import numpy as np
from pandas import read_csv

import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from scipy import special

from mpmath import nsum
import mpmath as mp
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.signal import argrelextrema

#import seaborn as sns
#sns.set()

import time

from matplotlib.collections import PatchCollection
from matplotlib import cm 

import copy
import sys




def range_prod(lo,hi):
    if lo+1 < hi:
        mid = (hi+lo)//2
        return range_prod(lo,mid) * range_prod(mid+1,hi)
    if lo == hi:
        return lo
    return lo*hi

def treefactorial(n):
    if n < 2:
        return 1
    return range_prod(1,n)



#0.1
def DiffractionControl(fname,IntervalLength,TransmitterHeight,ReceiverHeight,Frequency,kfactor = 4/3,roundEarth = 0,EarthDiffraction = 0,KnifeEdgeMethod=[0],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [1], PlotFunc = 0):
    
    wavel = WaveLength(Frequency)
    data, colnames = GetTerrain(fname)
    validation = InputValidate(data)

    if validation == 0:
        print('Input data is incomplete.')
        print('Ensure that each terrain profile consists of a height column and distance column.')
        print('Ensure that each distance value is accompanied by a height value.')
        return 0

    GetRadiusses = 0
    if (RoundedObstacleMethod.count(0)) > len(RoundedObstacleMethod) or (1 in TwoObstacleMethod) or (1 in SingleObstacleMethod):
        GetRadiusses = 1
    re_ = kfactor * 6371000
    Diffraction_dict = {}
    #Loop start
    for colnum in range(0,len(colnames),2):

        Diffraction_dict[colnames[colnum]] = []
        maxdist = max(data[colnames[colnum]])
        iterations = math.ceil(maxdist/(IntervalLength/1000))
        for i in range(iterations):

            distarr, heightarr, distance = TerrainDivide(data,colnames[colnum],colnames[colnum+1],IntervalLength,i+1)

            Diffraction_dict[colnames[colnum]].append(distance)

            xintersect, yintersect, Tdist, Theight, Rdist, Rheight = FresnelZoneClearance(distarr,heightarr,ReceiverHeight,TransmitterHeight,wavel,plotZone = PlotFunc,Searth = roundEarth,re = re_)
            knifeX, knifeY, radiusses = KnifeEdges(xintersect, yintersect, distarr, heightarr, Rheight, Theight, 4, GetRadiusses, PlotFunc)

#-----------------------------------------------------------------------------------------------------------------------

            if 0 in KnifeEdgeMethod:
                heightarrN = copy.deepcopy(heightarr)
                heightarrN[0] = heightarrN[0] + TransmitterHeight
                heightarrN[-1] = heightarrN[-1] + ReceiverHeight
                L, Lba, Lsph = DeltaBullington(distarr,heightarrN,wavel,re = re_)
                L = OutputValidate(L)
                key = 'Delta Bullington (dB) ' + str(int(colnum/2))
                key1 = 'Lba (dB) ' + str(int(colnum/2))
                key2 = 'Lsph (dB) ' + str(int(colnum/2))
                if key not in Diffraction_dict.keys():
                    Diffraction_dict[key] = [0]*iterations
                    Diffraction_dict[key1] = [0]*iterations
                    Diffraction_dict[key2] = [0]*iterations

                Diffraction_dict[key][i] = L
                Diffraction_dict[key1][i] = Lba
                Diffraction_dict[key2][i] = Lsph


            if len(knifeX) >= 4:

                if 1 in KnifeEdgeMethod:
                    key = 'Bullington (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Bullington (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = Bullington(knifeX,knifeY,wavel,PlotFunc,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['Bullington (dB) ' + str(int(colnum/2))][i] = L

                if 2 in KnifeEdgeMethod:
                    key = 'Epstein Peterson (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Epstein Peterson (dB) ' + str(int(colnum/2))] = [0]*iterations
                    L = EpsteinPeterson(knifeX,knifeY,wavel,PlotFunc,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)

                    Diffraction_dict['Epstein Peterson (dB) ' + str(int(colnum/2))][i] = L

                if 3 in KnifeEdgeMethod:
                    key = 'Deygout (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Deygout (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = Deygout(knifeX,knifeY,wavel,PlotFunc,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['Deygout (dB) ' + str(int(colnum/2))][i] = L

                if 4 in KnifeEdgeMethod:
                    key = 'Giovaneli (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Giovaneli (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = Giovaneli(knifeX,knifeY,wavel,PlotFunc,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['Giovaneli (dB) ' + str(int(colnum/2))][i] = L


#-----------------------------------------------------------------------------------------------------------------------
            if len(knifeX) >= 4:
                if 1 in RoundedObstacleMethod:
                    key = 'Deygout Rounded (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Deygout Rounded (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = DeygoutRounded(knifeX,knifeY,wavel,radiusses,PlotFunc,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['Deygout Rounded (dB) ' + str(int(colnum/2))][i] = L

                if 2 in RoundedObstacleMethod:
                    key = 'ITU Multiple Cylinders (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['ITU Multiple Cylinders (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = ITUMultipleCylinders(xintersect, yintersect,wavel,TransmitterHeight,ReceiverHeight,re = re_, pltIllustration = PlotFunc)
                    L = OutputValidate(L)
                    Diffraction_dict['ITU Multiple Cylinders (dB) ' + str(int(colnum/2))][i] = L

#-----------------------------------------------------------------------------------------------------------------------         
            if len(knifeX) == 4:
                if 1 in TwoObstacleMethod:
                    key = 'ITU Two Edge (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['ITU Two Edge (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = ITUTwoEdge(knifeX,knifeY,wavel)
                    L = OutputValidate(L)
                    Diffraction_dict['ITU Two Edge (dB) ' + str(int(colnum/2))][i] = L


                if 2 in TwoObstacleMethod:
                    key = 'ITU Two Rounded (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['ITU Two Rounded (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = ITUTwoRounded(knifeX,knifeY,radiusses,wavel)
                    L = OutputValidate(L)
                    Diffraction_dict['ITU Two Rounded (dB) ' + str(int(colnum/2))][i] = L



#-----------------------------------------------------------------------------------------------------------------------
            if len(knifeX) == 3:
                if 1 in SingleObstacleMethod:
                    key = 'Fresnel-Kirchoff (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['Fresnel-Kirchoff (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = FresnelKirchoff(knifeX,knifeY,wavel,Searth = roundEarth,re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['Fresnel-Kirchoff (dB) ' + str(int(colnum/2))][i] = L


                if 2 in SingleObstacleMethod:
                    key = 'ITU Single Rounded (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['ITU Single Rounded (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = ITUSingleRounded(knifeX,knifeY,wavel,radiusses[0])
                    L = OutputValidate(L)
                    Diffraction_dict['ITU Single Rounded (dB) ' + str(int(colnum/2))][i] = L

#-----------------------------------------------------------------------------------------------------------------------
            if len(knifeX) == 2:
                if EarthDiffraction == 1:
                    key = 'ITU Sperical Earth Diffraction (dB) ' + str(int(colnum/2))
                    if key not in Diffraction_dict.keys():
                        Diffraction_dict['ITU Sperical Earth Diffraction (dB) ' + str(int(colnum/2))] = [0]*iterations

                    L = ITUSpericalEarthDiffraction(distance,wavel,knifeY[0],knifeY[-1],re = re_)
                    L = OutputValidate(L)
                    Diffraction_dict['ITU Sperical Earth Diffraction (dB) ' + str(int(colnum/2))][i] = L


    
    df = pd.DataFrame.from_dict(Diffraction_dict,orient='index').transpose()
    print(df)
    of = fname[0:-4] +'dif_loss.csv'
    df.to_csv(of,index=False)

#0.2
def InputValidate(TerrainData):
    colnames = TerrainData.columns
    if (len(colnames) % 2) != 0:
        return 0
    for colnum in range(0,len(colnames),2):
        if len(TerrainData[colnames[colnum]]) != len(TerrainData[colnames[colnum+1]]):
            return 0

    return 1

#0.3
def OutputValidate(Loss):
    if Loss < 0:
        return 0
    else:
        return Loss

#0.4
def GetTerrain(fname): 

    df = pd.read_csv(fname)

    return df, df.columns

#0.5
def TerrainDivide(data, colnamex, colnamey, intlength, iterationNum): 
    pathlength = intlength/1000*iterationNum

    pathdata = data[[colnamex,colnamey]]
    pathdata = pathdata.dropna().astype(float)

    pathdata = pathdata.loc[pathdata[colnamex] <= pathlength]

    distarr = np.asarray(pathdata[colnamex])
    heightarr = np.asarray(pathdata[colnamey])

    if pathlength > max(data[colnamex]):
        pathlength = max(data[colnamex])

    distarr = distarr*1000

    return distarr, heightarr, pathlength



#1.1
def FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel, plotZone = 0,Searth = 0,re = 8500000):#

    Tdist = distarr[0] 
    Theight = theight + heightarr[0]
    Rdist = distarr[len(distarr)-1]
    Rheight = rheight + heightarr[len(heightarr)-1]
    m = (Rheight-Theight)/(Rdist-Tdist)-((Rdist-Tdist)/(2*re))*Searth
    b = Theight
    length = math.sqrt(Rdist**2+(Rheight-Theight)**2)
    rangle = np.arctan((Rheight-Theight)/Rdist)

    RadiusXValues1 = []
    RadiusXValues2 = []
    RadiusYValues1 = []
    RadiusYValues2 = []

    for x in range(math.ceil(Rdist)+1):
        y = m*x + b
        d1 = math.sqrt((y-Theight)**2+(x**2))
        r = ((wavel*(d1)*((length-d1)))/(length))**(1/2)
        dx = math.sin(rangle)*r
        dy = math.cos(rangle)*r
        x1 = x - dx
        x2 = x + dx    
        y1 = y + dy
        y2 = y - dy
        RadiusXValues1.append(x1.real)
        RadiusXValues2.append(x2.real)
        RadiusYValues1.append(y1.real)
        RadiusYValues2.append(y2.real)


    xintersect = []
    yintersect = []
    #start_time = time.time()
    for xcoord, ycoord in zip(distarr,heightarr):
        index = np.where(RadiusXValues2 >= xcoord)
        pIndex = index[0][0]


        if ycoord >= RadiusYValues2[pIndex]:
            xintersect.append(xcoord)
            yintersect.append(ycoord)
    

    #end_time = time.time()
    #print('1 Time: ',end_time-start_time)

    #start_time = time.time()
    #xcoord = [(xcoord, ycoord) for xcoord, ycoord in zip(distarr,heightarr) if ycoord >= RadiusYValues2]
    #end_time = time.time()
    #print('2 Time: ',end_time-start_time)


    if plotZone == 1:
        #plt.gca().set_aspect('equal', adjustable='box')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.plot(RadiusXValues2,RadiusYValues2,'k-')
        plt.plot(RadiusXValues1,RadiusYValues1,'-')
        plt.plot(distarr,heightarr,'-')
        plt.plot(xintersect,yintersect,'m-')
        plt.plot((Tdist,Rdist),(Theight,Rheight),'r-')
        plt.show()

    return xintersect, yintersect, Tdist, Theight, Rdist, Rheight

#1.2
def WaveLength(frequency):
    c = 300000000 #speed of light m/s
    wavel = c/frequency #wavelength or lambda
    return wavel

#1.3
def ObstacleValues(Xcoords,Ycoords,Searth = 0,re = 8500000):
    distance1 = ((Xcoords[1]-Xcoords[0])**2+(Ycoords[1]-Ycoords[0])**2)**(1/2)
    distance2 = ((Xcoords[2]-Xcoords[1])**2+(Ycoords[2]-Ycoords[1])**2)**(1/2)
    mLoS = (Ycoords[2]-Ycoords[0])/(Xcoords[2]-Xcoords[0])-((Xcoords[2]-Xcoords[0])/(2*re))*Searth
    bLoS = Ycoords[2] - Xcoords[2]*mLoS
    height = Ycoords[1]-(mLoS*Xcoords[1]+bLoS)

    return distance1,distance2,height

#1.4
def KnifeEdges(xintersect, yintersect, distarr, heightarr, Rheight, Theight, sensitivity, cylinders = 0, plotoutput = 0):

    if sensitivity == 0:
        ysmoothed = heightarr
    else:
        ysmoothed = gaussian_filter1d(heightarr,sigma=sensitivity)
    ysmoothed[0] = Theight
    ysmoothed[-1] = Rheight

    localmin = (np.diff(np.sign(np.diff(ysmoothed))) > 0).nonzero()[0] + 1         #local minimums
    localmax = (np.diff(np.sign(np.diff(ysmoothed))) < 0).nonzero()[0] + 1         # local max
    #print(localmin)
    #localmaxi = argrelextrema(ysmoothed, np.greater)
    #localmini = argrelextrema(ysmoothed, np.less)
    #print(localmini)
    peakIndeces = []
    knifeX = []
    knifeY = []
    sknifeX = []

    #Get knife edges--------------------------------------------------------------------------------------------------------
    for i in range(len(localmin)-1):
        
        hindex = np.where(heightarr[localmin[i]:localmin[i+1]] == heightarr[localmin[i]:localmin[i+1]].max())[0][0]
        peakIndeces.append(hindex+localmin[i])

    for i in range(len(peakIndeces)):
        X = distarr[peakIndeces[i]]
        
        if X in xintersect:
            knifeY.append(heightarr[peakIndeces[i]])
            knifeX.append(X)
            sknifeX.append(distarr[localmax[i]])

    #-----------------------------------------------------------------------------------------------------------------------

    #Get cylinder radiusses-------------------------------------------------------------------------------------------------
    radiusses = []
    if cylinders == 1:
        radiusses = []
        county = 0
        countr = 0
        p1 = 25
        p3 = 75

        for i in range(len(localmin)-1):
            cylindexes = np.where(np.logical_and(xintersect>distarr[localmin[i]], xintersect<distarr[localmin[i+1]]))
            g = cylindexes[0]
            if len(g)>1:
                middley = (max(yintersect[g[0]:g[-1]]) - min(yintersect[g[0]:g[-1]]))/2
                q1 = np.percentile(xintersect[g[0]:g[-1]],p1)
                q3 = np.percentile(xintersect[g[0]:g[-1]],p3)
                middlex = (q3-q1)/2
                if((middlex == 0)|(middley==0)):
                    radiusses.append(0)
                else:
                    radiusses.append(middlex)
            elif len(g) == 1:
                radiusses.append(0)
    
        if plotoutput == 1:
            
            fig, ax = plt.subplots() 
            circles = []
            for x1, y1, r in zip(knifeX,knifeY,radiusses):
                circle = plt.Circle((x1,y1), r)
                circles.append(circle)

            p = PatchCollection(circles, cmap = cm.prism, alpha = 0.4)
            ax.add_collection(p)


    #-----------------------------------------------------------------------------------------------------------------------

    if plotoutput == 1:
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.plot(knifeX,knifeY,'x')
        plt.plot(distarr,heightarr,'-')
        plt.plot(xintersect,yintersect,'m-')
        plt.show()

    knifeX.append(distarr[-1])
    knifeX.insert(0,distarr[0])
    knifeY.append(Rheight)
    knifeY.insert(0,Theight)
    return knifeX, knifeY, radiusses



#2.1
def FresnelKirchoff(Xcoords,Ycoords,wavel,distMeth = 0, meth = 0,Searth = 0,re = 8500000):
    SE = Searth
    AE = re
    distance1,distance2,height = ObstacleValues(Xcoords,Ycoords,Searth = SE,re = AE)
    if distMeth == 0:
        v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
    else:
        v = height*math.sqrt(2/wavel*(1/(Xcoords[1]-Xcoords[0])+1/(Xcoords[2]-Xcoords[1])))

    Cv = 0
    Sv = 0
    if meth == 1:
    #METHOD 1
        s = sp.Symbol('s')
        def C(s):
            return math.cos((math.pi*s**2)/2)
        def S(s):
            return math.sin((math.pi*s**2)/2)

        Cv = quad(C,0,v)
        Sv = quad(S,0,v)

    if meth == 0:
    #METHOD 2
        Vals = special.fresnel(v)
        Cv = Vals[1]
        Sv = Vals[0]
    Jv = -20*math.log10(math.sqrt((1-Cv-Sv)**2+(Cv-Sv)**2)/2)  #J(v) is the diffraction loss in dB
    return Jv

#2.2
def ITUSingleRounded(Xcoords,Ycoords,wavel,radius):
    if radius == 0:
        radius = 0.01
    distance1,distance2,height = ObstacleValues(Xcoords,Ycoords)
    Jv = FresnelKirchoff(Xcoords,Ycoords,wavel)
    m = radius*((distance1+distance2)/(distance1*distance2))/(math.pi*radius/wavel)**(1/3)
    n = height*(math.pi*radius/wavel)**(2/3)/radius
    mn = m*n
    if mn<=4:
        Tmn = 7.2*m**(1/2)-(2-12.5*n)*m+3.6*m**(3/2)-0.8*m**2
        return (Tmn + Jv)
    if mn > 4:
        Tmn = -6-20*math.log10(mn) + 7.2*m**(1/2)-(2-17*n)*m+3.6*m**(3/2)-0.8*m**2
        return (Tmn + Jv)

#2.3
def ITUNLoS(d,wavel,h1,h2,re = 8500000):

    b = 1
    f = 300000000/wavel/1000000

    X = b*d*(math.pi/(wavel*(re**2)))**(1/3)
    #X = 2.188*b*f**(1/3)*re**(-2/3)*d

    Y1 = 2*b*h1*(math.pi**2/((wavel**2)*re))**(1/3)
    Y2 = 2*b*h2*(math.pi**2/((wavel**2)*re))**(1/3)

    #Y1 = 9.575*10**(-3)*b*f**(2/3)*re**(-1/3)*h1
    #Y2 = 9.575*10**(-3)*b*f**(2/3)*re**(-1/3)*h2


    FX = 0
    if X > 1.6:
        FX = 11+10*math.log10(X)-17.6*X
    else:
        FX = -20*math.log10(X)-5.6488*X**(1.425)

    GY1 = 0
    GY2 = 0

    if Y1 == 0:
        Y1 = 0.01
    if Y2 == 0:
        Y2 = 0.01

    B1 = b*Y1
    B2 = b*Y2

    

    if B1 > 2:
        GY1 = 17.6*(B1-1.1)**(1/2)-5*math.log10(B1-1.1)-8
    else:
        GY1 = 20*math.log10(B1+0.1*B1**3)

    if B2 > 2:
        GY2 = 17.6*(B2-1.1)**(1/2)-5*math.log10(B2-1.1)-8
    else:
        GY2 = 20*math.log10(B2+0.1*B2**3)

    L = -(FX + GY1 + GY2)

    #print('X: ',X)
    #X3.append(X)
    #print('Y1: ',Y1)
    #Y1_4.append(Y1)
    #print('Y2: ',Y2)
    #Y2_5.append(Y2)
    #print('FX: ',FX)
    #FX6.append(FX)
    ##print('GY1: ',GY1)
    #GY1_7.append(GY1)
    #print('GY2: ',GY2)
    #GY2_8.append(GY2)

   

    return L

#2.4
def ITUSpericalEarthDiffraction(dm,wavel,h1,h2,re = 8500000):
    L = 0
    b = 1

    d_los = (2*re)**(1/2)*(h1**(1/2)+h2**(1/2))


    if dm >= d_los/1000:
        L = ITUNLoS(dm,wavel,h1,h2,re)

    if dm < d_los:
        c = (h1-h2)/(h1+h2)
        m = dm**2/(4*re*(h1+h2))
        b_ = 2*math.sqrt((m+1)/(3*m))*math.cos(math.pi/3+1/3*math.acos(3*c/2*math.sqrt(3*m/(m+1)**3)))
        d1 = dm/2*(1+b_)
        d2 = dm-d1
        if d2 < 0:
            d2 = 0
        h = ((h1-(d1**2)/(2*re))*d2+(h2-(d2**2)/(2*re))*d1)/dm
        hreq = 0.552*(d1*d2*wavel/dm)**0.5

        if h > hreq:
            L = 0
        else:
            aem = 0.5*(dm/(h1**(1/2)+h2**(1/2)))**2
            Ah = ITUNLoS(dm,wavel,h1,h2,aem)
            if Ah < 0:
                L = 0
            else:
                A = (1-h/hreq)*Ah
                L = A

    return L

#2.5
def ITUTwoEdge(Xcoords,Ycoords,wavel): #how do you determine that an edge is predominant?

    Tx = Xcoords[0]
    Ty = Ycoords[0]
    Rx = Xcoords[3]
    Ry = Ycoords[3]

    Ob1x = Xcoords[1]
    Ob1y = Ycoords[1]
    Ob2x = Xcoords[2]
    Ob2y = Ycoords[2]

    m1 = (Ry-Ty)/(Rx-Tx)
    b1 = Ty - m1*Tx
    

    h1 = Ob1y-(m1*Ob1x + b1)
    h2 = Ob2y-(m1*Ob2x + b1)

    r1 = ((wavel*(Ob1x-Tx)*((Rx-Ob1x-Tx)))/(Rx-Tx))**(1/2)
    r2 = ((wavel*(Ob2x-Tx)*((Rx-Ob2x-Tx)))/(Rx-Tx))**(1/2)
    ratio1 = h1/r1
    ratio2 = h2/r2

    a = Ob1x-Tx
    b = Ob2x-Ob1x
    c = Rx-Ob2x

    
    if abs(ratio1-ratio2) < 0.5: #This condition must be refined
        L1 = FresnelKirchoff([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel)
        L2 = FresnelKirchoff([Ob1x,Ob2x,Ry],[Ob1y,Ob2y,Ry],wavel)
        Lc = 0
        
        if (L1 > 15)&(L2>15):
            Lc = 10*math.log10(((a+b)*(b+c))/(b*(a+b+c)))

        #print(L1)
        #print(L2)
        #print(Lc)

        return (L1 + L2 + Lc)
    
    elif ratio1 > ratio2:

        L1 = FresnelKirchoff([Tx,Ob1x,Rx],[Ty,Ob1y,Ry],wavel)
        L2 = FresnelKirchoff([Ob1x,Ob2x,Rx],[Ob1y,Ob2y,Ry],wavel)

        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.atan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        #print(L1)
        #print(L2)
        #print(Tc)

        return (L1+L2-Tc)


    elif ratio2 > ratio1:

        L1 = FresnelKirchoff([Tx,Ob2x,Rx],[Ty,Ob2y,Ry],wavel)
        L2 = FresnelKirchoff([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel)
        
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.atan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(p/q)**(2*q)

        #print(L1)
        #print(L2)
        #print(Tc)

        return (L1+L2-Tc)

#2.6
def ITUTwoRounded(Xcoords,Ycoords,radii,wavel): 

    Tx = Xcoords[0]
    Ty = Ycoords[0]
    Rx = Xcoords[3]
    Ry = Ycoords[3]

    Ob1x = Xcoords[1]
    Ob1y = Ycoords[1]
    Ob2x = Xcoords[2]
    Ob2y = Ycoords[2]

    m1 = (Ry-Ty)/(Rx-Tx)
    b1 = Ty - m1*Tx
    
    h1 = Ob1y-(m1*Ob1x + b1)
    h2 = Ob2y-(m1*Ob2x + b1)

    r1 = ((wavel*(Ob1x-Tx)*((Rx-Ob1x-Tx)))/(Rx-Tx))**(1/2)
    r2 = ((wavel*(Ob2x-Tx)*((Rx-Ob2x-Tx)))/(Rx-Tx))**(1/2)
    ratio1 = h1/r1
    ratio2 = h2/r2

    a = Ob1x-Tx
    b = Ob2x-Ob1x
    c = Rx-Ob2x

    if abs(ratio1-ratio2) < 0.5: #This condition must be refined

        L1 = ITUSingleRounded([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel,radii[0])
        L2 = ITUSingleRounded([Ob1x,Ob2x,Ry],[Ob1y,Ob2y,Ry],wavel,radii[1])
        Lc = 0
        if (L1 > 15)&(L2>15):
            Lc = 10*math.log10(((a+b)*(b+c))/(b*(a+b+c)))

        return (L1 + L2 + Lc)
    
    elif ratio1 > ratio2:

        L1 = ITUSingleRounded([Tx,Ob1x,Rx],[Ty,Ob1y,Ry],wavel,radii[0])
        L2 = ITUSingleRounded([Ob1x,Ob2x,Rx],[Ob1y,Ob2y,Ry],wavel,radii[1])
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.atan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        return (L1+L2-Tc)


    elif ratio2 > ratio1:

        L1 = ITUSingleRounded([Tx,Ob2x,Rx],[Ty,Ob2y,Ry],wavel,radii[0])
        L2 = ITUSingleRounded([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel,radii[1])
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.atan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(p/q)**(2*q)

        return (L1+L2-Tc)

#2.7
def Bullington(Xcoords,Ycoords,wavel,pltIllustration = 0,Searth = 0,re = 8500000): ####
    if len(Xcoords) < 3:
        return 0
    Tx = Xcoords[0]
    Ty = Ycoords[0]
    Rx = Xcoords[len(Xcoords)-1]
    Ry = Ycoords[len(Ycoords)-1]

    maxy = max(Ycoords[1:(len(Ycoords)-1)])

    mTR = (Ry-Ty)/(Rx-Tx)-((Rx-Tx)/(2*re))*Searth
    bTR = Ry - mTR*Rx
    ldy = 0


    for xcoord, ycoord in zip(Xcoords[1:(len(Xcoords)-1)],Ycoords[1:(len(Ycoords)-1)]):
        LoSy = mTR*xcoord + bTR
        if LoSy < ycoord:
            ldy = 1
    m2 = 0
    m1 = 0
    if ldy > 0:
        m1 = 0
        m2 = 0
    else:
        m1 = -100
        m2 = 100

    if(Ty > maxy)&(Ry < maxy):
        m1 = -100
        m2 = 0

    b1 = 0
    b2 = 0

    for xcoord, ycoord in zip(Xcoords[1:(len(Xcoords)-1)],Ycoords[1:(len(Ycoords)-1)]):     #!?
        mtemp1 = (ycoord-Ty)/(xcoord-Tx)-((xcoord-Tx)/(2*re))*Searth
        mtemp2 = (Ry-ycoord)/(Rx-xcoord)-((Rx-xcoord)/(2*re))*Searth

        if mtemp1 > m1:
            m1 = mtemp1
            b1 = Ty - m1*Tx

        if mtemp2 < m2:
            m2 = mtemp2
            b2 = Ry - m2*Rx

    detM = 0
    Xpoint = 0
    Ypoint = 0
    for xcoord, ycoord in zip(Xcoords[1:(len(Xcoords)-1)],Ycoords[1:(len(Ycoords)-1)]):
        y1 = xcoord*m1 + b1
        y2 = xcoord*m2 + b2
        if (y1 == y2):
            detM = 1
            Xpoint = xcoord
            Ypoint = ycoord


    if detM == 0:
        Xpoint = (b2-b1)/(m1-m2)
        Ypoint = m1*Xpoint+b1

    if pltIllustration == 1:
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.plot(Xcoords,Ycoords,'x')
        plt.plot([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],'-')
        plt.show()
    SE = Searth
    AE = re
    return FresnelKirchoff([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],wavel,Searth = SE,re = AE)

#2.8
def EpsteinPeterson(Xcoords,Ycoords,wavel,pltIllustration = 0,Searth = 0,re = 8500000):
    NumEdges = len(Xcoords) - 2
    L = 0

    for i in range(NumEdges):
        Lt = FresnelKirchoff(Xcoords[i:i+3],Ycoords[i:i+3] ,wavel ,Searth=Searth ,re=re)
        if pltIllustration == 1:
            plt.plot(Xcoords[i:i+3],Ycoords[i:i+3],'-')
        if Lt < 0:
            Lt = 0
        L = L + Lt
    if pltIllustration == 1:
        plt.plot(Xcoords,Ycoords,'x')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
    return L

#2.9
def Deygout(Xcoords,Ycoords,wavel,pltIllustration = 0,Searth = 0,re = 8500000):
        
    def DeygoutLoss(Xcoords,Ycoords,wavel): #Rekursie is stadig, improve
        NumEdges = len(Xcoords) - 2
        FresnelParams = []
        for i in range(NumEdges):
            distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[-1]],[Ycoords[0],Ycoords[i+1],Ycoords[-1]],Searth=Searth,re=re)
            v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
            FresnelParams.append(v)
        if len(Xcoords) < 3:
            return 0
        else:
            MaxV = np.where(FresnelParams == np.amax(FresnelParams))
 
            L = FresnelKirchoff([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[Ycoords[0],Ycoords[MaxV[0][0].astype(int)+1],Ycoords[-1]],wavel,Searth=Searth,re=re)
            #print(L)
            if L < 0:
                L = 0

            if pltIllustration == 1:
                plt.plot([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[Ycoords[0],Ycoords[MaxV[0][0].astype(int)+1],Ycoords[-1]])

            L = L + DeygoutLoss(Xcoords[0:(MaxV[0][0].astype(int)+2)],Ycoords[0:(MaxV[0][0].astype(int)+2)],wavel) #Python is weird en die twede parameter moet een meer wees as die index wat jy soek

            L = L + DeygoutLoss(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],Ycoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],wavel)

            if L < 0:
                L = 0
            return L

    L = DeygoutLoss(Xcoords,Ycoords,wavel)
    if pltIllustration == 1 :
        plt.plot(Xcoords,Ycoords,'*')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.show()
    return L

#2.10
def Giovaneli(Xcoords,Ycoords,wavel, pltIllustration = 0,Searth = 0,re = 8500000):
    def GiovaneliLoss(Xcoords,Ycoords,wavel, pltIllustration = 0):
        if len(Xcoords) < 3:
            return 0
        NumEdges = len(Xcoords) - 2
        FresnelParams = []

        for i in range(NumEdges):
            distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[-1]],[Ycoords[0],Ycoords[i+1],Ycoords[-1]],Searth=Searth,re=re)
            v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
            FresnelParams.append(v)
        
        MaxV = np.where(FresnelParams == np.amax(FresnelParams))

        yT = 0
        yR = 0
        FresnelParams1 = []
        FresnelParams2 = []
        if len(Xcoords[0:(MaxV[0][0].astype(int)+2)])>2:
            for i in range(len(Xcoords[0:(MaxV[0][0].astype(int)+2)])-2):
                distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[MaxV[0][0].astype(int)+1]],[Ycoords[0],Ycoords[i+1],Ycoords[MaxV[0][0].astype(int)+1]],Searth=Searth,re=re)
                v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
                FresnelParams1.append(v)
            MaxV1 = np.where(FresnelParams1 == np.amax(FresnelParams1))

            tT1x = Xcoords[0]
            tO1x = Xcoords[MaxV1[0][0].astype(int)+1]
            tO1y = Ycoords[MaxV1[0][0].astype(int)+1]

            tR1x = Xcoords[MaxV[0][0].astype(int)+1]
            tR1y = Ycoords[MaxV[0][0].astype(int)+1]

            m1 = (tR1y-tO1y)/(tR1x-tO1x)-((tR1x-tO1x)/(2*re))*Searth
            b1 = tR1y - tR1x*m1
            yT = m1*tT1x+b1

        else:
            yT = Ycoords[0]

        tempX = Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)]
        tempY = Ycoords[(MaxV[0][0].astype(int)+1):len(Ycoords)]

        if len(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)])>2:
            for i in range(len(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)])-2):
                #distance1, distance2, height = ObstacleValues([Xcoords[MaxV[0][0].astype(int)+1],Xcoords[i+MaxV[0][0].astype(int)+2],Xcoords[-1]],[Ycoords[MaxV[0][0].astype(int)+1],Ycoords[i+MaxV[0][0].astype(int)+2],Ycoords[-1]])
                distance1, distance2, height = ObstacleValues([tempX[0],tempX[i+1],tempX[-1]],[tempY[0],tempY[i+1],tempY[-1]],Searth=Searth,re=re)
                v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
                FresnelParams2.append(v)
            MaxV2 = np.where(FresnelParams2 == np.amax(FresnelParams2))


            tT2x = tempX[0]
            tT2y = tempY[0]
            tO2x = Xcoords[MaxV2[0][0].astype(int)+2+MaxV[0][0].astype(int)]
            tO2y = Ycoords[MaxV2[0][0].astype(int)+2+MaxV[0][0].astype(int)]
            tR2x = tempX[-1]
            m2 = (tO2y-tT2y)/(tO2x-tT2x)-((tO2x-tT2x)/(2*re))*Searth
            b2 = tO2y - tO2x*m2
            yR = m2*tR2x+b2
        else:
            yR = Ycoords[-1]
    
        if yT < Ycoords[0]:
            yT = Ycoords[0]

        if yR < Ycoords[-1]:
            yR = Ycoords[-1]

        if pltIllustration == 1:
            plt.plot([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[yT,Ycoords[MaxV[0][0].astype(int)+1],yR])
        L = FresnelKirchoff([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[yT,Ycoords[MaxV[0][0].astype(int)+1],yR],wavel,Searth=Searth,re=re)

        L = L + GiovaneliLoss(Xcoords[0:(MaxV[0][0].astype(int)+2)],Ycoords[0:(MaxV[0][0].astype(int)+2)],wavel,pltIllustration)

        L = L + GiovaneliLoss(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],Ycoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],wavel,pltIllustration)
        return L
    L = GiovaneliLoss(Xcoords,Ycoords,wavel, pltIllustration)
    if pltIllustration == 1:
        plt.plot(Xcoords,Ycoords,'*')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.show()
    return L

#2.11.1
def DeltaBullingtonA(Xcoords,Ycoords,wavel,re=8500):

    Ce = 1/re
    Stim = -100
    Xcoords = np.asarray(Xcoords)
    hts = Ycoords[0]
    hrs = Ycoords[-1]
    #hts = theight
    #hrs = rheight
    #print('hrs:',hrs)
    #print('hts:',hts)

    d = Xcoords[-1]-Xcoords[0]
    Str = (hrs-hts)/(d)
    Luc = 0

    for i in range(len(Xcoords)-2):
        di = Xcoords[i+1]-Xcoords[0]
        hi = Ycoords[i+1]
        stim = (hi+500*Ce*di*(d-di)-hts)/di
        if stim > Stim:
            Stim = stim
    Vmax = -100
    if Stim < Str: #Path is LoS
        for i in range(len(Xcoords)-2):
            di = Xcoords[i+1]-Xcoords[0]
            hi = Ycoords[i+1]
            vmax = (hi+500*Ce*di*(d-di)-(hts*(d-di)+hrs*di)/d)*math.sqrt(((0.002*d)/(wavel*di*(d-di))))
            if vmax > Vmax:
                Vmax = vmax
        #print('v max 1:',Vmax)
        if Vmax > -0.78:
            Vals = special.fresnel(Vmax)
            Cv = Vals[1]
            Sv = Vals[0]
            Luc = -20*math.log10(math.sqrt((1-Cv-Sv)**2+(Cv-Sv)**2)/2)
            #Luc = 6.9 + 20*math.log(math.sqrt((Vmax-0.1)**2+1)+Vmax-0.1)
        else:
            Jv = 0
            Luc = Jv

    if Stim >= Str: #Path is trans-horizon
        Srim = -100
        for i in range(len(Xcoords)-2):
            di = Xcoords[i+1]-Xcoords[0]
            hi = Ycoords[i+1]
            srim = (hi + 500*Ce*di*(d-di)-hrs)/(d-di)
            if srim > Srim:
                Srim = srim

        db = (hrs-hts+Srim*d)/(Stim+Srim)                                                           ##
        vb = (hts+Stim*db-((hts*(d-db)+hrs*db)/d))*math.sqrt((0.002*d)/(wavel*db*(d-db)))           ##

        Vals = special.fresnel(vb)
        Cv = Vals[1]
        Sv = Vals[0]
        Luc = -20*math.log10(math.sqrt((1-Cv-Sv)**2+(Cv-Sv)**2)/2)
        #Luc = 6.9 + 20*math.log(math.sqrt((vb-0.1)**2+1)+vb-0.1)

    Lb = Luc + (1-math.exp(-Luc/6))*(10+0.02*d)
    return Lb

#2.11.0
def DeltaBullington(Xcoords,Ycoords,wavel,re=8500000):
    re = re/1000
    Xcoords = Xcoords/1000
    Lba = DeltaBullingtonA(Xcoords,Ycoords,wavel,re=re)
    #print('Lba: ',Lba)
    #Lba9.append(Lba)
    d = Xcoords[-1]-Xcoords[0]
    hts = Ycoords[0]
    hrs = Ycoords[-1]
    v1 = 0
    v2 = 0
    for i in range(len(Xcoords)-1):
        di = Xcoords[i+1]
        dim1 = Xcoords[i]
        hi = Ycoords[i+1]
        him1 = Ycoords[i]

        v1 = v1 + (di-dim1)*(hi+him1)
        v2 = v2 + (di-dim1)*(hi*(2*di+dim1)+him1*(di+2*dim1))

    hstip = (2*v1*d-v2)/(d**2)
    hsrip = (v2-v1*d)/(d**2)

    hobi = Ycoords[1:(len(Ycoords)-1)] - (hts*(d-Xcoords[1:(len(Xcoords)-1)])+hrs*Xcoords[1:(len(Xcoords)-1)])/d
    hobs = max(hobi)
    alphas = hobi/Xcoords[1:(len(Ycoords)-1)]
    alpha_obt = max(alphas)
    alphas =  hobi/(d - Xcoords[1:(len(Ycoords)-1)])
    alpha_obr = max(alphas)

    gt = alpha_obt/(alpha_obt+alpha_obr)
    gr = alpha_obr/(alpha_obt+alpha_obr)

    if hobs <= 0:
        hstp = hstip
        hsrp = hsrip
    else:
        hstp = hstip - hobs*gt
        hsrp = hsrip - hobs*gr
    
    hst = 0
    hsr = 0
    if hstp > Ycoords[1]:
        hst = Ycoords[1]
    else:
        hst = hstp
    if hsrp > Ycoords[-2]:
        hsr = Ycoords[-2]
    else:
        hsr = hsrp

    h_aksent_ts = hts - hst
    h_aksent_rs = hrs - hsr
    #print('hts aksent:',h_aksent_ts)
    #h_ts1.append(h_aksent_ts)
    #print('hrs aksent:',h_aksent_rs)
    #h_rs2.append(h_aksent_rs)

    Xc = Xcoords
    #Yc = Ycoords
    Yc = [0] * len(Ycoords)

    Yc[0] = h_aksent_ts
    Yc[-1] = h_aksent_rs
    Lbs = DeltaBullingtonA(Xc,Yc,wavel,re=re)
    #print('Lbs ',Lbs," dB")
    #Lbs10.append(Lbs)
    Lsph = ITUSpericalEarthDiffraction(d*1000,wavel,h_aksent_ts,h_aksent_rs,re=re*1000)

    #print('Lsph: ',Lsph)
    #Lsph11.append(Lsph)

    L = Lba + (Lsph - Lbs)
    #L12.append(L)

    return L, Lba, Lsph

#2.12
def ITUMultipleCylinders(Xcoords,Ycoords,wavel,rheight,theight,re = 8500000,pltIllustration = 0):
    stringX = [Xcoords[0]]
    stringY = [Ycoords[0]]

    maxeindex = 0
    j = 0
    currentmaxeindex = 0

    while j == 0:
        maxe = -100
        hs = Ycoords[currentmaxeindex]

        for i in range(len(Xcoords)-1-maxeindex):

            hi = Ycoords[i+1+currentmaxeindex]
            dsi = Xcoords[i+1+currentmaxeindex] - Xcoords[currentmaxeindex]
            e = ((hi - hs)/dsi)-(dsi/(2*re))
            #e = (Ycoords[i+1+currentmaxeindex]-Ycoords[currentmaxeindex])/(Xcoords[i+1+currentmaxeindex]-Xcoords[currentmaxeindex]) #???????

            if e > maxe:
                maxe = e
                maxeindex = i+1+currentmaxeindex

        currentmaxeindex = maxeindex
        stringX.append(Xcoords[maxeindex])
        stringY.append(Ycoords[maxeindex])

        if maxeindex == (len(Xcoords)-1):
            j = 1
            
        #--------------------------------------------------------------------------------------------------------------------------------------------------------
    obstaclesX = {}
    obstaclesY = {}
    groupsX = {}
    groupsY = {}
    j = 0
    for i in range(len(stringX)-3):
        Xpoint = stringX[i+1]
        if Xpoint + 250 > stringX[i+2]:
            obstaclesX[j] = [stringX[i+1],stringX[i+2]]
            obstaclesY[j] = [stringY[i+1],stringY[i+2]]
            j = j+1
        elif j > 0:
            if stringX[i+1] not in obstaclesX[j-1]:
                obstaclesX[j] = [stringX[i+1]]
                obstaclesY[j] = [stringY[i+1]]
                j = j+1
        else:
            obstaclesX[j] = [stringX[i+1]]
            obstaclesY[j] = [stringY[i+1]]
            j = j+1

    for i in range(len(obstaclesX)-1):
        for j in range(i+1, (len(obstaclesX)-1)):
            if obstaclesX[i][len(obstaclesX[i])-1] == obstaclesX[j][0]:
                obstaclesX[i].append(obstaclesX[j][len(obstaclesX[j])-1])
                obstaclesY[i].append(obstaclesY[j][len(obstaclesY[j])-1])
    groupsX = copy.deepcopy(obstaclesX)
    groupsY = copy.deepcopy(obstaclesY)

    for i in range(len(obstaclesX)-1):
        #for j in range(len(obstaclesX)-1):
        if obstaclesX[i][len(obstaclesX[i])-1] == obstaclesX[i+1][len(obstaclesX[i+1])-1]:
            if (i+1) in groupsX:
                groupsX.pop(i+1)
                groupsY.pop(i+1)



    groupsX[-1] = [Xcoords[0]]
    groupsY[-1] = [Ycoords[0]]
    groupsX[len(Xcoords)] = [Xcoords[-1]]
    groupsY[len(Xcoords)] = [Ycoords[-1]]

    groupsX = {k: v for k, v in sorted(groupsX.items(), key=lambda item: item[1])}    
    
    #print(groupsX)
    #print(groupsY) 
    
    Xkeys = list(groupsX)

    L = 0
    S1 = []
    S2 = []


    
    for i in range(len(Xkeys)-2):
        #print('key:',Xkeys[i])

        Wx = groupsX[Xkeys[i]][-1]
        Wy = groupsY[Xkeys[i]][-1]
        #print('Wx: ',Wx)
        #print('Wy: ',Wy)
        Zx = groupsX[Xkeys[i+2]][0]
        Zy = groupsY[Xkeys[i+2]][0]
        #print('Zx: ',Zx)
        #print('Zy: ',Zy)
        Xx = groupsX[Xkeys[i+1]][0]
        Xy = groupsY[Xkeys[i+1]][0]
        #print('Xx: ',Xx)
        #print('Xy: ',Xy)
        
        Yx = groupsX[Xkeys[i+1]][-1]
        Yy = groupsY[Xkeys[i+1]][-1]
        #print('Yx: ',Yx)
        #print('Yy: ',Yy)
        
        #plt.plot([Wx,Xx,Yx,Zx],[Wy,Xy,Yy,Zy])
        #plt.show()

        alphw = (Xy-Wy)/(Xx-Wx)-(Xx-Wx)/(2*re)
        #print('alphw: ',alphw)
        alphz = (Yy-Zy)/(Zx-Yx)-(Zx-Yx)/(2*re)
        #print('alphz: ',alphz)
        alphe = (Zx - Wx)/re
        #print('alphe: ',alphe)

        Theta = alphw+alphz+alphe
        dwv = 0

        if len(groupsX[Xkeys[i+1]]) == 1:
            dwv = Xx-Wx
        elif (Theta*re)>=(Yx-Xx):
            dwv = ((alphz+alphe/2)*(Zx-Wx)+Zy-Wy)/Theta
        elif (Theta*re)<(Yx-Xx):
            dwv = (Xx-Wx+Yx-Wx)/2
        if Yx==Xx:  #
            dwv = Xx - Wx#

        dvz = Zx - Wx - dwv
        if dvz+dwv != Zx-Wx:
            print('error')
            
        #print('dwv: ',dwv)
        #print('dvz: ', dvz)
        
        S1.append(dwv)
        S2.append(dvz)

        hv = 0
        if len(groupsX[Xkeys[i+1]]) == 1:
            hv = Xy
        elif len(groupsX[Xkeys[i+1]]) > 1:
            hv = dwv*alphw + Wy+(dwv**2)/(2*re)

        h = hv + (dwv*dvz)/(2*re) - (Wy*dvz+Zy*dwv)/(Zx-Wx)
        if isinstance(Xcoords,list):
            ip = Xcoords.index(Xx)-1
            iq = Xcoords.index(Yx)+1
        else:
            ip = Xcoords.tolist().index(Xx)-1
            iq = Xcoords.tolist().index(Yx)+1
        ph = 0
        if ip == 0:
            ph = Ycoords[0] - theight
        else:
            ph = Ycoords[ip]

        qh = 0
        if iq == len(Xcoords)-1:
            qh = Ycoords[-1] - rheight
        else:
            qh = Ycoords[iq]
    
        dpx = Xx - Xcoords[ip]
        dyq = Xcoords[iq] -Yx
        dpq  = Xcoords[iq] - Xcoords[ip]

        t = (Xy-ph)/dpx + (Yy-qh)/dyq - dpq/re
        
        

        v = h*math.sqrt(2/wavel*(1/(dwv)+1/(dvz)))

        R = (dpq/t)*(1-math.exp(-4*v))**3
        
        L = L + ITUSingleRounded([Wx,(Wx+dwv),Zx],[Wy,hv,Zy],wavel,R)
        #print('------------------------------')
        #--------------------------------------------------------------------------------------
    Lsp = 0
    for i in range(len(Xkeys)-1):
        Ux = groupsX[Xkeys[i]][-1]
        Uy = groupsY[Xkeys[i]][-1]
        Vx = groupsX[Xkeys[i+1]][0]
        Vy = groupsY[Xkeys[i+1]][0]
        if isinstance(Xcoords,list):
            Ui = Xcoords.index(Ux)
            Vi = Xcoords.index(Vx)
        else:
            Ui = Xcoords.tolist().index(Ux)
            Vi = Xcoords.tolist().index(Vx)

        ip = Ui + 1
        iq = Vi - 1
        px = Xcoords[ip]
        py = Ycoords[ip]
        
        qx = Xcoords[iq]
        qy = Ycoords[iq]
        #print()
        if (py != qy)&(abs(ip-iq)>1):
            minCf = 10000
            
            for j in range(Ui+1,Vi-1):
                dui = Xcoords[j]-Xcoords[Ui]
                div = Xcoords[Vi]-Xcoords[j]
                duv = Xcoords[Vi]-Xcoords[Ui]
                F1 = math.sqrt(wavel*dui*div/duv)
                
                hr = (Uy*div+Vy*dui)/duv
                ht = Ycoords[j]+dui*div/(2*re)
                
                hz = hr - ht
                Cf = hz/F1
                if Cf < minCf:
                    minCf = Cf

            v = -Cf*math.sqrt(2)
            Vals = special.fresnel(v)
            Cv = Vals[1]
            Sv = Vals[0]

            Jv = -20*math.log10(math.sqrt((1-Cv-Sv)**2+(Cv-Sv)**2)/2)
            if Jv > 0:
                Lsp = Lsp + Jv
                
    #print(Lsp)
    
    Pa = S1[0]*np.prod(S2)*(S1[0]+sum(S2))
    Pb = S1[0]*S2[-1]*np.prod(S1+S2)
    CN = (Pa/Pb)**0.5
    
    LCN = -20*math.log10(CN)
    
    #print(LCN)
    if pltIllustration == 1:
        plt.plot(Xcoords,Ycoords,'-')
        plt.plot(stringX,stringY,'-')
        plt.xlabel('Distance (m)')
        plt.ylabel('Height above mean sea level (m)')
        plt.show()

    return (L+Lsp+LCN)

#2.13
def DeygoutRounded(Xcoords,Ycoords,wavel,Radiusses,pltIllustration = 0,Searth = 0,re = 8500000):
    if pltIllustration == 1 :
               
        fig, ax = plt.subplots() 
        circles = []
        for x1, y1, r in zip(Xcoords[1:len(Xcoords)],Ycoords[1:len(Ycoords)],Radiusses):
            circle = plt.Circle((x1,y1), r)
            circles.append(circle)

        p = PatchCollection(circles, cmap = cm.prism, alpha = 0.4)
        ax.add_collection(p)
        plt.plot(Xcoords,Ycoords,'*')
        plt.xlabel('Distance (m)  r')
        plt.ylabel('Height above sea level (m)  r')
        
    def DeygoutRoundedLoss(Xcoords,Ycoords,wavel,radiusses): #Rekursie is stadig, improve
        L = 0
        NumEdges = len(Xcoords) - 2
        FresnelParams = []

        for i in range(NumEdges):
            distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[-1]],[Ycoords[0],Ycoords[i+1],Ycoords[-1]],Searth=Searth,re=re)
            v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
            FresnelParams.append(v)
        if len(Xcoords) < 3:
            return 0
        else:
            MaxV = np.where(FresnelParams == np.amax(FresnelParams))

            L = L + ITUSingleRounded([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[Ycoords[0],Ycoords[MaxV[0][0].astype(int)+1],Ycoords[-1]],wavel,radiusses[MaxV[0][0].astype(int)])

            if pltIllustration == 1:
                plt.plot([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[Ycoords[0],Ycoords[MaxV[0][0].astype(int)+1],Ycoords[-1]])
            if len(Xcoords[0:(MaxV[0][0].astype(int)+2)]) >=3:
                L = L + DeygoutRoundedLoss(Xcoords[0:(MaxV[0][0].astype(int)+2)],Ycoords[0:(MaxV[0][0].astype(int)+2)],wavel,radiusses[0:(MaxV[0][0].astype(int)+1)]) #Python is weird en die twede parameter moet een meer wees as die index wat jy soek
            if len(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)]) >= 3:
                #print('N')

                L = L + DeygoutRoundedLoss(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],Ycoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],wavel,radiusses[MaxV[0][0].astype(int):len(radiusses)])

            return L

    L = DeygoutRoundedLoss(Xcoords,Ycoords,wavel,Radiusses)
    if pltIllustration == 1 :
        plt.show() 
    return L


def main():

    intlength = 1000000 #meter
    rheight = 30 #meter
    theight = 30 #meter

    f = 1000000000#Hz
    wavel = WaveLength(f)
    kfactor = 4/3

    GetRadiusses = 1
    re_ = kfactor * 6371000


    #DiffractionControl("C:/Users/marko/Desktop/FYP/book2.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    DiffractionControl("C:/Users/marko/Desktop/FYP/book3.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 0,EarthDiffraction = 0,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book4.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book5.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    
    rheight = 10 #meter
    theight = 50 #meter
    f = 2500000000#Hz


    #DiffractionControl("C:/Users/marko/Desktop/FYP/book2.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book3.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book4.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book5.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    
    rheight = 20 #meter
    theight = 20 #meter
    f = 600000000#Hz



    #DiffractionControl("C:/Users/marko/Desktop/FYP/book2.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book3.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book4.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book5.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    
    rheight = 50 #meter
    theight = 40 #meter
    f = 200000000#Hz

    #DiffractionControl("C:/Users/marko/Desktop/FYP/book2.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book3.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book4.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book5.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    

    #rheight = 5 #meter
    #theight = 70 #meter
    #f = 150000000#Hz

    #DiffractionControl("C:/Users/marko/Desktop/FYP/book2.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book3.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book4.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    #DiffractionControl("C:/Users/marko/Desktop/FYP/book5.csv",intlength,theight,rheight,f,kfactor = 4/3,roundEarth = 1,EarthDiffraction = 1,KnifeEdgeMethod=[0,1,2,3,4],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [0,1,2], PlotFunc = 0)
    
    #filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =(("csv files","*.csv"),("all files","*.*")))

    #data, colnames = GetTerrain("C:/Users/marko/Desktop/FYP/book5.csv")

    #validation = InputValidate(data)

    #if validation == 0:
    #    print('Input data is incomplete.')
    #    print('Ensure that each terrain profile consists of a height column and distance column.')
    #    print('Ensure that each distance value is accompanied by a height value.')



    #distarr, heightarr, distance = TerrainDivide(data,colnames[0],colnames[1],intlength,1)

    #L = ITUMultipleCylinders(distarr,heightarr,wavel,rheight,theight,re = 8500000,pltIllustration = 1)

    #xintersect, yintersect, Tdist, Theight, Rdist, Rheight = FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel,plotZone = 0,Searth = 0,re = re_)

    #knifeX, knifeY, radiusses = KnifeEdges(xintersect, yintersect, distarr, heightarr, Rheight, Theight, 4, GetRadiusses, 0)
    #L = EpsteinPeterson(knifeX,knifeY,wavel,Searth = 0,re = re_)
    #print(L)
    #L = Bullington(knifeX,knifeY,wavel,pltIllustration = 0,Searth = 0,re = re_)
    #print(L)
    #L = Deygout(knifeX,knifeY,wavel,pltIllustration = 0,Searth = 0,re = re_)
    #print(L)
    #L = Giovaneli(knifeX,knifeY,wavel,pltIllustration = 0,Searth = 0,re = re_)
    #print(L)
    #print(knifeX)
    #print(radiusses)
    #yintersectN = copy.deepcopy(yintersect)
    #yintersectN.insert(0,Theight)
    #yintersectN.insert(-1,Rheight)
    #xintersectN = copy.deepcopy(xintersect)
    #xintersectN.insert(0,distarr[0])
    #xintersectN.append(distarr[-1])

    #x = np.arange(0, 8000, 10)
    #y = 30*np.sin(0.002*x)+70
    #y[0]=y[0]+10
    #y[-1]=y[-1]+10

    #plt.plot(x, y)
    #plt.show()
    

    #L = EpsteinPetersonRounded(knifeX,knifeY,radiusses,wavel,Searth = 0,re = 8500000)
    #print(L)
    #x3 = np.asarray([0,800,1300,1400,2000,2300,2700,3100,3600,3650,3800,4200])
    #y3 = np.asarray([60,45,60,29,83,70,45,50,90,76,45,60])
    #L = ITUMultipleCylinders(x3, y3,wavel,60,60,re = re_,pltIllustration = 1)
    #print(L)

    #heightarrN = copy.deepcopy(heightarr)
    #heightarrN[0] = heightarrN[0] + theight
    #heightarrN[-1] = heightarrN[-1] + rheight

    #L = DeltaBullington(distarr,heightarrN,wavel)

    #L = Bullington(knifeX,knifeY,wavel,pltIllustration = 1,Searth = 0,re = re_)
    #L = Bullington(knifeX,knifeY,wavel,pltIllustration = 1,Searth = 0,re = re_)

    #L = EpsteinPeterson(knifeX,knifeY,wavel,Searth = 0,re = re_)
    #L = EpsteinPeterson(knifeX,knifeY,wavel,Searth = 0,re = re_)

    #L = Deygout(knifeX,knifeY,wavel,pltIllustration = 1,Searth = 0,re = re_)
    #L = Deygout(knifeX,knifeY,wavel,pltIllustration = 1,Searth = 0,re = re_)

    #L = Giovaneli(knifeX,knifeY,wavel,pltIllustration = 1,Searth = 0,re = re_)

    #L = Vogler([0,5000,11000,15000,22000,25000],[10,40,25,10,15,5],0.5,re = re_)

    #L = DeygoutRounded(knifeX,knifeY,wavel,radiusses,pltIllustration = 1,Searth = 0,re = re_)

    #L = ITUMultipleCylinders(xintersect, yintersect,wavel,theight,rheight,re = re_,pltIllustration = 1)

    #L = ITUTwoEdge(knifeX,knifeY,wavel)

    #L = ITUTwoRounded(knifeX,knifeY,radiusses,wavel)

    #L = FresnelKirchoff(knifeX,knifeY,wavel,Searth = 0,re = re_)

    #L = ITUSingleRounded(knifeX,knifeY,wavel,radiusses[0])

    #L = ITUSpericalEarthDiffraction(distance,wavel,knifeY[0],knifeY[-1],re = re_)

    #L = Bullington([0,7000,12000,22000,26000],[0,30,50,20,0],0.5)
    #print(L)
    #L = EpsteinPeterson([0,7000,12000,22000,26000],[0,30,50,20,0],0.5)
    #print(L)
    #L = Deygout([0,7000,12000,22000,26000],[0,30,50,20,0],0.5)
    #print(L)

    x1 = [0,5000,9000,15000,22000,28000]
    y1 = [0,30,22,51,35,10]

    x2 = [0,3000,10000,13000,17000,21000,26000,30000,33000]
    y2 = [15,50,37,45,120,22,45,29,30]

    x3 = [0,8000,13000,14000,20000,23000,27000,31000,36000,36500,38000,42000]
    y3 = [60,45,60,29,83,70,45,50,90,76,45,60]

    #L = Giovaneli(x1,y1,wavel,pltIllustration = 1,Searth = 0,re = re_)
    #L = Giovaneli(x2,y2,wavel,pltIllustration = 1,Searth = 0,re = re_)
    #L = Giovaneli(x3,y3,wavel,pltIllustration = 1,Searth = 0,re = re_)

    #plt.plot([0,0],[0,10],'k')
    #plt.plot([28000,28000],[0,10],'k')
    #plt.plot([0],[10],'kx')
    #plt.plot([28000],[10],'kx')
    #plt.plot([0,28000],[10,10],'--')

    #plt.plot([5000,5000],[0,30],'k')
    #plt.plot([9000,9000],[0,22],'k')
    #plt.plot([15000,15000],[0,51],'k')
    #plt.plot([22000,22000],[0,35],'k')
    #plt.xlabel('Distance (m)')
    #plt.ylabel('Height above mean sea level (m)')
    #plt.autoscale(enable=True, axis='y', tight=True)
    #plt.show()

    #plt.plot(x1,y1,'*')
    #plt.xlabel('Distance (m)')
    #plt.ylabel('Height (m)')
    #plt.grid()
    #plt.show()
    #plt.plot(x2,y2,'*')
    #plt.xlabel('Distance (m)')
    #plt.ylabel('Height (m)')
    #plt.grid()
    #plt.show()
    #plt.plot(x3,y3,'*')
    #plt.xlabel('Distance (m)')
    #plt.ylabel('Height (m)')
    #plt.grid()
    #plt.show()

    #L1b = Bullington(x1,y1,0.5,pltIllustration = 1)
    #print('--------------------------------------------')
    #L2b = Bullington(x2,y2,0.3)
    #print('--------------------------------------------')
    #L3b = Bullington(x3,y3,5)
    #print('--------------------------------------------')
    #print('b1 ',L1b)
    #print('b2 ',L2b)
    #print('b3 ',L3b)


    #l1 = FresnelKirchoff([0,12459.02,28000],[0,74.75,10],0.5)
    #print(l1)
    #l1 = FresnelKirchoff([0,11602.41,33000],[15,150.36,30],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([0,34146.34,42000],[60,99.27,60],5)
    #print(l1)



    #l1 = FresnelKirchoff(x3[0:3],y3[0:3],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[1:4],y3[1:4],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[2:5],y3[2:5],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[3:6],y3[3:6],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[4:7],y3[4:7],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[5:8],y3[5:8],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[6:9],y3[6:9],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[7:10],y3[7:10],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[8:11],y3[8:11],5)
    #print(l1)
    #l1 = FresnelKirchoff(x3[9:12],y3[9:12],5)
    #print(l1)

    #L1e = EpsteinPeterson(x1,y1,0.5)
    #print('--------------------------------------------')
    #L2e = EpsteinPeterson(x2,y2,0.3)
    #print('--------------------------------------------')
    #L3e = EpsteinPeterson(x3,y3,5)
    #print('--------------------------------------------')
    #print('e1 ',L1e)
    #print('e2 ',L2e)
    #print('e3 ',L3e)


    L1d = Bullington(x1,y1,0.5,pltIllustration = 1)
    #print('--------------------------------------------')
    L2d = Bullington(x2,y2,0.3,pltIllustration = 1)
    #print('--------------------------------------------')
    L3d = Bullington(x3,y3,5,pltIllustration = 1)
    #print('--------------------------------------------')
    #print('d1 ',L1d)
    #print('d2 ',L2d)
    #print('d3 ',L3d)

    #l1 = FresnelKirchoff([x1[0],x1[3],x1[5]],[y1[0],y1[3],y1[5]],0.5)
    #print(l1)
    #l1 = FresnelKirchoff([x1[0],x1[1],x1[3]],[y1[0],y1[1],y1[3]],0.5)
    #print(l1)
    #l1 = FresnelKirchoff([x1[1],x1[2],x1[3]],[y1[1],y1[2],y1[3]],0.5)
    #print(l1)
    #l1 = FresnelKirchoff([x1[3],x1[4],x1[5]],[y1[3],y1[4],y1[5]],0.5)
    #print(l1)


    #l1 = FresnelKirchoff([x2[0],x2[4],x2[8]],[y2[0],y2[4],y2[8]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[0],x2[1],x2[4]],[y2[0],y2[1],y2[4]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[1],x2[2],x2[4]],[y2[1],y2[2],y2[4]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[2],x2[3],x2[4]],[y2[2],y2[3],y2[4]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[4],x2[7],x2[8]],[y2[4],y2[7],y2[8]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[4],x2[6],x2[7]],[y2[4],y2[6],y2[7]],0.3)
    #print(l1)
    #l1 = FresnelKirchoff([x2[4],x2[5],x2[6]],[y2[4],y2[5],y2[6]],0.3)
    #print(l1)


    #l1 = FresnelKirchoff([x3[0],x3[8],x3[11]],[y3[0],y3[8],y3[11]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)
    #l1 = FresnelKirchoff([x3[0],x3[0],x3[0]],[y3[0],y3[0],y3[0]],5)
    #print(l1)


if __name__ == '__main__':
    main()
