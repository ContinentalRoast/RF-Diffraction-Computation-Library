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

import seaborn as sns
sns.set()

import time

from matplotlib.collections import PatchCollection
from matplotlib import cm 

import copy
import sys




#0.1
def DiffractionControl(fname,IntervalLength,TransmitterHeight,ReceiverHeight,Frequency,kfactor = 4/3,roundEarth = 0,EarthDiffraction = 0,KnifeEdgeMethod=[0],RoundedObstacleMethod = [0],TwoObstacleMethod = [0],SingleObstacleMethod = [1], PlotFunc = 0):
    '''
    INPUT PARAMETERS
    fname: The file path of the csv file that contains one or more terrain profiles.
    IntervalLength: The length of each interval along the propagation path that diffraction loss must be calculated. (m)
    TransmitterHeight: This is the height of the transmitter above the terrain. (m)
    ReceiverHeight: This is the height of the receiver above the terrain. (m)
    Frequency: The frequency of the transmitted signal. (Hz)
    kfactor: The factor used to determine the effective radius of Earth. Is taken as 4/3 when not provided.
    roundEarth: If this variable is equal to 1 a round Earth approach is used when implementing certain diffraction computation methods.
    EarthDiffraction: The diffraction due to the spherical shape of the Earth when there are no obstacles in the first Fresnel zone is only calculated if this parameter is assigned a value of one.
    KnifeEdgeMethod: This array can contain any of the following values in order to use certain methods:
                    •	0: Delta-Bullington method 
                    •	1: Bullington method 
                    •	2: Epstein-Peterson method 
                    •	3: Deygout method 
                    •	4: Giovaneli method

    RoundedObstacleMethod: This array can contain any of the following values in order to use certain methods:
                    •	Only 0: No rounded obstacle method is used.
                    •	1: Deygout rounded
                    •	2: ITU Multiple Cylinders

    TwoObstacleMethod: This array can contain any of the following values in order to use certain methods:
                    •	Only 0: No two-obstacle method is used.
                    •	1: ITU Two Knife-Edge method 
                    •	2: ITU Two Rounded Obstacle method

    SingleObstacleMethod: This array can contain any of the following values in order to use certain methods:
                    •	Only 0: No single knife-edge method is used when there is only one obstacle present in the propagation path.
                    •	1: Fresnel-Kirchoff method 
                    •	2: ITU Single Rounded Obstacle method 
    PlotFunc: If this variable is assigned a value of one, all of the functions used with the capability to visualize the diffraction computation process will do so.
    '''
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
    return df

#0.2
def InputValidate(TerrainData):
    '''
    INPUT PARAMETERS
    TerrainData: Terrain profiles in the form of a DataFrame 
    '''
    colnames = TerrainData.columns
    if (len(colnames) % 2) != 0:
        return 0
    for colnum in range(0,len(colnames),2):
        if len(TerrainData[colnames[colnum]]) != len(TerrainData[colnames[colnum+1]]):
            return 0

    return 1

#0.3
def OutputValidate(Loss):
    '''
    INPUT PARAMETERS
    Loss: Diffraction loss. (dB)
    '''
    if Loss < 0:
        return 0
    else:
        return Loss

#0.4
def GetTerrain(fname):
    '''
    INPUT PARAMETERS
    fname: The file path of the csv file that contains one or more terrain profiles.
    '''

    df = pd.read_csv(fname)

    return df, df.columns

#0.5
def TerrainDivide(data, colnamex, colnamey, intlength, iterationNum): 
    '''
    INPUT PARAMETERS
    data: All of the terrain profiles in the form of a DataFrame. 
    colnamex: The key is used to select the correct column of values from the terrain profile data frame.
    colnamey: The key is used to select the correct column of values from the terrain profile data frame.
    intlength: The length of one interval along the propagation path.
    iterationNum: This is used to determine the total distance of terrain profile that must be extracted. That is how many intervals along the path.  
    '''
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
    ''' 
    INPUT PARAMETERS
    distarr: Distance values. (m)
    heightarr: Mean height above sea level values. (m)
    rheight: This is the height of the receiver above the terrain. (m)
    theight: This is the height of the transmitter above the terrain. (m)
    wavel: Wavelength of the signal. (m)
    plotZone: If this variable is assigned a value of one the Freznel zone and terrain is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Frequency: The frequency of the transmitted signal. (Hz)
    '''
    c = 300000000 #speed of light m/s
    wavel = c/frequency #wavelength or lambda
    return wavel

#1.3
def ObstacleValues(Xcoords,Ycoords,Searth = 0,re = 8500000):
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. (m)
    Ycoords: Mean height above sea level values. (m)
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
    distance1 = ((Xcoords[1]-Xcoords[0])**2+(Ycoords[1]-Ycoords[0])**2)**(1/2)
    distance2 = ((Xcoords[2]-Xcoords[1])**2+(Ycoords[2]-Ycoords[1])**2)**(1/2)
    mLoS = (Ycoords[2]-Ycoords[0])/(Xcoords[2]-Xcoords[0])-((Xcoords[2]-Xcoords[0])/(2*re))*Searth
    bLoS = Ycoords[2] - Xcoords[2]*mLoS
    height = Ycoords[1]-(mLoS*Xcoords[1]+bLoS)

    return distance1,distance2,height

#1.4
def KnifeEdges(xintersect, yintersect, distarr, heightarr, Rheight, Theight, sensitivity, cylinders = 0, plotoutput = 0):
    '''
    INPUT PARAMETERS
    xintersect: Distance values in first Fresnel zone. (m)
    yintersect: Height values in first Fresnel zone.
    distarr: Distance values. (m)
    heightarr: Mean height above sea level values. (m)
    Rheight: Receiver height above mean sea level (m)
    Theight: Transmitter height above mean sea level (m)
    sensitivity: The sensitivity of the Gaussian function parameter enables the user to adjust the degree of the filter. 
                 Thus, for example a sensitivity value of one will in turn mean that a first order Gaussian filter will be implemented.
    cylinders: This Boolean value has the following effect on computations depending on the value assigned:
                •	0: Radiuses of obstacles are not calculated 
                •	1: Radiuses of obstacles are calculated

    plotoutput: This Boolean value has the following effect depending on the value assigned:
                •	0: Output values are not visualised 
                •	1: Radiuses and knife-edge values of the terrain is plotted.
    '''

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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    radius: Radius of obstacle. (m)
    '''
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
    '''
    INPUT PARAMETERS
    d: Path length. (m)
    wavel: Wavelength of signal. (m)
    h1: Transmitter height. (m)
    h2: Receiver height. (m)
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    dm: Path length. (m)
    wavel: Wavelength of signal. (m)
    h1: Transmitter height. (m)
    h2: Receiver height. (m)
    re: Equivalent Earth radius. (m)
    '''
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
def ITUTwoEdge(Xcoords,Ycoords,wavel):
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    '''

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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    radii: Radiuses of obstacles. (m)
    '''

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
def Bullington(Xcoords,Ycoords,wavel,pltIllustration = 0,Searth = 0,re = 8500000):
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. First and last value represents transmitter end receiver. (km)
    Ycoords: Mean height above sea level values. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    re: Equivalent Earth radius. (km)
    '''

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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    rheight: Receiver height above terrain. (m)
    theigh: Transmitter height above terrain. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    re: Equivalent Earth radius. (m)
    '''
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
    '''
    INPUT PARAMETERS
    Xcoords: Distance values of knife-edges. First and last value represents transmitter end receiver. (m)
    Ycoords: Mean height above sea level values of knife-edges. First and last value represents transmitter end receiver. (m)
    wavel: Wavelength of the signal. (m)
    Radiusses: Radiusses of obstacles. (m)
    pltIllustration: If this variable is assigned a value of one the method is visualised.
    Searth: If this variable is equal to 1 a round Earth approach is used.
    re: Equivalent Earth radius. (m)
    '''
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


