import csv
import os
import math
#from numpy import genfromtxt
from tkinter import filedialog
import pandas as pd
import numpy as np
from pandas import read_csv
#import matplotlib
#from sympy import Ellipse, Poinr, Rational
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import quad
from scipy import special

from mpmath import nsum
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import find_peaks


def WaveLength(frequency):
    c = 300000000 #speed of light m/s
    wavel = c/frequency #wavelength or lambda
    return wavel

def TerrainDivide(fname, intlength,ptpindex):
    Dist = []
    Height = []
    #filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =(("csv files","*.csv"),("all files","*.*")))
    filename = fname
    dist = []
    heig = []
    ofile = open(filename,'r')
    reader = csv.reader(ofile, delimiter=',')
    for row in reader:
        dist.append(row[0])
        heig.append(row[1])
        
    x = np.array(dist[1:])
    Dist= x.astype(np.float)
    Dist = Dist*1000
    
    y = np.array(heig[1:])
    Height = y.astype(np.float)


    if Dist[-1] > intlength:
        index = np.where(Dist >= intlength)
        pindex = index[0][0]
    else:
       pindex = len(Dist)

    #plt.plot(Dist[:pindex], Height[:pindex],'-')
    #plt.show()
    return Dist[:pindex], Height[:pindex]

def FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel):

    Tdist = distarr[0] 
    Theight = theight + heightarr[0]
    Rdist = distarr[len(distarr)-1]
    Rheight = rheight + heightarr[len(heightarr)-1]

    m = (Rheight-Theight)/(Rdist-Tdist)
    b = Theight
    length = math.sqrt(Rdist**2+(Rheight-Theight)**2)
    rangle = np.arctan((Rheight-Theight)/Rdist)

    #---------------------------------------------------------------------------------------------------------------------------
    #The following section of commented code was used to test ellipse obtained from the matplotlib.patches Ellipse function
    #against the ellipse function as descriped by Fresnel

    #cx = distarr[len(distarr)//2-1]
    #cy = m*cx+b
    

    #R = ((wavel*(length/2)**2)/(length))**(1/2)

    #angle = np.arctan((Rheight-Theight)/Rdist)*180/np.pi
    

    #ells = Ellipse((cx, cy), length, R*2, angle)

    #print(ells.get_path)

    #a = plt.subplot()

    #a.add_artist(ells)
    #---------------------------------------------------------------------------------------------------------------------------

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
        RadiusXValues1.append(x1)
        RadiusXValues2.append(x2)
        RadiusYValues1.append(y1)
        RadiusYValues2.append(y2)

    plt.plot(RadiusXValues2,RadiusYValues2,'k-')
    plt.plot(RadiusXValues1,RadiusYValues1,'-')

    xintersect = []
    yintersect = []

    for xcoord, ycoord in zip(distarr,heightarr):
        index = np.where(RadiusXValues2 >= xcoord)
        pIndex = index[0][0]
        #print(xcoord)
        #print(RadiusXValues2[pIndex])
        #print('---------------')

        if ycoord >= RadiusYValues2[pIndex]:
            xintersect.append(xcoord)
            yintersect.append(ycoord)


    


    plt.plot(distarr,heightarr,'-')
   #plt.plot(distarr,ysmoothed,'-')
    plt.plot(xintersect,yintersect,'m-')
    plt.plot((Tdist,Rdist),(Theight,Rheight),'r-')
    plt.show()

    return xintersect, yintersect, Tdist, Theight, Rdist, Rheight




def KnifeEdges(xintersect, yintersect, wavel, distarr, heightarr, Rheight, Theight, cylinders):

    ysmoothed = gaussian_filter1d(heightarr,sigma=4)
    ysmoothed[0] = Theight
    ysmoothed[-1] = Rheight

    localmin = (np.diff(np.sign(np.diff(ysmoothed))) > 0).nonzero()[0] + 1         #local minimums
    localmax = (np.diff(np.sign(np.diff(ysmoothed))) < 0).nonzero()[0] + 1         # local max

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
    county = 0
    countr = 0
    p1 = 25
    p3 = 75
    for i in range(len(localmin)-1):
        g = np.where(np.logical_and(xintersect>distarr[localmin[i]], xintersect<distarr[localmin[i+1]]))
        radiussum = 0
        if len(g[0]):
            fi = len(g[0])-1
            middley = (max(yintersect[g[0][0]:g[0][fi]])-min(yintersect[g[0][0]:g[0][fi]]))/2
            q1 = np.percentile(xintersect[g[0][0]:g[0][fi]],p1)
            q3 = np.percentile(xintersect[g[0][0]:g[0][fi]],p3)
            middlex = (q3-q1)/2
            if((middlex == 0)|(middley==0)):
                radiusses.append(0)
            else:
                radiusses.append(middlex)
            #fi = len(g[0])-1
            #averageX = sum(xintersect[g[0][0]:g[0][fi]])/(fi+1)
            #for xc, yc in zip(xintersect[g[0][0]:g[0][fi]],yintersect[g[0][0]:g[0][fi]]):
            #    xi = abs(averageX-xc)
            #    yi = knifeY[county]-yc
            #    if((yi!=0)&(xi!=0)):
            #        r = (xi**2)/(2*yi)
            #        print(r)
            #        radiussum = radiussum + r
            #        countr = countr+1
            #radiusses.append(radiussum/countr)
            #countr = 0
            #county = county + 1
    #-----------------------------------------------------------------------------------------------------------------------
    #print(radiusses)
    #circle1 = plt.Circle((knifeX[0],knifeY[0]), radiusses[0])
    #circle2 = plt.Circle((knifeX[1],knifeY[1]), radiusses[1])
    #circle3 = plt.Circle((knifeX[2],knifeY[2]), radiusses[2])
    #circle4 = plt.Circle((knifeX[3],knifeY[3]), radiusses[3])

    #fig, ax = plt.subplots()
    #ax.add_artist(circle1)
    #ax.add_artist(circle2)
    #ax.add_artist(circle3)
    #ax.add_artist(circle4)

    #print('Knife edges: ',len(knifeX))
    #print('Cylinders: ',len(radiusses))
    #print(radiusses)
    ##plt.plot(distarr[localmin], ysmoothed[localmin], "o", label="min", color='r')
    plt.plot(knifeX,knifeY,'x')
    #plt.plot(sknifeX,knifeY,'x')
    plt.plot(distarr,heightarr,'-')
    #plt.plot(distarr,ysmoothed,'-')
    plt.plot(xintersect,yintersect,'m-')
    plt.show()

    return knifeX, knifeY

def ITUSpericalEarthDiffraction(d,wavel,h1,h2):
    L = 0
    dm = d *1000
    ae = 8500000
    b = 1
    #print('d:',d)
    d_los = math.sqrt(2*ae)*(h1**(1/2)+h2**(1/2))
    #print('d_los:',d_los)
    if dm >= d_los/1000:
        L = ITUNLoS(dm,wavel,h1,h2,ae)

    if dm < d_los:
        c = (h1-h2)/(h1+h2)
        m = dm**2/(4*ae*(h1+h2))
        b_ = 2*math.sqrt((m+1)/(3*m))*math.cos(math.pi/3+1/3*math.acos(3*c/2*math.sqrt(3*m/(m+1)**3)))
        d1 = dm/2*(1+b_)
        d2 = dm-d1
        h = ((h1-(d1**2)/(2*ae))*d2+(h2-(d2**2)/(2*ae))*d1)/dm
        hreq = 0.552*math.sqrt(d1*d2*wavel/dm)

        if h > hreq:
            L = 0
        else:
            aem = 0.5*(dm/(h1**(1/2)+h2**(1/2)))**2
            Ah = ITUNLoS(dm,wavel,h1,h2,aem)
            if Ah < 0:
                L = 0
            else:
                A = [1-h/hreq]*Ah
                L = A

    return L

    


def ITUNLoS(d,wavel,h1,h2,ae):

    b = 1
    f = 300000000/wavel/1000000

    X = b*d*(math.pi/(wavel*(ae**2)))**(1/3)
    #X = 2.188*b*f**(1/3)*ae**(-2/3)*d

    Y1 = 2*b*h1*(math.pi**2/((wavel**2)*ae))**(1/3)
    Y2 = 2*b*h2*(math.pi**2/((wavel**2)*ae))**(1/3)

    #Y1 = 9.575*10**(-3)*b*f**(2/3)*ae**(-1/3)*h1
    #Y2 = 9.575*10**(-3)*b*f**(2/3)*ae**(-1/3)*h2


    FX = 0
    if X > 1.6:
        FX = 11+10*math.log10(X)-17.6*X
    else:
        FX = -20*math.log10(X)-5.6488*X**(1.425)

    GY1 = 0
    GY2 = 0
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

    print('X: ',X)
    print('Y1: ',Y1)
    print('Y2: ',Y2)

    print('FX: ',FX)
    print('GY1: ',GY1)
    print('GY2: ',GY2)

    
    print('20log(E/E0) = ',L)

    return L



def ObstacleValues(Xcoords,Ycoords):
    distance1 = ((Xcoords[1]-Xcoords[0])**2+(Ycoords[1]-Ycoords[0])**2)**(1/2)
    distance2 = ((Xcoords[2]-Xcoords[1])**2+(Ycoords[2]-Ycoords[1])**2)**(1/2)

    mLoS = (Ycoords[2]-Ycoords[0])/(Xcoords[2]-Xcoords[0])
    bLoS = Ycoords[2] - Xcoords[2]*mLoS
    height = Ycoords[1]-(mLoS*Xcoords[1]+bLoS)
    return distance1,distance2,height

def FresnelKirchoff(Xcoords,Ycoords,wavel):

    distance1,distance2,height = ObstacleValues(Xcoords,Ycoords)
    #print('Height: ',height,'d1:',distance1,'d2',distance2)
    v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))

    #METHOD 1
    #s = sp.Symbol('s')
    #def C(s):
    #    return math.cos((math.pi*s**2)/2)
    #def S(s):
    #    return math.sin((math.pi*s**2)/2)

    #Cv = quad(C,0,v)
    #Sv = quad(S,0,v)
    #print(Cv[0])
    #print(Sv[0])

    #METHOD 2
    #print(special.fresnel(v)[1])    #C(v)
    #print(special.fresnel(v)[0])    #S(v)
    Vals = special.fresnel(v)
    Cv = Vals[1]
    Sv = Vals[0]
    Jv = -20*math.log10(math.sqrt((1-Cv-Sv)**2+(Cv-Sv)**2)/2)  #J(v) is the diffraction loss in dB
    #print(Jv)
    return Jv

def ITUSingleRounded(Xcoords,Ycoords,wavel,radius):
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
    
    m2 = (Ry-Ob1y)/(Rx-Ob1x)
    b2 = Ob1y - m2*Ob1x

    h1 = Ob1y-(m1*Ob1x + b1)
    h2 = Ob2y-(m2*Ob2x + b2)

    r1 = ((wavel*(Ob1x-Tx)*((Rx-Ob1x-Tx)))/(Rx-Tx))**(1/2)
    r2 = ((wavel*(Ob2x-Tx)*((Rx-Ob2x-Tx)))/(Rx-Tx))**(1/2)
    ratio1 = h1/r1
    ratio2 = h2/r2

    a = Ob1x-Tx
    b = Ob2x-Ob1x
    c = Rx-Ob2x

    
    if (ratio1-ratio2)**2 > 2: #This condition must be refined
        L1 = FresnelKirchoff([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel)
        L2 = FresnelKirchoff([Ob1x,Ob2x,Ry],[Ob1y,Ob2y,Ry],wavel)
        Lc = 0
        if (L1 > 15)&(L2>15):
            Lc = 10*math.log10(((a+b)*(b+c))/(b*(a+b+c)))

        return (L1 + L2 + Lc)
    
    elif ratio1 > ratio2:

        L1 = FresnelKirchoff([Tx,Ob1x,Rx],[Ty,Ob1y,Ry],wavel)
        L2 = FresnelKirchoff([Ob1x,Ob2x,Rx],[Ob1y,Ob2y,Ry],wavel)
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.arctan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        return (L1+L2-Tc)


    elif ratio2 > ratio1:

        L1 = FresnelKirchoff([Tx,Ob2x,Rx],[Ty,Ob2y,Ry],wavel)
        L2 = FresnelKirchoff([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel)
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.arctan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        return (L1+L2-Tc)

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
    
    m2 = (Ry-Ob1y)/(Rx-Ob1x)
    b2 = Ob1y - m2*Ob1x

    h1 = Ob1y-(m1*Ob1x + b1)
    h2 = Ob2y-(m2*Ob2x + b2)

    r1 = ((wavel*(Ob1x-Tx)*((Rx-Ob1x-Tx)))/(Rx-Tx))**(1/2)
    r2 = ((wavel*(Ob2x-Tx)*((Rx-Ob2x-Tx)))/(Rx-Tx))**(1/2)
    ratio1 = h1/r1
    ratio2 = h2/r2

    a = Ob1x-Tx
    b = Ob2x-Ob1x
    c = Rx-Ob2x

    if (ratio1-ratio2)**2 > 2: #This condition must be refined

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
        alpha = math.arctan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        return (L1+L2-Tc)


    elif ratio2 > ratio1:

        L1 = ITUSingleRounded([Tx,Ob2x,Rx],[Ty,Ob2y,Ry],wavel,radii[0])
        L2 = ITUSingleRounded([Tx,Ob1x,Ob2x],[Ty,Ob1y,Ob2y],wavel,radii[1])
        p = (2/wavel*((a+b+c)/((b+c)*a)))**(1/2)*h1
        q = (2/wavel*((a+b+c)/((b+a)*c)))**(1/2)*h2
        alpha = math.arctan((b*(a+b+c)/(a*c))**(1/2))
        Tc = (12-20*math.log10(2/(1-(alpha/math.pi))))*(q/p)**(2*p)

        return (L1+L2-Tc)

def Bullington(Xcoords,Ycoords,wavel):
    Tx = Xcoords[0]
    Ty = Ycoords[0]
    Rx = Xcoords[len(Xcoords)-1]
    Ry = Ycoords[len(Ycoords)-1]

    maxy = max(Ycoords[1:(len(Ycoords)-1)])

    mTR = (Ry-Ty)/(Rx-Tx)
    bTR = Ry - mTR*Rx
    ldy = 0

    print(Xcoords)
    print(Xcoords[1:(len(Xcoords)-1)])

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
        mtemp1 = (ycoord-Ty)/(xcoord-Tx)
        mtemp2 = (Ry-ycoord)/(Rx-xcoord)
        print(mtemp1)
        print(mtemp2)

        #if ldy > 0:
        #    if mtemp1 > m1:
        #        m1 = mtemp1
        #        b1 = Ty - m1*Tx

        #    if mtemp2 < m2:
        #        m2 = mtemp2
        #        b2 = Ry - m2*Rx
        #else:
        #    if mtemp1 > m1:
        #        m1 = mtemp1
        #        b1 = Ty - m1*Tx

        #    if mtemp2 < m2:
        #        m2 = mtemp2
        #        b2 = Ry - m2*Rx

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
    #print('Ypoint:',Ypoint)
    #print("m1: ",m1)
    #print("m2: ",m2)
    #print("Lenth: ",len(Xcoords)) 
    #print(Xcoords[:(len(Xcoords))])#!?
    #print(Xcoords)
    plt.plot(Xcoords,Ycoords,'x')
    plt.plot([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],'-')
    plt.show()
    return FresnelKirchoff([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],wavel)

def EpsteinPeterson(Xcoords,Ycoords,wavel):
    NumEdges = len(Xcoords) - 2
    L = 0

    for i in range(NumEdges):
        L = L + FresnelKirchoff([Xcoords[i],Xcoords[i+1],Xcoords[i+2]],[Ycoords[i],Ycoords[i+1],Ycoords[i+2]],wavel)

    return L

def Deygout(Xcoords,Ycoords,wavel):

    def DeygoutLoss(Xcoords,Ycoords,wavel,FresnelParams): #Rekursie is stadig, improve
        if len(Xcoords) < 3:
            return 0
        else:
            MaxV = np.where(FresnelParams == np.amax(FresnelParams))

            L = FresnelKirchoff([Xcoords[0],Xcoords[MaxV[0][0].astype(int)+1],Xcoords[-1]],[Ycoords[0],Ycoords[MaxV[0][0].astype(int)+1],Ycoords[-1]],wavel)
            L = L + DeygoutLoss(Xcoords[0:(MaxV[0][0].astype(int)+2)],Ycoords[0:(MaxV[0][0].astype(int)+2)],wavel,FresnelParams[0:MaxV[0][0].astype(int)])
            L = L + DeygoutLoss(Xcoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],Ycoords[(MaxV[0][0].astype(int)+1):len(Xcoords)],wavel,
                                FresnelParams[MaxV[0][0].astype(int)+1:len(FresnelParams)])
                
            return L

    NumEdges = len(Xcoords) - 2
    FresnelParams = []

    for i in range(NumEdges):
        distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[-1]],[Ycoords[0],Ycoords[i+1],Ycoords[-1]])
        v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
        FresnelParams.append(v)

    L = DeygoutLoss(Xcoords,Ycoords,wavel,FresnelParams)
    print(L)

def Vogler(Xcoords,Ycoords,wavel):
    r = []
    #heights = []
    #Theight = Ycoords[0]
    #Rheight = Ycoords[-1]
    length = len(Xcoords)
    N = length-2
    theta = []  #There are some possible knife edge angle events not covered by Vogler
    alpha = []
    beta = []
    k = 2*np.pi/wavel

    for i in range(length-1):
        r.append(Xcoords[i+1]-Xcoords[i])

    for i in range(length-2):
        ang1 = np.arctan((Ycoords[i+1]-Ycoords[i])/(Xcoords[i+1]-Xcoords[i]))*180/np.pi
        ang2 = np.arctan((Ycoords[i+2]-Ycoords[i+1])/(Xcoords[i+2]-Xcoords[i+1]))*180/np.pi
        theta.append(ang1-ang2)

    for i in range(length - 3):
        a = ((r[i]*r[i+2])/(r[i]+r[i+1])*(r[i+1]+r[i+2]))**(1/2)
        alpha.append(a)

    for i in range(length - 2):
        b = theta[i]*((i*k*r[i]*r[i+1])/(2*(r[i]+r[i+1])))**(1/2)
        beta.append(b)


    def integrand(x):
        output = np.exp(-(x)**2.0)
        return output

    g = quad(integrand1,20,np.inf)
    print(g)

    jj = nsum(lambda x: exp(-x**2), [-inf, inf])

def DeltaBullingtonA(Xcoords,Ycoords,wavel):
    #print('Length X:',len(Xcoords))
    #k = 4/3
    #ae = k*6371
    re = 8500
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
        print('v max 1:',Vmax)
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

def DeltaBullingtonB(Xcoords,Ycoords,wavel):
    Lba = DeltaBullingtonA(Xcoords,Ycoords,wavel)
    print('Lba: ',Lba)
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
    print('hts aksent:',h_aksent_ts)
    print('hrs aksent:',h_aksent_rs)

    Xc = Xcoords
    #Yc = Ycoords
    Yc = [0] * len(Ycoords)

    Yc[0] = h_aksent_ts
    Yc[-1] = h_aksent_rs
    Lbs = DeltaBullingtonA(Xc,Yc,wavel)
    print('Lbs ',Lbs," dB")
    Lsph = ITUSpericalEarthDiffraction(d,wavel,h_aksent_ts,h_aksent_rs)

    print('Lsph: ',Lsph)

    L = Lba + (Lsph - Lbs)

    return L
    


def main():
    #the transmitter is at point zero on the distnace axis
    intlength = 200000 #meter
    rheight = 30 #meter
    theight = 30 #meter
    frequency = 1000000000 #Hz

    wavel = WaveLength(frequency)
    #print(wavel)
    distarr, heightarr = TerrainDivide("C:/Users/marko/Desktop/FYP/Book2.csv",intlength,1)
    #xintersect, yintersect,Tdist,Theight,Rdist,Rheight = FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel)

    #knifeX, knifeY = KnifeEdges(xintersect, yintersect, wavel, distarr, heightarr, Rheight, Theight, 1)
    #Cylinders(xintersect, yintersect, wavel, distarr, heightarr, Rheight, Theight, knifeX)

    #ObstacleSmoothness(xintersect, yintersect, wavel)

    #Jv = FresnelKirchoff(20,10000,5000,wavel)
    #A = ITUSingleRounded(20,10000,5000,wavel,15)
    #Xcoords = knifeX
    #Xcoords.append(distarr[-1])
    #Xcoords.insert(0,distarr[0])
    #Ycoords = knifeY
    #Ycoords.append(rheight+heightarr[-1])
    #Ycoords.insert(0,theight+heightarr[0])
    #print(Xcoords)
    #print(Ycoords)
    bXcoords = distarr
    #print(bXcoords)
    bYcoords = heightarr
    bYcoords[-1] = (rheight+heightarr[-1])
    bYcoords[0] = (theight+heightarr[0])
    #L = Bullington(Xcoords,Ycoords,wavel)
    #L = Bullington([0,7000,12000,22000,26000],[0,30,50,20,0],wavel)
    #L = EpsteinPeterson(Xcoords,Ycoords,wavel)
    #L = Deygout(Xcoords,Ycoords,wavel)

    L = DeltaBullingtonB(bXcoords/1000,bYcoords,wavel)

    
    print(L)

    #L = EpsteinPeterson([0,7000,12000,22000,26000],[0,30,50,20,0],wavel)
    #L = Deygout([0,7000,12000,22000,26000],[0,30,50,20,0],wavel)
    #print('Loss: ',L)



if __name__ == '__main__':
    main()