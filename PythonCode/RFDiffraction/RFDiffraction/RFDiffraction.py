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

def WaveLength(frequency):
    c = 299792458 #speed of light m/s
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
    y = np.array(heig[1:])
    Height = y.astype(np.float)

    if Dist[len(Dist)-1] > intlength:
        index = np.where(Dist >= intlength)
        pindex = index[0][0]
    else:
       pindex = len(Dist)-1

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
    #plt.plot(RadiusXValues1,RadiusYValues1,'-')

    xintersect = []
    yintersect = []

    for xcoord, ycoord in zip(distarr,heightarr):
        index = np.where(RadiusXValues2 >= xcoord)
        pIndex = index[0][0]
        #print(xcoord)
        #print(RadiusXValues2[pIndex])
        #print('---------------')

        if ycoord > RadiusYValues2[pIndex]:
            xintersect.append(xcoord)
            yintersect.append(ycoord)

    plt.plot(distarr,heightarr,'-')
    plt.plot(xintersect,yintersect,'m.')
    plt.plot((Tdist,Rdist),(Theight,Rheight),'r-')
    plt.show()

    return xintersect, yintersect


#def ObstacleSmoothness(xintersect, yintersect, wavel):
    
#def KnifeEdges():

#def Cylinders():

#def ITUNLoS(): #10 MHz and above

#def ITULoS():

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

    mTR = (Ry-Ty)/(Rx-Tx)
    bTR = Ry - mTR*Rx
    ldy = 0

    for xcoord, ycoord in zip(Xcoords[1:(len(Xcoords)-1)],Ycoords[1:(len(Ycoords)-1)]):
        LoSy = mTR*xcoord + bTR
        if LoSy < ycoord:
            ldy = ycoord

    m1 = 0
    b1 = 0
    m2 = 0
    b2 = 0

    for xcoord, ycoord in zip(Xcoords[1:(len(Xcoords)-1)],Ycoords[1:(len(Ycoords)-1)]):     #!?
        mtemp1 = (ycoord-Ty)/(xcoord-Tx)
        mtemp2 = (Ry-ycoord)/(Rx-xcoord)
        print(mtemp1)
        print(mtemp2)

        if ldy > 0:
            if mtemp1 > m1:
                m1 = mtemp1
                b1 = Ty - m1*Tx

            if mtemp2 < m2:
                m2 = mtemp2
                b2 = Ry - m2*Rx
        else:
            if mtemp1 < m1:
                m1 = mtemp1
                b1 = Ty - m1*Tx

            if mtemp2 > m2:
                m2 = mtemp2
                b2 = Ry - m2*Rx

    Xpoint = (b2-b1)/(m1-m2)
    Ypoint = m1*Xpoint+b1
    #print('Ypoint:',Ypoint)
    #print("m1: ",m1)
    #print("m2: ",m2)
    #print("Lenth: ",len(Xcoords)) 
    #print(Xcoords[:(len(Xcoords))])#!?
    #plt.plot(Xcoords,Ycoords,'-')
    #plt.plot([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],'-')
    #plt.show()
    return FresnelKirchoff([Tx,Xpoint,Rx],[Ty,Ypoint,Ry],wavel)

def EpsteinPeterson(Xcoords,Ycoords,wavel):
    NumEdges = len(Xcoords) - 2
    L = 0

    for i in range(NumEdges):
        L = L + FresnelKirchoff([Xcoords[i],Xcoords[i+1],Xcoords[i+2]],[Ycoords[i],Ycoords[i+1],Ycoords[i+2]],wavel)

    return L

def Deygout(Xcoords,Ycoords,wavel):

    NumEdges = len(Xcoords) - 2
    FresnelParams = []

    for i in range(NumEdges):
        distance1, distance2, height = ObstacleValues([Xcoords[0],Xcoords[i+1],Xcoords[-1]],[Ycoords[0],Ycoords[i+1],Ycoords[-1]])
        v = height*math.sqrt(2/wavel*(1/(distance1)+1/(distance2)))
        FresnelParams.append(v)

    L = DeygoutLoss(Xcoords,Ycoords,wavel,FresnelParams)
    print(L)

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




def main():
    #the transmitter is at point zero on the distnace axis
    intlength = 60 #meter
    rheight = 50 #meter
    theight = 50 #meter
    frequency = 600000000 #Hz

    wavel = WaveLength(frequency)
    print(type(wavel))
    #print(wavel)
    distarr, heightarr = TerrainDivide("C:/Users/marko/Desktop/FYP/Book5.csv",intlength,1)
    print(type(distarr[5]))
    #xintersect, yintersect = FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel)

    #ObstacleSmoothness(xintersect, yintersect, wavel)

    #Jv = FresnelKirchoff(20,10000,5000,wavel)
    #A = ITUSingleRounded(20,10000,5000,wavel,15)

    #L = Bullington([0,7000,12000,22000,26000],[0,30,30,20,0],wavel)
    #L = EpsteinPeterson([0,7000,12000,22000,26000],[0,30,50,20,0],wavel)
    L = Deygout([0,7000,12000,22000,26000],[0,30,50,20,0],wavel)
    print('Loss: ',L)


if __name__ == '__main__':
    main()