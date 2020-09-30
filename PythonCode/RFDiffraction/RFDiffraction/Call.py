from RFDiffraction import *



def main():
    intlength = 200000 #meter
    rheight = 50 #meter
    theight = 50 #meter

    f = 1000000000 #Hz
    wavel = WaveLength(f)

    distarr, heightarr = TerrainDivide("C:/Users/marko/Desktop/FYP/Book2.csv",intlength,1)
    rheight = 50
    theight = 50

    xintersect, yintersect, Tdist, Theight, Rdist, Rheight = FresnelZoneClearance(distarr,heightarr,rheight,theight,wavel,plotZone = 1)

if __name__ == '__main__':
    main()





    #---------------------------------------------------------------------------------------------------------------------------
    #The following section of commented code was used to test ellipse obtained from the matplotlib.patches Ellipse function
    #against the ellipse function as described by Fresnel

    #cx = distarr[len(distarr)//2-1]
    #cy = m*cx+b
    

    #R = ((wavel*(length/2)**2)/(length))**(1/2)

    #angle = np.arctan((Rheight-Theight)/Rdist)*180/np.pi
    

    #ells = matplotlib.patches.Ellipse((cx, cy), length, R*2, angle)

    #a = plt.subplot()

    #a.add_artist(ells)
    #---------------------------------------------------------------------------------------------------------------------------





    #filename = filedialog.askopenfilename(initialdir =  "/", title = "Select A File", filetype =(("csv files","*.csv"),("all files","*.*"))) #sit in controller function