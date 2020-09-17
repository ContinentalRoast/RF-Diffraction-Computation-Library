from RFDiffraction import *



def main():
    #the transmitter is at point zero on the distnace axis
    intlength = 200000 #meter
    rheight = 30 #meter
    theight = 30 #meter
    frequency = 1000000000 #Hz

    wavel = WaveLength(frequency)
    distarr, heightarr = TerrainDivide("C:/Users/marko/Desktop/FYP/Book2.csv",intlength,1)

    bXcoords = distarr

    bYcoords = heightarr
    bYcoords[-1] = (rheight+heightarr[-1])
    bYcoords[0] = (theight+heightarr[0])


    h_aksent_ts, h_aksent_rs, X, Y1, Y2, FX, GY1, GY2, Lsph, Xa, Y1a, Y2a, FXa, GY1a, GY2a, Lspha, L1, L2, Lbs, Lba = DeltaBullingtonB(bXcoords/1000,bYcoords,wavel)
    




if __name__ == '__main__':
    main()