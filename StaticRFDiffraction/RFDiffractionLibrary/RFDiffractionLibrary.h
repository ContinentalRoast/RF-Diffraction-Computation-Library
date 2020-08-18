// RFDiffractionLibrary.h
#pragma once


namespace RFDiffractionLibrary
{
    class TerrainAnalysis
    {

    public:
        //fcn 1.1
        static double TerrainDivide(char TerrainFile[], double IntervalLength);

        
        //fcn 1.2
        static double FreznelZoneClearance(double TransmitterHeight, double ReceiverHeight, double frequency, double TerrainProfile[][2], int DiffractionMethod);

        
        //fcn 1.3
        static double ObstacleSmoothness(double Wavelength, double TerrainProfile[][2]);
    };

    class RFDiffraction
    {

    public:
        //fcn 2.1
        static double ITUNLoS(double Frequency, double PathLength, double RadiusEarth, double TransmitterHeight, double ReceiverHeight);


        //fcn 2.2
        static double ITULoS(double Frequency, double PathLength, double RadiusEarth, double TransmitterHeight, double ReceiverHeight);


        //fcn 2.3
        static double FresnelKirchoff(double TransmitterHeight, double ReceiverHeight, double DistnaceT, double DistanceR, double ObstacleHeight, double Frequency);


        //fcn 2.4
        static double ITUSingleRounded(double TransmitterHeight, double ReceiverHeight, double DistnaceT, double DistanceR, double ObstacleHeight, double Frequency, double Radius);


        //fcn 2.5
        static double ITUTwoEdge(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.6
        static double Bullington(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.7
        static double EpsteinPeterson(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.8
        static double Deygout(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.9
        static double Vogler(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.10
        static double Giovanelli(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.11
        static double ITUDeltaBullington(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[]);


        //fcn 2.12
        static double ITUMultipleCylinders(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[], double Radii);


        //fcn 2.13
        static double DeygoutRounded(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[], double Radii[]);
    };
}