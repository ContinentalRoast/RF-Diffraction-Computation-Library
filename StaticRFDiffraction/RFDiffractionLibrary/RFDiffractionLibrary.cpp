// RFDiffractionLibrary.cpp
// compile with: cl /c /EHsc RFDiffractionLibrary.cpp
// post-build command: lib RFDiffractionLibrary.obj

#include "RFDiffractionLibrary.h"

namespace RFDiffractionLibrary
{
    double TerrainAnalysis::TerrainDivide(char TerrainFile[], double IntervalLength)
    {
        return 1;
    }

    double TerrainAnalysis::FreznelZoneClearance(double TransmitterHeight, double ReceiverHeight, double frequency, double TerrainProfile[][2], int DiffractionMethod)
    {
        return 1;
    }

    double TerrainAnalysis::ObstacleSmoothness(double Wavelength, double TerrainProfile[][2])
    {
        return 1;
    }

    double RFDiffraction::ITUNLoS(double Frequency, double PathLength, double RadiusEarth, double TransmitterHeight, double ReceiverHeight)
    {
        return 1;
    }


    double RFDiffraction::ITULoS(double Frequency, double PathLength, double RadiusEarth, double TransmitterHeight, double ReceiverHeight)
    {
        return 1;
    }


    double RFDiffraction::FresnelKirchoff(double TransmitterHeight, double ReceiverHeight, double DistnaceT, double DistanceR, double ObstacleHeight, double Frequency)
    {
        return 1;
    }


    double RFDiffraction::ITUSingleRounded(double TransmitterHeight, double ReceiverHeight, double DistnaceT, double DistanceR, double ObstacleHeight, double Frequency, double Radius)
    {
        return 1;
    }


    double RFDiffraction::ITUTwoEdge(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::Bullington(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::EpsteinPeterson(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::Deygout(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::Vogler(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::Giovanelli(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::ITUDeltaBullington(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[])
    {
        return 1;
    }


    double RFDiffraction::ITUMultipleCylinders(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[], double Radii)
    {
        return 1;
    }


    double RFDiffraction::DeygoutRounded(double TransmitterHeight, double ReceiverHeight, double Frequency, double ObstacleHeights[], double ObstacleDistances[], double Radii[])
    {
        return 1;
    }
}
