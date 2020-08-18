// compile with: cl /EHsc RFDifrractionClient.cpp /link RFDiffractionLibrary.lib

#include <iostream>
#include "RFDiffractionLibrary.h"
#include <string>

int main()
{
    char TerrainFile[] = "C:/Users/marko/Desktop/FYP/Book2.csv";
    double IntervalLength = 50;
    double TransmitterHeight = 20;
    double ReceiverHeight = 70;
    double frequency = 6000000;
    int DiffractionMethod = 0;

    std::cout << TerrainFile << std::endl;

    return 0;
}