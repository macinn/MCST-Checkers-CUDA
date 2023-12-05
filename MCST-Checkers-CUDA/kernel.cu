
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <chrono>

#include "checkersGPU.cuh"

int main()
{
    PlayerGPU whiteGPU = PlayerGPU(true);
    whiteGPU.Print();
    whiteGPU.Simulate(whiteGPU.root);
    std::cout << whiteGPU.root->gamesWon;
    return 0;
}

