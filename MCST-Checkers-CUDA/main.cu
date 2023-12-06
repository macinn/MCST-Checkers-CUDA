
#include <stdio.h>
#include <iostream>
#include <math.h>
#include <string>
#include <chrono>

#include "checkersGPU.cuh"

// Run modes
#define BENCHMARK // Compare CPU and GPU capatibility
//#define DUEL // Make CPU and GPU play agains each other
//#define PLAY_AGAINST // Play against CPU or GPU

#define TIMELIMIT 1000 // ms for simulation each move
// #define WAIT_FOR_INPUT // if defined user needs to press eneter to see next move
#define SIMULATIONS_GPU 1024 * 16 // only valid for DUEL and PLAY_AGAINST
#define SIMULATIONS_CPU 1024 * 4 // only valid for DUEL and PLAY_AGAINST


int main()
{
    std::cout << "MCST-Checkers-CUDA Skrzypczak Marcin" << std::endl;
#ifdef BENCHMARK
    const uint32_t numberSimulations = 1024 * 32;
    PlayerGPU whiteGPU = PlayerGPU(true, numberSimulations);
    PlayerCPU whiteCPU = PlayerCPU(true, numberSimulations);

    // warmup
    for(uint8_t k = 0; k < 5; k++)
        whiteGPU.Simulate(whiteGPU.root);

    auto start = std::chrono::system_clock::now();
    whiteGPU.Simulate(whiteGPU.root);
    auto end = std::chrono::system_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Benchmark " << numberSimulations << " simulations" << std::endl;
    std::cout << "GPU " << elapsed.count() << " ms" << std::endl; 

    // warmup
    for (uint8_t k = 0; k < 5; k++)
        whiteCPU.Simulate(whiteCPU.root);

    start = std::chrono::system_clock::now();
    whiteCPU.Simulate(whiteCPU.root);
    end = std::chrono::system_clock::now();
    elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "CPU " << elapsed.count() << " ms" << std::endl;

#endif // BENCHMARK

#ifdef DUEL
        const bool GPUasWhite = true;
        PlayerGPU GPU = PlayerGPU(GPUasWhite);
        PlayerCPU CPU = PlayerCPU(!GPUasWhite, 1024 * 4);
        GPU.timeLimit = TIMELIMIT;
        CPU.timeLimit = TIMELIMIT;

        Player* white, * black;
        if (GPUasWhite)
        {
            white = &GPU;
            black = &CPU;
        }
        else
        {
            white = &CPU;
            black = &GPU;
        }

        while (true)
        {
            white->Print();

            node* move = white->FindNextMove();
            if (!move) 
                break;
            node* moveCopy = new node();

            moveCopy->whitePieces = move->whitePieces;
            moveCopy->blackPieces = move->blackPieces;
            moveCopy->promotedPieces = move->promotedPieces;
            moveCopy->movesWithoutTake = move->movesWithoutTake;
            moveCopy->whiteToPlay = move->whiteToPlay;

#ifdef WAIT_FOR_INPUT
            std::cout << "Press ENTER" << std::endl;
            std::cin.get();
#endif // WAIT_FOR_INPUT
            system("cls");
            white->Print();

            black->MakeMove(moveCopy);

            move = black->FindNextMove();
            if (!move) 
                break;
            moveCopy = new node();

            moveCopy->whitePieces = move->whitePieces;
            moveCopy->blackPieces = move->blackPieces;
            moveCopy->promotedPieces = move->promotedPieces;
            moveCopy->movesWithoutTake = move->movesWithoutTake;
            moveCopy->whiteToPlay = move->whiteToPlay;

            white->MakeMove(moveCopy);

#ifdef WAIT_FOR_INPUT
            std::cout << "Press ENTER" << std::endl;
            std::cin.get();
#endif // WAIT_FOR_INPUT
            system("cls");
        }
   
#endif // DUEL

#ifdef PLAY_AGAINST
    const bool playAsWhite = true;
    const bool playAgainstCPU = false;
    Player* oponent;
    if (playAgainstCPU)
    {
        oponent = new PlayerCPU(!playAsWhite, SIMULATIONS_CPU);
    }
    else
    {
        oponent = new PlayerGPU(!playAsWhite, SIMULATIONS_GPU);
    }

    while (true)
    {
        uint8_t moveFrom, moveTo;
        std::string inputFrom, inputTo;
        if (!playAsWhite)
            if (!oponent->FindNextMove())
                return;
#ifdef WAIT_FOR_INPUT
        std::cout << "Press ENTER" << std::endl;
        std::cin.get();
#endif // WAIT_FOR_INPUT
        oponent->Print();
        std::cout << "Type you move as [from] [to] fx 'A4 B3'" << std::endl;
        do
        {
            std::cin >> inputFrom >> inputTo;
            moveFrom = (std::toupper(inputFrom[0]) - 'A') / 2 + (inputFrom[1] - '1') * 4;
            moveTo = (std::toupper(inputTo[0]) - 'A') / 2 + (inputTo[1] - '1') * 4;
        } while (!oponent->InputMove(moveFrom, moveTo));
        system("cls");
        oponent->Print();
    }

    delete oponent;

#endif // PLAY_AGAINST
}

