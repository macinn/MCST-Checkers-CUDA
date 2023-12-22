#include <stdio.h>
#include <iostream>
#include <string>
#include <sstream>
#include <conio.h>
#include <chrono>
#include "checkersGPU.cuh"

// Run modes
//#define BENCHMARK // Compare CPU and GPU capatibility
//#define DUEL // Make CPU and GPU play agains each other
//#define PLAY_AGAINST // Play against CPU or GPU

#define TIMELIMIT 100 // ms for simulation each move
// #define WAIT_FOR_INPUT // if defined user needs to press eneter to see next move
#define SIMULATIONS_GPU 512 * 8 // only valid for DUEL and PLAY_AGAINST
#define SIMULATIONS_CPU 128 // only valid for DUEL and PLAY_AGAINST


void LoadOption(std::string nazwaZmiennej, uint32_t* var);

int main()
{
    system("cls");


    // Check if cuda capable
    cudaError_t cudaStatus = cudaSetDevice(0);
    const bool noCudaDevice = cudaStatus != cudaSuccess;

    while(true)
    {
        std::cout << "MCST-Checkers-CUDA Skrzypczak Marcin" << std::endl;
        std::string input;
        bool askAgain;
        Player* white, * black;
        bool userWhite = false, userBlack = false;
        std::cout << "White:" << std::endl;
        std::cout << "1. Player" << std::endl;
        std::cout << "2. CPU" << std::endl;
        if(noCudaDevice)
            std::cout << "[No CUDA-capable GPU!] 3. GPU" << std::endl;
        else
            std::cout << "3. GPU" << std::endl;
        do
        {
            askAgain = false;
            std::cin >> input;
            switch (stoi(input))
            {
            case 1:
                userWhite = true;
                white = new PlayerCPU(true, 0);
                break;
            case 2:
                white = new PlayerCPU(true, SIMULATIONS_CPU);
                break;
            case 3:
                if (noCudaDevice)
                {
                    askAgain = true;
                    break;
                }
                white = new PlayerGPU(true, SIMULATIONS_GPU);
                break;
            default:
                askAgain = true;
                std::cout << "Invalid value!" << std::endl;
                break;
            }
        } while (askAgain);
        system("cls");

        std::cout << "Black:" << std::endl;
        std::cout << "1. Player" << std::endl;
        std::cout << "2. CPU" << std::endl;
        if (noCudaDevice)
            std::cout << "[No CUDA-capable GPU!] 3. GPU" << std::endl;
        else
            std::cout << "3. GPU" << std::endl;
        do
        {
            askAgain = false;
            std::cin >> input;
            switch (stoi(input))
            {
            case 1:
                userBlack = true;
                black = new PlayerCPU(false, 0);
                break;
            case 2:
                black = new PlayerCPU(false, SIMULATIONS_CPU);
                break;
            case 3:
                if (noCudaDevice)
                {
                    askAgain = true;
                    break;
                }
                black = new PlayerGPU(false, SIMULATIONS_GPU);
                break;
            default:
                askAgain = true;
                std::cout << "Invalid value!" << std::endl;
                break;
            }
        } while (askAgain);
        system("cls");
        white->timeLimit = TIMELIMIT;
        black->timeLimit = TIMELIMIT;

        if (!userBlack || !userWhite)
        {
            std::cout << "Do you want to set the bots' parameters? [Y/N]";
            std::cin >> input;
            if (input.compare("Y") == 0 || input.compare("y") == 0)
            {
                if (!userWhite)
                {
                    std::cout << "White:" << std::endl;
                    LoadOption("Time limit [ms]", &white->timeLimit);
                    LoadOption("The number of simulations from the chosen leaf:", &white->numberSimulations);
                }
                if (!userBlack)
                {
                    std::cout << "Black:" << std::endl;
                    LoadOption("Time limit [ms]", &black->timeLimit);
                    LoadOption("The number of simulations from the chosen leaf::", &black->numberSimulations);
                }
            }
        }
        // GAME LOOP
        bool gameEnded = false;
        while (!gameEnded)
        {
            system("cls");
            white->Print();
            node* move;
            gameEnded |= black->gameEnded();
            if (!gameEnded)
            {
                if (userWhite)
                {
                    uint8_t moveFrom, moveTo;
                    std::string inputFrom, inputTo;

                    std::cout << "Type your move as [from] [to] fx 'A4 B3'" << std::endl;
                    do
                    {
                        std::cin >> inputFrom >> inputTo;
                        moveFrom = (std::toupper(inputFrom[0]) - 'A') / 2 + (inputFrom[1] - '1') * 4;
                        moveTo = (std::toupper(inputTo[0]) - 'A') / 2 + (inputTo[1] - '1') * 4;
                    } while (!white->InputMove(moveFrom, moveTo));
                    black->InputMove(moveFrom, moveTo);
                }
                else
                {
                    move = white->FindNextMove();
                    if (!move)
                    {
                        gameEnded = true;
                        break;
                    }
                    else
                    {
                        node* moveCopy = new node(*move);
                        moveCopy->children.clear();
                        black->MakeMove(moveCopy);
                    }
                }
            }

            system("cls");
            white->Print();
            gameEnded |= white->gameEnded();
            if (userBlack)
            {
                uint8_t moveFrom, moveTo;
                std::string inputFrom, inputTo;

                std::cout << "Type your move as [from] [to] fx 'A4 B3'" << std::endl;
                do
                {
                    std::cin >> inputFrom >> inputTo;
                    moveFrom = (std::toupper(inputFrom[0]) - 'A') / 2 + (inputFrom[1] - '1') * 4;
                    moveTo = (std::toupper(inputTo[0]) - 'A') / 2 + (inputTo[1] - '1') * 4;
                } while (!black->InputMove(moveFrom, moveTo));
                white->InputMove(moveFrom, moveTo);
            }
            else
            {
                move = black->FindNextMove();
                if (!move)
                {
                    gameEnded = true;
                    break;
                }
                else
                {
                    node* moveCopy = new node(*move);
                    moveCopy->children.clear();
                    white->MakeMove(moveCopy);
                }
            }
        }

        if (white->root->whitePieces == 0 || black->root->whitePieces == 0)
        {
            std::cout << "Czarny wygral!" << std::endl;
        }
        else if (white->root->blackPieces == 0 || black->root->blackPieces == 0)
        {
            std::cout << "Bialy wygral!" << std::endl;
        }
        else if (white->root->movesWithoutTake > 40 || black->root->movesWithoutTake > 40)
        {
            std::cout << "Remis!" << std::endl;
        }
        else
        {
            std::cout << "Pat!" << std::endl;
        }
        delete white;
        delete black;
        std::cout << std::endl;
        std::cout << "Nacisnij ENTER aby zagrac ponownie!";
        getch();
        system("cls");
    }
// INSTRUKCJE PREPOCESORA
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
    bool playAsWhite = true;
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
        system("cls");
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
        playAsWhite = false; 
    }

    delete oponent;

#endif // PLAY_AGAINST
}

void LoadOption(std::string nazwaZmiennej, uint32_t* var)
{
    std::cout << nazwaZmiennej << " (Teraz: " << *var << ")" << std::endl;
    std::cout << "Wprowadz nowa wartosc lub 0 aby pominac: ";
    std::string input;
    std::cin >> input;
    if (!input.empty())
    {
        if(stoi(input) > 0)
            *var = stoi(input);
    }
}
