#include <curand.h>
#include <curand_kernel.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <algorithm>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "checkers.hpp"
#pragma once

// http://www.cse.yorku.ca/~oz/marsaglia-rng.html, random number generators
#define znew (z=36969*(z&65535)+(z>>16))
#define wnew (w=18000*(w&65535)+(w>>16))
#define MWC ((znew<<16)+wnew )
#define CONG (jcong=69069*jcong+1234567)
#define SHR3 (jsr^=(jsr<<17), jsr^=(jsr>>13), jsr^=(jsr<<5))
#define KISS ((MWC^CONG)+SHR3)
#define RANDOMS_PER_THREAD 4 // uint32 needed as a KISS seed

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    goto Error;}} while(0)

#define BLOCKSIZE 512
#define MAX_BLOCK_NUMBER 32

// #define DEBUG // uncomment to see time needed for cudamalloc, curand, thrust::reduce

__global__ void simulateGameKernel(uint32_t white, uint32_t black, uint32_t promoted, uint8_t movesWithoutTake, bool whiteToPlay, bool* result, uint8_t * random, uint32_t numberSimulations)
{
    const uint32_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numberSimulations)
        return;

    uint32_t movesArray[MOVES_Q_SIZE];
    Queue<uint32_t> availbleMovesQ = Queue<uint32_t>(movesArray, MOVES_Q_SIZE);

    // seed for each thread
    uint32_t z = random[idx * RANDOMS_PER_THREAD], w = random[idx * RANDOMS_PER_THREAD + 1], 
        jsr = random[idx * RANDOMS_PER_THREAD + 2], jcong = random[idx * RANDOMS_PER_THREAD + 3];

    bool playerOnMove = true;
    if (!whiteToPlay)
    {
        SWAP(white, black);
        REVERSE32(white);
        REVERSE32(black);
        REVERSE32(promoted);
    }
    while (true)
    {
        if (movesWithoutTake > 40)
        {
            result[idx] = false;
            return;
        }
        simulateOne(white, black, promoted, movesWithoutTake, &availbleMovesQ);

        uint8_t length = availbleMovesQ.length() / 3;
        if (length == 0)
        {
            if (black == 0)
            {
                result[idx] = playerOnMove;
                return;
            }
            if (white == 0)
            {
                result[idx] = !playerOnMove;
                return;
            }
            result[idx] = false;
            return;
        }

        // custom fast random number generator
        length = KISS % length; 

        bool isCapture = movesArray[length * 3 + 1] != black;
        white = movesArray[length * 3];
        black = movesArray[length * 3 + 1];
        promoted = movesArray[length * 3 + 2];

        SWAP(white, black);
        REVERSE32(white);
        REVERSE32(black);
        REVERSE32(promoted);
        availbleMovesQ.clear();
        whiteToPlay = !whiteToPlay;
        playerOnMove = !playerOnMove;
        movesWithoutTake = !isCapture * (movesWithoutTake + 1);
    }
}

class PlayerGPU : public Player
{
private:
    static const uint32_t maxSimulations = BLOCKSIZE * MAX_BLOCK_NUMBER;
    cudaError_t SimlateCUDA(node* node)
    {
#ifdef DEBUG
        cudaEvent_t start, malloc, curand, thrustStart, thrustStop;
        cudaEventCreate(&start);
        cudaEventCreate(&malloc);
        cudaEventCreate(&curand);
        cudaEventCreate(&thrustStart);
        cudaEventCreate(&thrustStop);
        bool afterFirstLoop = false; // check functions in loop once
#endif // DEBUG

        const uint32_t simulationsPerCycle = std::min(maxSimulations, numberSimulations);
        const uint32_t requiredCycles = (numberSimulations - 1) / simulationsPerCycle + 1;
        const uint8_t numberBlocks = (simulationsPerCycle - 1) / BLOCKSIZE + 1;
        uint8_t* dev_random = 0;
        bool* dev_results = 0;
        curandGenerator_t gen;
        cudaError_t cudaStatus;
        
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

#ifdef DEBUG
        cudaEventRecord(start);
#endif // DEBUG

        cudaStatus = cudaMalloc((void**)&dev_random, simulationsPerCycle * RANDOMS_PER_THREAD * sizeof(uint8_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_results, simulationsPerCycle * sizeof(bool));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

#ifdef DEBUG
        cudaEventRecord(malloc);
#endif // DEBUG

        CURAND_CALL(curandCreateGenerator(&gen,
            CURAND_RNG_PSEUDO_DEFAULT));

        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
            time(NULL)));


        for (uint32_t k = 0; k < requiredCycles; k++)
        {
            CURAND_CALL(curandGenerate(gen, (uint32_t*)dev_random, simulationsPerCycle * RANDOMS_PER_THREAD * sizeof(uint8_t) / sizeof(uint32_t)));

#ifdef DEBUG
            if(!afterFirstLoop)
                cudaEventRecord(curand);           
#endif // DEBUG

            uint32_t simulationToRun;
            if (k == requiredCycles - 1)
            {
                simulationToRun = numberSimulations - maxSimulations * k;
            }
            else
            {
                simulationToRun = maxSimulations;               
            }
            simulateGameKernel << <numberBlocks, BLOCKSIZE >> > (node->whitePieces, node->blackPieces, node->promotedPieces,
                node->movesWithoutTake, node->whiteToPlay, dev_results, dev_random, simulationToRun);

            cudaStatus = cudaGetLastError();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "simulateGameKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
                goto Error;
            }

            cudaStatus = cudaDeviceSynchronize();
            if (cudaStatus != cudaSuccess) {
                fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
                goto Error;
            }

#ifdef DEBUG
            cudaEventRecord(thrustStart);
#endif // DEBUG

            int result = thrust::reduce(thrust::device, dev_results, dev_results + numberSimulations, 0);

#ifdef DEBUG
            cudaEventRecord(thrustStop);
            afterFirstLoop = true;
#endif // DEBUG
            node->gamesWon += result;
        }
    Error:
        cudaFree(dev_random);
        cudaFree(dev_results);
        CURAND_CALL(curandDestroyGenerator(gen));

#ifdef DEBUG
        cudaEventSynchronize(thrustStop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, malloc);
        std::cout << "Malloc: " << milliseconds  << " ms" << std::endl;
        cudaEventElapsedTime(&milliseconds, malloc, curand);
        std::cout << "cuRand: " << milliseconds  << " ms" << std::endl;
        cudaEventElapsedTime(&milliseconds, thrustStart, thrustStop);
        std::cout << "Thrust reduce: " << milliseconds  << " ms" << std::endl;

#endif // DEBUG

        return cudaStatus;
    }
public:
    PlayerGPU(bool isWhite, uint32_t numberSimulations) : Player(isWhite, numberSimulations) {}
	void Simulate(node* node) override
	{
        SimlateCUDA(node);
	}

};
