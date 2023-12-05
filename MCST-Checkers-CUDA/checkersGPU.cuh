#include <curand.h>
#include <curand_kernel.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "checkers.hpp"
#pragma once

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    goto Error;}} while(0)

#define BLOCKSIZE 512
#define BLOCKSNUMBER 32
#define RANDOM_SIZE3 31
#define WARP 32

struct is_true
{
    __host__ __device__
        bool operator()(bool& x)
    {
        return x;
    }
};

__global__ void simulateGameKernel(uint32_t white, uint32_t black, uint32_t promoted, uint8_t movesWithoutTake, bool whiteToPlay, bool* result, uint8_t* random)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int lowwerBound = (idx / 32) * 32;    // inclusive
    uint8_t k = idx - lowwerBound;

    uint32_t movesArray[MOVES_Q_SIZE];
    Queue<uint32_t> availbleMovesQ = Queue<uint32_t>(movesArray, MOVES_Q_SIZE);

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
        length = random[k + lowwerBound] % length; // pregenerated random

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
        k++;
        k = k - (k == WARP) * 32;
        lowwerBound += (k == WARP) * BLOCKSIZE * BLOCKSNUMBER;
        lowwerBound = lowwerBound < (BLOCKSIZE * BLOCKSNUMBER * RANDOM_SIZE3);
    }
}

class PlayerGPU : public Player
{
private:
    cudaError_t SimlateCUDA(node* node)
    {
        uint8_t* dev_random = 0;
        bool* dev_results = 0;
        curandGenerator_t gen;
        cudaError_t cudaStatus;

        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_random, BLOCKSIZE * BLOCKSNUMBER * RANDOM_SIZE3 * sizeof(uint8_t));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        cudaStatus = cudaMalloc((void**)&dev_results, BLOCKSIZE * BLOCKSNUMBER * sizeof(bool));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!");
            goto Error;
        }

        CURAND_CALL(curandCreateGenerator(&gen,
            CURAND_RNG_PSEUDO_DEFAULT));

        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,
            time(NULL)));

        CURAND_CALL(curandGenerate(gen, (uint32_t*)dev_random, BLOCKSIZE * BLOCKSNUMBER * RANDOM_SIZE3 * sizeof(uint8_t) / sizeof(uint32_t)));

        simulateGameKernel << <BLOCKSNUMBER, BLOCKSIZE >> > (node->whitePieces, node->blackPieces, node->promotedPieces,
            node->movesWithoutTake, node->whiteToPlay, dev_results, dev_random);

        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            goto Error;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
            goto Error;
        }

        int result = thrust::count_if(thrust::device, dev_results, dev_results+ BLOCKSIZE * BLOCKSNUMBER, is_true());
        //int result = thrust::count(dev_results, dev_results + 10, 0);
        node->gamesWon += result;
    Error:
        cudaFree(dev_random);
        cudaFree(dev_results);
        CURAND_CALL(curandDestroyGenerator(gen));
        return cudaStatus;
    }
public:
    PlayerGPU(bool whiteToPlay) :Player(whiteToPlay) {}
	void Simulate(node* node) override
	{
        SimlateCUDA(node);
	}

};

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    //addKernel << <1, size >> > (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}