#include <iostream>
#include <math.h>
#include "cuda_runtime.h"

#pragma once

// fast bit reversal for 32bit long digit
#define REVERSE32(b)    b = (b & 0xFFFF0000) >> 16 | (b & 0x0000FFFF) << 16; \
                        b = (b & 0xFF00FF00) >> 8 | (b & 0x00FF00FF) << 8; \
                        b = (b & 0xF0F0F0F0) >> 4 | (b & 0x0F0F0F0F) << 4; \
                        b = (b & 0xCCCCCCCC) >> 2 | (b & 0x33333333) << 2; \
                        b = (b & 0xAAAAAAAA) >> 1 | (b & 0x55555555) << 1;

// get bit value at given postion
#define BIT(a, b)   (bool)(((a) & (1 << (b))) >> (b))

// set bit value at given postion
#define SET_BIT(a, b, v)    a &= (~(1 << (b))); \
                            a |= ((v) << (b));

// swap two values, without aditional memory
#define SWAP(a, b)  a = a ^ b; \
                    b = a ^ b; \
                    a = a ^ b;

// calculate position after capture, from given indexes on 32 bit board representation
#define POSTION_AFTER_CAPTURE(positionBefore, positionOponent, postionAfter) \
    int8_t moveDiff = (positionOponent) - (positionBefore); \
    postionAfter = positionOponent \
        + (moveDiff > 0) * ( \
        + 3 * (moveDiff == 4 && BIT(moveLeftUpAvailble, positionOponent)) \
        + 4 * (moveDiff != 4) \
        + 5 * (moveDiff == 4 && BIT(moveRightUpAvailble, positionOponent))) \
        + (moveDiff <= 0) * ( \
        - 3 * (moveDiff == -4 && BIT(moveRightDownAvailble, positionOponent)) \
        - 4 * (moveDiff != -4) \
        - 5 * (moveDiff == -4 && BIT(moveLeftDownAvailble, positionOponent))); \

// Wyznaczone eksperymantalnie bezpieczne wartoœci
// size of Q for possible moves, stored as 3 consecutive uint32_t
constexpr uint8_t MOVES_Q_SIZE = 32 * 3;
// size of Q for possible captures
constexpr uint8_t CAPTURES_Q_SIZE = 32;

constexpr double EXPLORATION_CONST_SQARED = 2;

// queue interface, read from empty queue and write over queue size is undefined
template <class T>
class Queue {
public:
    uint8_t first;  // read  there
    uint8_t last;   // wirte there
    uint8_t size;
    T* Q;

    __host__ __device__ Queue(T* Q, uint8_t size)
    {
        this->Q = Q;
        this->size = size;
        first = last = 0;
    }
    __host__ __device__ void push(const T v)
    {
        Q[last] = v;
        last = (last + 1) % size;
    }
    __host__ __device__ void pop()
    {
        first = (first + 1) % size;
    }
    __host__ __device__ T front()
    {
        return Q[first];
    }
    __host__ __device__ bool empty()
    {
        return last == first;
    }
    __host__ __device__ uint8_t length()
    {
        return last - first + (last < first) * size;
    }
    __host__ __device__ void clear() {
        first = last = 0;
    }
};

// least significant bit, that is set, of given number
__host__ __device__ uint8_t firstBit(uint32_t x)
{
    for (uint8_t i = 0; i < 32; i++)
        if (x & (1 << i))
            return i;
    return 32;
}

// print board on stdout
void printBoard(uint32_t whitePieces, uint32_t blackPieces, uint32_t promotedPieces)
{
    std::string columns = " | A| B| C| D| E| F| G| H| ";
    std::string separator = "-+--+--+--+--+--+--+--+--+-";
    std::cout << std::endl;
    std::cout << '\t' << columns << std::endl;
    std::cout << '\t' << separator << std::endl;

    for (int i = 7; i >= 0; i--)
    {
        std::cout << '\t' << (char)(i + 49) << '|'; // (char)(i + '1')

        for (int j = 0; j < 4; j++)
        {
            if (i % 2)
                std::cout << "  |";
            if (whitePieces & (1 << (i * 4 + j)))
            {
                std::cout << 'W';
                if (promotedPieces & (1 << (i * 4 + j)))
                    std::cout << 'D';
                else
                    std::cout << ' ';
            }
            else if (blackPieces & (1 << (i * 4 + j)))
            {
                std::cout << 'B';
                if (promotedPieces & (1 << (i * 4 + j)))
                    std::cout << 'D';
                else
                    std::cout << ' ';
            }
            else
                std::cout << "  ";
            if (i % 2)
                std::cout << '|';
            else
                std::cout << "|  |";
        }

        std::cout << (char)(i + 49) << std::endl;
        std::cout << '\t' << separator << std::endl;
    }

    std::cout << '\t' << columns << std::endl;
    std::cout << '\t' << std::endl;
}

// print board as binary on stdout
void printBoardBinary(uint32_t pieces)
{
    for (int i = 7; i >= 0; i--)
    {
        std::cout << BIT(pieces, i * 4) << BIT(pieces, i * 4 + 1)
            << BIT(pieces, i * 4 + 2) << BIT(pieces, i * 4 + 3) << std::endl;
    }
}