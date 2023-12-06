#include <vector>
#include "utils.hpp"
#pragma once

#define APPEND_NEW_CHILD(parent, white, black, promoted, isCapture) \
        struct node* newChild = new struct node(); \
        newChild->whitePieces = white; \
        newChild->blackPieces = black; \
        newChild->promotedPieces = promoted; \
        newChild->parentNode = parent; \
        newChild->whiteToPlay = !parent->whiteToPlay; \
        newChild->movesWithoutTake = !isCapture * (parent->movesWithoutTake + 1); \
        parent->children.push_back(newChild);

#define MOVE(from, to, white, promoted)  \
        SET_BIT(white, to, true);               \
        SET_BIT(white, from, false); \
        SET_BIT(promoted, to, BIT(promoted, from));     \
        SET_BIT(promoted, from, false);

// MCT node
typedef struct node {
    // tablica 4 x 8 tylko dost�pne pola
    uint32_t whitePieces = 0;
    // tablica 4 x 8 tylko dost�pne pola
    uint32_t blackPieces = 0;
    // tablica 4 x 8 tylko dost�pne pola
    uint32_t promotedPieces = 0;

    node* parentNode;
    std::vector<node*> children;

    bool whiteToPlay = true;
    uint8_t movesWithoutTake = 0;
    uint64_t gamesPlayed = 0;
    uint64_t gamesWon = 0;
} node;

// recursive node deletion
void deleteNode(node* node)
{
    if (node == nullptr) return;
    for (struct node* child : node->children)
    {
        deleteNode(child);
    }
    delete node;
}

// popluates given queue with possible moves
__host__ __device__ void simulateOne(uint32_t player, uint32_t oponent, uint32_t promoted, uint8_t movesWithoutTake, Queue<uint32_t>* availbleMovesQ)
{
    uint32_t newWhite = player;
    uint32_t newBlack = oponent;
    uint32_t newPromoted = promoted;

    uint32_t playerDoublePawns = newWhite & newPromoted;

    // bitmasks describing which pawns can move additionally to standard +4, -4
    uint32_t moveRightUpAvailble = 0x00707070;    // << 5, +5
    uint32_t moveLeftUpAvailble = 0x0E0E0E0E;     // << 3, +3
    uint32_t moveRightDownAvailble = 0x70707070;  // >> 3, -3
    uint32_t moveLeftDownAvailble = 0x0E0E0E00;   // >> 5, -5

    uint32_t freeTiles = ~(newBlack | newWhite);

    uint32_t availbleMovesUp = newWhite << 4
        | (newWhite & moveRightUpAvailble) << 5
        | (newWhite & moveLeftUpAvailble) << 3;

    uint32_t availbleMovesDown = playerDoublePawns >> 4
        | (playerDoublePawns & moveLeftDownAvailble) >> 5
        | (playerDoublePawns & moveRightDownAvailble) >> 3;

    uint32_t oponentPiecesCapturableUp = (freeTiles >> 4 | (freeTiles & moveLeftDownAvailble) >> 5 | (freeTiles & moveRightDownAvailble) >> 3) & newBlack & 0x0E7E'7E70;
    uint32_t oponentPiecesCapturableDown = (freeTiles << 4 & (freeTiles & moveRightUpAvailble) << 5 | (freeTiles & moveLeftUpAvailble) << 3) & newBlack & 0x0E7E'7E70;

    uint32_t availbleMovesNoCapWhite = (availbleMovesUp | availbleMovesDown) & freeTiles;
    uint32_t availbleMovesCapWhite = availbleMovesUp & oponentPiecesCapturableUp
        | availbleMovesDown & oponentPiecesCapturableDown;

    bool anyTakeAvailble = false;

    // pieces that moved due to capture
    uint32_t boardAfterCapArray[CAPTURES_Q_SIZE];
    Queue<uint32_t> captureQ = Queue<uint32_t>(boardAfterCapArray, CAPTURES_Q_SIZE);

    if (availbleMovesCapWhite > 0)
        for (int8_t i = 4; i < 28; i++)
        {
            if (!BIT(availbleMovesCapWhite, i)) continue;
            int8_t positionAfter = 0;
            uint32_t boardAfterCap = 0;
            SET_BIT(boardAfterCap, i, true);

            {
                if (BIT(playerDoublePawns, i + 4))
                {
                    POSTION_AFTER_CAPTURE(i + 4, i, positionAfter);

                    if (positionAfter >= 0 && BIT(freeTiles, positionAfter))
                    {
                        SET_BIT(boardAfterCap, i + 4, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        boardAfterCap = 0;
                        SET_BIT(boardAfterCap, i, true);
                    }
                }
                if (BIT(playerDoublePawns & moveLeftDownAvailble, i + 5))
                {
                    POSTION_AFTER_CAPTURE(i + 5, i, positionAfter);

                    if (positionAfter >= 0 && BIT(freeTiles, positionAfter))
                    {
                        SET_BIT(boardAfterCap, i + 5, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        boardAfterCap = 0;
                        SET_BIT(boardAfterCap, i, true);
                    }
                }
                if (BIT(playerDoublePawns & moveRightDownAvailble, i + 3))
                {
                    POSTION_AFTER_CAPTURE(i + 3, i, positionAfter);

                    if (positionAfter >= 0 && BIT(freeTiles, positionAfter))
                    {
                        SET_BIT(boardAfterCap, i + 3, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        boardAfterCap = 0;
                        SET_BIT(boardAfterCap, i, true);
                    }
                }
            }
            {
                POSTION_AFTER_CAPTURE(i - 4, i, positionAfter);
                if (BIT(newWhite, i - 4) && BIT(freeTiles, positionAfter))
                {
                    SET_BIT(boardAfterCap, i - 4, true);
                    SET_BIT(boardAfterCap, positionAfter, true);
                    captureQ.push(boardAfterCap);
                    boardAfterCap = 0;
                    SET_BIT(boardAfterCap, i, true);
                }
                if (BIT(moveRightDownAvailble, i) && BIT(newWhite, i - 3))
                {
                    POSTION_AFTER_CAPTURE(i - 3, i, positionAfter);
                    if (BIT(freeTiles, positionAfter))
                    {
                        SET_BIT(boardAfterCap, i - 3, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        boardAfterCap = 0;
                        SET_BIT(boardAfterCap, i, true);
                    }
                }
                if (BIT(moveLeftDownAvailble, i) && BIT(newWhite, i - 5))
                {
                    POSTION_AFTER_CAPTURE(i - 5, i, positionAfter);
                    if (BIT(freeTiles, positionAfter))
                    {
                        SET_BIT(boardAfterCap, i - 5, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        // boardAfterCap = 0;
                        // SET_BIT(boardAfterCap, i, true);
                    }
                }
            }
        }

    while (!captureQ.empty())
    {
        uint32_t captureData = captureQ.front();
        captureQ.pop();
        uint32_t currentOponent = oponent & ~captureData;
        uint8_t currentPos = firstBit(captureData & freeTiles);
        bool isQueen = BIT(promoted, currentPos);
        bool nextCapturePossible = false;
        int8_t positionAfter = 0;
        uint32_t boardAfterCap = 0;

        {
            if (isQueen)
            {
                POSTION_AFTER_CAPTURE(currentPos, currentPos - 4, positionAfter);
                if (positionAfter >= 0 && BIT(currentOponent, currentPos - 4) && BIT(freeTiles, positionAfter))
                {
                    boardAfterCap = captureData;
                    SET_BIT(boardAfterCap, currentPos, false);
                    SET_BIT(boardAfterCap, currentPos - 4, true);
                    SET_BIT(boardAfterCap, positionAfter, true);
                    captureQ.push(boardAfterCap);
                    nextCapturePossible = true;
                }
                if (BIT(moveLeftDownAvailble, currentPos))
                {
                    POSTION_AFTER_CAPTURE(currentPos, currentPos - 5, positionAfter);
                    if (positionAfter >= 0 && BIT(currentOponent, currentPos - 5) && BIT(freeTiles, positionAfter))
                    {
                        boardAfterCap = captureData;
                        SET_BIT(boardAfterCap, currentPos, false);
                        SET_BIT(boardAfterCap, currentPos - 5, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        nextCapturePossible = true;
                    }
                }
                if (BIT(moveRightDownAvailble, currentPos))
                {
                    POSTION_AFTER_CAPTURE(currentPos, currentPos - 3, positionAfter);
                    if (positionAfter >= 0 && BIT(currentOponent, currentPos - 3) && BIT(freeTiles, positionAfter))
                    {
                        boardAfterCap = captureData;
                        SET_BIT(boardAfterCap, currentPos, false);
                        SET_BIT(boardAfterCap, currentPos - 3, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        nextCapturePossible = true;
                    }
                }
            }
            {
                POSTION_AFTER_CAPTURE(currentPos, currentPos + 4, positionAfter);
                if (positionAfter < 32 && BIT(currentOponent, currentPos + 4) && BIT(freeTiles, positionAfter))
                {
                    boardAfterCap = captureData;
                    SET_BIT(boardAfterCap, currentPos, false);
                    SET_BIT(boardAfterCap, currentPos + 4, true);
                    SET_BIT(boardAfterCap, positionAfter, true);
                    captureQ.push(boardAfterCap);
                    nextCapturePossible = true;
                }
                if (BIT(moveLeftUpAvailble, currentPos))
                {
                    POSTION_AFTER_CAPTURE(currentPos, currentPos + 3, positionAfter);
                    if (positionAfter < 32 && BIT(currentOponent, currentPos + 3) && BIT(freeTiles, positionAfter))
                    {
                        boardAfterCap = captureData;
                        SET_BIT(boardAfterCap, currentPos, false);
                        SET_BIT(boardAfterCap, currentPos + 3, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        nextCapturePossible = true;
                    }
                }
                if (BIT(moveRightUpAvailble, currentPos))
                {
                    POSTION_AFTER_CAPTURE(currentPos, currentPos + 5, positionAfter);
                    if (positionAfter < 32 && BIT(currentOponent, currentPos + 5) && BIT(freeTiles, positionAfter))
                    {
                        boardAfterCap = captureData;
                        SET_BIT(boardAfterCap, currentPos, false);
                        SET_BIT(boardAfterCap, currentPos + 5, true);
                        SET_BIT(boardAfterCap, positionAfter, true);
                        captureQ.push(boardAfterCap);
                        nextCapturePossible = true;
                    }
                }
            }
        }

        if (!nextCapturePossible)
        {
            uint32_t currentPlayer = player;
            uint32_t currentPromoted = promoted;
            uint8_t startingPos = firstBit(player & captureData);
            SET_BIT(currentPlayer, currentPos, true);
            SET_BIT(currentPromoted, currentPos, BIT(promoted, startingPos) || currentPos > 27);

            SET_BIT(currentPlayer, startingPos, false);
            SET_BIT(currentPromoted, startingPos, false);

            currentPromoted = currentPromoted & (currentPlayer | currentOponent);

            availbleMovesQ->push(currentPlayer);
            availbleMovesQ->push(currentOponent);
            availbleMovesQ->push(currentPromoted);
        }
    }

    anyTakeAvailble = !availbleMovesQ->empty();

    if (!anyTakeAvailble && availbleMovesNoCapWhite > 0)
        for (uint8_t i = 0; i < 32; i++)
        {
            if (!BIT(availbleMovesNoCapWhite, i)) continue;

            SET_BIT(newWhite, i, true);

            if (BIT(newWhite, i - 4))
            {
                bool isPromoted = BIT(newPromoted, i - 4);
                SET_BIT(newWhite, i - 4, false);
                SET_BIT(newPromoted, i - 4, false);
                SET_BIT(newPromoted, i, isPromoted || i > 27);

                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);

                SET_BIT(newPromoted, i, false);
                SET_BIT(newPromoted, i - 4, isPromoted);
                SET_BIT(newWhite, i - 4, true);
            }
            if (BIT(newWhite & moveLeftUpAvailble, i - 3))
            {
                bool isPromoted = BIT(newPromoted, i - 3);
                SET_BIT(newWhite, i - 3, false);
                SET_BIT(newPromoted, i, isPromoted || i > 27);
                SET_BIT(newPromoted, i - 3, false);
                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);
                SET_BIT(newPromoted, i, false);
                SET_BIT(newPromoted, i - 3, isPromoted);
                SET_BIT(newWhite, i - 3, true);
            }
            if (BIT(newWhite & moveRightUpAvailble, i - 5))
            {
                bool isPromoted = BIT(newPromoted, i - 5);
                SET_BIT(newWhite, i - 5, false);
                SET_BIT(newPromoted, i - 5, false);
                SET_BIT(newPromoted, i, isPromoted || i > 27);
                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);
                SET_BIT(newPromoted, i, false);
                SET_BIT(newPromoted, i - 5, isPromoted);
                SET_BIT(newWhite, i - 5, true);
            }

            if (BIT(newWhite & newPromoted, i + 4))
            {
                SET_BIT(newWhite, i + 4, false);
                SET_BIT(newPromoted, i + 4, false);
                SET_BIT(newPromoted, i, true);
                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);
                SET_BIT(newPromoted, i, false);
                SET_BIT(newWhite, i + 4, true);
                SET_BIT(newPromoted, i + 4, true);
            }
            if (BIT(newWhite & newPromoted & moveLeftDownAvailble, i + 5))
            {
                SET_BIT(newWhite, i + 5, false);
                SET_BIT(newPromoted, i + 5, false);
                SET_BIT(newPromoted, i, true);
                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);
                SET_BIT(newPromoted, i, false);
                SET_BIT(newWhite, i + 5, true);
                SET_BIT(newPromoted, i + 5, true);
            }
            if (BIT(newWhite & newPromoted & moveRightDownAvailble, i + 3))
            {
                SET_BIT(newWhite, i + 3, false);
                SET_BIT(newPromoted, i + 3, false);
                SET_BIT(newPromoted, i, true);
                availbleMovesQ->push(newWhite);
                availbleMovesQ->push(newBlack);
                availbleMovesQ->push(newPromoted);
                SET_BIT(newPromoted, i, false);
                SET_BIT(newWhite, i + 3, true);
                SET_BIT(newPromoted, i + 3, true);
            }

            SET_BIT(newWhite, i, false);
        }
}

// simulare whole game, return true if player won
bool simulateTillEnd(uint32_t white, uint32_t black, uint32_t promoted, uint8_t movesWithoutTake, bool whiteOnMove)
{
    uint32_t movesArray[MOVES_Q_SIZE];
    Queue<uint32_t> availbleMovesQ = Queue<uint32_t>(movesArray, MOVES_Q_SIZE);
    bool playerOnMove = true;
    if (!whiteOnMove)
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
            return false;
        }
        simulateOne(white, black, promoted, movesWithoutTake, &availbleMovesQ);

        uint8_t length = availbleMovesQ.length() / 3;
        if (length == 0)
        {
            if (black == 0)
                return playerOnMove;
            if (white == 0)
                return !playerOnMove;
            return !playerOnMove;
        }
        length = rand() % length;

        bool isCapture = movesArray[length * 3 + 1] != black;
        white = movesArray[length * 3];
        black = movesArray[length * 3 + 1];
        promoted = movesArray[length * 3 + 2];
        // printBoard(white, black, promoted, whiteOnMove);

        SWAP(white, black);
        REVERSE32(white);
        REVERSE32(black);
        REVERSE32(promoted);
        availbleMovesQ.clear();
        whiteOnMove = !whiteOnMove;
        playerOnMove = !playerOnMove;
        movesWithoutTake = !isCapture * (movesWithoutTake + 1);
    }
}

// popluates node->children 
void generateChildren(node* node)
{
    uint32_t movesArray[MOVES_Q_SIZE];
    Queue<uint32_t> availbleMovesQ = Queue<uint32_t>(movesArray, MOVES_Q_SIZE);
    uint32_t white = node->whitePieces, black = node->blackPieces, promoted = node->promotedPieces;
    bool swap = !node->whiteToPlay;
    if (swap)
    {
        SWAP(white, black);
        REVERSE32(white);
        REVERSE32(black);
        REVERSE32(promoted);
    }

    simulateOne(white, black, promoted, node->movesWithoutTake, &availbleMovesQ);

    for (uint8_t i = 0; i < availbleMovesQ.length() / 3; i++)
    {
        bool isCapture = (movesArray[i * 3 + 1] ^ node->blackPieces) != 0;

        white = movesArray[i * 3];
        black = movesArray[i * 3 + 1];
        promoted = movesArray[i * 3 + 2];

        if (swap)
        {
            SWAP(white, black);
            REVERSE32(white);
            REVERSE32(black);
            REVERSE32(promoted);
        }
        APPEND_NEW_CHILD(node, white, black, promoted, isCapture);
    }
}