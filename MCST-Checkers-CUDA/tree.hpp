#include <vector>
#include "utils.hpp"
#pragma once

// modify board as there was a move
#define MOVE(from, to, playerPieces, promoted)  \
        SET_BIT(playerPieces, to, true);               \
        SET_BIT(playerPieces, from, false); \
        SET_BIT(promoted, to, BIT(promoted, from));     \
        SET_BIT(promoted, from, false);

// MCT node
typedef struct node {
    // tablica 4 x 8 tylko dostêpne pola
    uint32_t whitePieces = 0;
    // tablica 4 x 8 tylko dostêpne pola
    uint32_t blackPieces = 0;
    // tablica 4 x 8 tylko dostêpne pola
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

    // pieces capturable while moving up
    uint32_t oponentPiecesCapturableUp = (freeTiles >> 4 | (freeTiles & moveLeftDownAvailble) >> 5 | (freeTiles & moveRightDownAvailble) >> 3) & newBlack & 0x0E7E'7E70;
    // pieces capturable while moving down
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

            // doubled pawn moves
            // if postion after capture is free and oponent is on rechable position, add it to capture queue to check if multicapture is possible
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
                if (i + 5 < 32 && BIT(playerDoublePawns & moveLeftDownAvailble, i + 5))
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
                if (i - 5 >= 0 && BIT(moveLeftDownAvailble, i) && BIT(newWhite, i - 5))
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

    // check capture queue for multicaptures
    while (!captureQ.empty())
    {
        uint32_t captureData = captureQ.front();
        captureQ.pop();
        uint32_t currentOponent = oponent & ~captureData;
        uint8_t currentPos = firstBit(captureData & freeTiles);
        bool isQueen = captureData & player & promoted;
        bool nextCapturePossible = false;
        int8_t positionAfter = 0;
        uint32_t boardAfterCap = 0;

        // if another capture possible add to capture queue
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

        // if no more capture possible add position after capture to availble moves queue
        if (!nextCapturePossible)
        {
            uint32_t currentPlayer = player;
            uint32_t currentPromoted = promoted;
            uint8_t startingPos = firstBit(player & captureData);
            SET_BIT(currentPlayer, currentPos, true);
            SET_BIT(currentPromoted, currentPos, isQueen || currentPos > 27);

            SET_BIT(currentPlayer, startingPos, false);
            SET_BIT(currentPromoted, startingPos, false);

            currentPromoted = currentPromoted & (currentPlayer | currentOponent);

            availbleMovesQ->push(currentPlayer);
            availbleMovesQ->push(currentOponent);
            availbleMovesQ->push(currentPromoted);
        }
    }

    anyTakeAvailble = !availbleMovesQ->empty();

    // if no take is possible check for no capture moves
    if (!anyTakeAvailble && availbleMovesNoCapWhite > 0)
        for (int8_t i = 0; i < 32; i++)
        {
            // for every field that is rechable, check how you can reach it and for every possible move add obtained position to availbleMovesQ
            if (!BIT(availbleMovesNoCapWhite, i)) continue;

            SET_BIT(newWhite, i, true);

            if (i - 4 >= 0 && BIT(newWhite, i - 4))
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
            if (i - 3 >= 0 && BIT(newWhite & moveLeftUpAvailble, i - 3))
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
            if (i - 5 >= 0 && BIT(newWhite & moveRightUpAvailble, i - 5))
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

            if (i + 4 < 32 && BIT(newWhite & newPromoted, i + 4))
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
            if (i + 5 < 32 && BIT(newWhite & newPromoted & moveLeftDownAvailble, i + 5))
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
            if (i + 3 < 32 && BIT(newWhite & newPromoted & moveRightDownAvailble, i + 3))
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
bool simulateTillEnd(uint32_t white, uint32_t black, uint32_t promoted, uint8_t movesWithoutTake, bool whiteOnMove, uint32_t& drawCount)
{
    uint32_t movesArray[MOVES_Q_SIZE];
    Queue<uint32_t> availbleMovesQ = Queue<uint32_t>(movesArray, MOVES_Q_SIZE);
    bool playerOnMove = true;   // if person calling the function is making decision
    if (!whiteOnMove)
    {
        // flip board, so player on move is always in white position
        SWAP(white, black);
        REVERSE32(white);
        REVERSE32(black);
        REVERSE32(promoted);
    }
    while (true)
    {
        // draw condition
        if (movesWithoutTake > 40)
        {
            drawCount++;
            return false;
        }
        simulateOne(white, black, promoted, movesWithoutTake, &availbleMovesQ);

        uint8_t length = availbleMovesQ.length() / 3;
        if (length == 0)
        {
            if (black == 0)
                return playerOnMove;    // if player is to play, he is white
            if (white == 0)
                return !playerOnMove;
            return !playerOnMove;       // stalemate condition
        }
        length = rand() % length;

        //bool isCapture = movesArray[length * 3 + 1] != black; // if oponent's pieces moves there was a capture
        bool isCapture = true;
        white = movesArray[length * 3];
        black = movesArray[length * 3 + 1];
        promoted = movesArray[length * 3 + 2];

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

// popluates node->children, same logic as simulateTillEnd
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
        bool isCapture = movesArray[i * 3 + 1] != black && movesArray[i * 3] != white;

        struct node* newChild = new struct node();
        newChild->whitePieces = white; 
        newChild->blackPieces = black;
        newChild->promotedPieces = promoted; newChild->parentNode = node; 
        newChild->whiteToPlay = !node->whiteToPlay; 
        newChild->movesWithoutTake = !isCapture * (node->movesWithoutTake + 1); 
        node->children.push_back(newChild);
    }
}