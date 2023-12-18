#include <chrono>
#include <algorithm>

#include "tree.hpp"
#pragma once

#define INIT_PIECES(whitePieces, blackPieces) \
    whitePieces = 0x00000FFF; \
    blackPieces = 0xFFF00000; \

void printBoard(uint32_t whitePieces, uint32_t blackPieces, uint32_t promotedPieces);
void printBoardBinary(uint32_t piecies);

#define DEFAULT_TIME_LIMIT 1000         // ms to calculate new move
#define DEFAULT_NUM_SIMULATIONS 4096    // number of simulations after each exploration stage

class Player
{
public:
    uint32_t numberSimulations = DEFAULT_NUM_SIMULATIONS;
    uint32_t timeLimit = DEFAULT_TIME_LIMIT;
    const bool isWhite;
    struct node* root;
    Player(bool isWhite, uint32_t numberSimulations = DEFAULT_NUM_SIMULATIONS) : isWhite(isWhite), numberSimulations(numberSimulations)
    {
        root = new node();
        root->whiteToPlay = true;
        INIT_PIECES(root->whitePieces, root->blackPieces);
    }

    ~Player()
    {
        deleteNode(root);
    }

    void LoadPosition(uint32_t white, uint32_t black, uint32_t promoted, bool whiteToPlay)
    {
        deleteNode(root);
        root = new node();
        root->whitePieces = white;
        root->blackPieces = black;
        root->promotedPieces = promoted;
        root->whiteToPlay = whiteToPlay;
    }
    // find next move using MCTS and make it
    node* FindNextMove()
    {
#ifdef DEBUG
        std::cout << "Debug: " << root->whitePieces << " " << root->blackPieces << " " << root->promotedPieces << std::endl;
#endif // DEBUG

        if (!(root->whitePieces > 0 && root->blackPieces > 0 && root->movesWithoutTake <= 80))
            return NULL;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        node* nextBest = root;
        while(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - begin).count() < timeLimit)
        {
            // Selection
            nextBest = Selection();

            // Expansion
            generateChildren(nextBest);
            if (nextBest->children.size() == 0)
            {
                nextBest->gamesPlayed += numberSimulations;

                if (nextBest->blackPieces == 0)
                {
                    if (isWhite)
                        nextBest->gamesWon += numberSimulations;
                }
                else if (nextBest->whitePieces == 0)
                {
                    if (!isWhite)
                        nextBest->gamesWon += numberSimulations;
                }
            }
            else
            {
                nextBest = nextBest->children[rand() % nextBest->children.size()];

                // Simulation
                nextBest->gamesPlayed += numberSimulations;
                Simulate(nextBest);

            }
            // Backpropagation
            BackPropagate(nextBest);
        }

        MakeBestMove();

#ifdef DEBUG
        std::cout << "Debug: " << root->whitePieces << " " << root->blackPieces << " " << root->promotedPieces << std::endl;
#endif // DEBUG

        return root;
    }

    // print current board
    void Print()
    {
#ifdef DEBUG
        std::cout << "Debug: " << root->whitePieces << " " << root->blackPieces << " " << root->promotedPieces << std::endl;
#endif // DEBUG

        printBoard(root->whitePieces, root->blackPieces, root->promotedPieces);

#ifdef DEBUG
        if (root->whiteToPlay == isWhite) // player on move
            std::cout << "Winrate: " << root->gamesWon / (double)root->gamesPlayed << std::endl;
        else // oponent on move
        {
            std::cout << "Winrate: " << 1.0 - root->gamesWon / (double)root->gamesPlayed << std::endl;
        }
#endif // DEBUG
    }

    // move root to a node that mathes given
    bool MakeMove(node* newNode)
    {
        node* newRoot = nullptr;
        for (struct node* child : root->children)
        {
            if (child->whitePieces != newNode->whitePieces
                || child->blackPieces != newNode->blackPieces
                || child->promotedPieces != newNode->promotedPieces)
            {
                deleteNode(child);
            }
            else
                newRoot = child;
        }
        if (newRoot != nullptr)
        {
            newRoot->parentNode = nullptr;
            root = newRoot;
            return true;
        }
        else
        {
            root = newNode;
            root->parentNode = nullptr;
            return true;
        }
    }

    // move root to a node that can be achived by moving piece from to a given idx
    bool InputMove(uint8_t moveFrom, uint8_t moveTo)
    {
        const std::string badRequestInfo = "Illegal move!";
        uint32_t newWhite = root->whitePieces;
        uint32_t newBlack = root->blackPieces;
        // generate all postion reachable from current, in case root has not expanded 
        node fakeRoot = *root;
        fakeRoot.children.clear();
        generateChildren(&fakeRoot);
        // if move is legal make it, if its not possible display message return false
        if (root->whiteToPlay)
        {
            if ((!BIT(newWhite, moveFrom)) || (BIT((newWhite | newBlack), moveTo)))
            {
                std::cerr << badRequestInfo << std::endl;
                return false;
            }
            SET_BIT(newWhite, moveFrom, false);
            SET_BIT(newWhite, moveTo, true);
            for (node* child : fakeRoot.children)
            {
                if (child->whitePieces == newWhite)
                {
                    return MakeMove(child);
                }
            }
            std::cerr << badRequestInfo << std::endl;
            return false;
        }
        else
        {
            if (!BIT(newBlack, moveFrom) || BIT(newWhite | newBlack, moveTo))
            {
                std::cerr << "Illegal move" << std::endl;
                return false;
            }
            SET_BIT(newBlack, moveFrom, false);
            SET_BIT(newBlack, moveTo, true);
            for (node* child : fakeRoot.children)
            {
                if (child->blackPieces == newBlack)
                {
                    return MakeMove(child);
                }
            }
            std::cerr << "Illegal move" << std::endl;
            return false;
        }
    }

    // backpropagation stage of MCTS
    void BackPropagate(node* child)
    {
        node* p = child->parentNode;
        while (p)
        {
            p->gamesPlayed += child->gamesPlayed;
            bool sameColor = p->whiteToPlay && isWhite;
            p->gamesWon += sameColor * child->gamesWon +
                !sameColor * (child->gamesPlayed - child->gamesWon);
            p = p->parentNode;
        }
    }

    // selection stage of MCTS
    node* Selection()
    {
        node* p = root;
        struct node* best = nullptr;
        double bestScoree = 0;
        double logN = log(p->gamesPlayed);
        while (p && !p->children.empty())
        {
            for (struct node* child : p->children)
            {
                if (child->gamesPlayed == 0)
                {
                    best = child;
                    break;
                }
                // in node we store how many games are a win for player currently on move, so every time we choose node with least wins for oponent
                double score = child->gamesPlayed - child->gamesWon;
                score /= child->gamesPlayed;
                score += sqrt(EXPLORATION_CONST_SQARED * logN / child->gamesPlayed);
                if (score > bestScoree)
                {
                    best = child;
                    bestScoree = score;
                }
            }
            p = best;
            best = nullptr;
            bestScoree = 0;
        }
        return p;
    }

    // select best move from children of root and make that move
    void MakeBestMove()
    {
        struct node* newRoot = nullptr;
        double bestScoree = 0;
        double logN = log(root->gamesPlayed);
        for (struct node* child : root->children)
        {
            double score = child->gamesPlayed - child->gamesWon;
            score /= child->gamesPlayed;
            score += sqrt(EXPLORATION_CONST_SQARED * logN / child->gamesPlayed);
            if (score > bestScoree)
            {
                deleteNode(newRoot);
                newRoot = child;
                bestScoree = score;
            }
            else
            {
                deleteNode(child);
            }
        }
        root = newRoot;
        delete root->parentNode;
        root->parentNode = nullptr;
    }

    // method to implement simulations
    virtual void Simulate(node* node) = 0;

    bool gameEnded()
    {
        return !(root->whitePieces > 0 && root->blackPieces > 0 && root->movesWithoutTake <= 80);
    }
};

class PlayerCPU : public Player
{
public:
    PlayerCPU(bool isWhite, uint32_t numberSimulations): Player(isWhite, numberSimulations){}

    void Simulate(node* node) override
    {
        uint32_t drawCount = 0;
        for (uint32_t j = 0; j < numberSimulations; j++)
        {
            node->gamesWon += simulateTillEnd(node->whitePieces, node->blackPieces, node->promotedPieces,
                node->movesWithoutTake, node->whiteToPlay, drawCount);
        }
        // half point for a draw, possible loss of 0.5
        node->gamesWon += (drawCount - 1) / 2 + 1;
    }
};


