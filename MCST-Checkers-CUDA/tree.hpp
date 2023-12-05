#include <vector>
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

// wierzcho³ek MCT
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
    uint32_t movesWithoutTake = 0;
    uint32_t gamesPlayed = 0;
    uint32_t gamesWon = 0;
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