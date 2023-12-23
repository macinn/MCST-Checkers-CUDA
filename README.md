# Monte Carlo Tree Search Checkers
## Abstract
Simulate checkers game using move prediction performed using Monte-Carlo Tree search method. 
**Monte Carlo tree search (MCTS)** is a heuristic search algorithm for some kinds of decision processes, most notably those employed in software that plays board games. In that context MCTS is used to solve the game tree.
### Principle of operation
- _Selection_: Start from root R and select successive child nodes until a leaf node L is reached. The root is the current game state and a leaf is any node that has a potential child from which no simulation (playout) has yet been initiated. The section below says more about a way of biasing choice of child nodes that lets the game tree expand towards the most promising moves, which is the essence of Monte Carlo tree search. <br>
- _Expansion_: Unless L ends the game decisively (e.g. win/loss/draw) for either player, create one (or more) child nodes and choose node C from one of them. Child nodes are any valid moves from the game position defined by L. <br>
- _Simulation_: Complete one random playout from node C. This step is sometimes also called playout or rollout. A playout may be as simple as choosing uniform random moves until the game is decided (for example in chess, the game is won, lost, or drawn). <br>
- _Backpropagation_: Use the result of the playout to update information in the nodes on the path from C to R. <br>

![MCST-Principle](https://github.com/macinn/MCST-Checkers-CUDA/assets/118574079/bedda759-7c74-492a-89d9-81b7a02fd36e)
### UCB
The main difficulty in selecting child nodes is maintaining some balance between the exploitation of deep variants after moves with high average win rate and the exploration of moves with few simulations. The first formula for balancing exploitation and exploration in games, called UCT (Upper Confidence Bound 1 applied to trees). It is recommend to choose in each node of the game tree the move for which the following expression has the highest value.
```math
\frac {w_{i}} {n_{i}}+c{\sqrt {\frac {\ln N_{i}}{n_{i}}}}
```
- $w_{i}$ - number of wins for the node considered after the i-th move. <br>
- $n_{i}$ - number of simulations for the node considered after the i-th move. <br>
- $N_{i}$ - total number of simulations after the i-th move run by the parent node of the one considered. <br>
- $c$ - exploration parameter, theoretically equal to $\sqrt{2}$

## Description
Implementation of [american checkers](https://www.usacheckers.com/) AI using [MCST](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) using [CUDA 12.2](https://developer.nvidia.com/cuda-downloads) and C++. The application includes CPU single-core and GPU implementation. The user can face off against one of the bots, or the bots can play against each other. 
A simple command line interface is provided. 

         | A| B| C| D| E| F| G| H|
        -+--+--+--+--+--+--+--+--+-
        8|  |B |  |B |  |B |  |B |8
        -+--+--+--+--+--+--+--+--+-
        7|B |  |B |  |B |  |B |  |7
        -+--+--+--+--+--+--+--+--+-
        6|  |B |  |B |  |B |  |B |6
        -+--+--+--+--+--+--+--+--+-
        5|  |  |  |  |  |  |  |  |5
        -+--+--+--+--+--+--+--+--+-
        4|  |  |  |  |  |  |  |  |4
        -+--+--+--+--+--+--+--+--+-
        3|W |  |W |  |W |  |W |  |3
        -+--+--+--+--+--+--+--+--+-
        2|  |W |  |W |  |W |  |W |2
        -+--+--+--+--+--+--+--+--+-
        1|W |  |W |  |W |  |W |  |1
        -+--+--+--+--+--+--+--+--+-
         | A| B| C| D| E| F| G| H|

### Performance
Using NVIDIA GTX 1060 performance of **over 3 million** simulations per second was achived.
Following techniques were used:
  - bitwise operations
  - branchless programming
  - board is stored in 96 bits
  - capture is saved in 32 bits
  - custom random number generator

  
## How To Use

### Prerequisites

Download and install the [CUDA Toolkit 12.2](https://developer.nvidia.com/cuda-downloads) for your corresponding platform.
For system requirements and installation instructions of cuda toolkit, please refer to the [Linux Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-linux/), and the [Windows Installation Guide](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html).

### Building solution

```bash
# Clone this repository
$ git clone https://github.com/macinn/MCST-Checkers-CUDA

# Go into the repository
$ cd MCST-Checkers-CUDA

# Compile
$ make
# or
$ nvcc .\MCST-Checkers-CUDA\main.cu -lcurand -o MCST-Checkers

# Run the app
$ .\MCST-Checkers.exe
```
