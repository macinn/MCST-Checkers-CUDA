# Monte Carlo Tree Search Checkers

## Description
Implementation of [american checkers](https://www.usacheckers.com/) AI using [MCST](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) using [CUDA 12.2](https://developer.nvidia.com/cuda-downloads) and C++. The application includes CPU single-core and GPU implementation. The user can face off against one of the bots, or the bots can play against each other. 
A simple command line interface is provided. Using NVIDIA GTX 1060 performance of **over 3 million** game simulations per second was achived.

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

### Optimization
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
$ MCST-Checkers.exe
```

## License
MIT

