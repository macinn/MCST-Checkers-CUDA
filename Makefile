all: main

main:
	nvcc .\MCST-Checkers-CUDA\main.cu -lcurand -o MCST-Checkers

clean:
	del .\MCST-Checkers.*
