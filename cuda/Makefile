CC = /usr/local/cuda-7.5/bin/nvcc

permutations: *.cu
	$(CC) -std=c++11 permutations.cu entry_point.cu -O3 -arch=sm_30 -o permutations

clean:
	rm -f *.o permutations
