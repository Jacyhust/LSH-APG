#scl enable devtoolset-9 bash

CC=g++ -std=c++17 -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -march=native -fpic -w -ftree-vectorize -ftree-vectorizer-verbose=0

CCOMP=g++ -std=c++17 -Ofast -lrt -DNDEBUG  -DHAVE_CXX0X -openmp -march=native -fpic -w -fopenmp -ftree-vectorize -ftree-vectorizer-verbose=0
SRCS=$(wildcard ./src/*.cpp)


lgo:$(SRCS)
	rm -rf lgo
	$(CCOMP) $(SRCS) -o lgo

lg:$(SRCS)
	rm -rf lg
	$(CC) $(SRCS) -o lg

clean:
	rm -rf lgo lg

cleancodes:
	rm -rf $(wildcard *.cpp) $(wildcard *.h) $(wildcard ./src/*.cpp) $(wildcard ./src/*.h)

getcode:
	7za x wLSH.7z -r -o./

cleanother:
	rm -rf $(wildcard *.txt)