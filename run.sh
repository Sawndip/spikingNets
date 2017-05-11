#!/bin/bash
g++ -I /home/lib/armadillo/usr/local/include -L /home/lib/armadillo/usr/local/lib64 -larmadillo ./src/$1.cpp -o ./bin/$1.x -O3
LD_LIBRARY_PATH=/home/lib/armadillo/usr/local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
#./bin/$1.x
