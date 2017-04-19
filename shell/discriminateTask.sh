#!/bin/bash
g++ -I /home/lib/armadillo/usr/local/include -L /home/lib/armadillo/usr/local/lib64 -larmadillo ../src/discriminateTask.cpp -o ../bin/discriminateTask.x -O3
LD_LIBRARY_PATH=/home/lib/armadillo/usr/local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
../bin/discriminateTask.x
