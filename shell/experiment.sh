#!/bin/bash
g++ -I /home/lib/armadillo/usr/local/include -L /home/lib/armadillo/usr/local/lib64 -larmadillo ../src/experiment.cpp -o ../bin/experiment.x
LD_LIBRARY_PATH=/home/lib/armadillo/usr/local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
../bin/experiment.x
