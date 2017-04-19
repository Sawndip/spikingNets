#!/bin/bash
#PBS -N discriminateTask
#PBS -M javier.prietop@estudiante.uam.es
#PBS -m abe
#PBS -q workq
#PBS -l nodes=1:ppn=1
#PBS -j oe
echo "Running on Node `hostname`"
date
LD_LIBRARY_PATH=/home/lib/armadillo/usr/local/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH
#/home/javier/spikingNets/bin/experiment.x
