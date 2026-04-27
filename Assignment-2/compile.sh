#!/bin/bash
module purge
module load compiler/oneapi-2024/mpi/2021.12
mpicc -O3 src.c -lm -o src
