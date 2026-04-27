#!/bin/bash
#SBATCH --job-name=ass2_p32
#SBATCH -N 1
#SBATCH --ntasks=32
#SBATCH -t 00:10:00
#SBATCH --output=logs/job32_%j.out
#SBATCH --error=logs/job32_%j.err

module purge
module load compiler/oneapi-2024/mpi/2021.12

mkdir -p results logs

P=32
PX=4; PY=4; PZ=2; PPN=32
D=7
T=5
SEED=1000
F=2
ISO=500
REPEAT=5

CSV="results/times_P${P}.csv"
echo "P,M,run,time" > $CSV

export I_MPI_PMI_LIBRARY=/lib64/libpmi2.so
for M in 120 240; do
	for ((run=1; run<=REPEAT; run++)); do
		echo "P=$P run=$run nx=ny=nz=$M"
		output=$(srun --mpi=pmi2 ./src $D $PPN $PX $PY $PZ $M $M $M $T $SEED $F $ISO)
		echo "$output"
		time=$(echo "$output" | tail -1)
		echo "$P,$M,$run,$time" >> $CSV
	done
done
