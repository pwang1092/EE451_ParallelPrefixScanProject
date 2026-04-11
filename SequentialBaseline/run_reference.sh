#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=SequentialData/logs/reference_%j.out
#SBATCH --error=SequentialData/logs/reference_%j.err

mkdir -p SequentialData/logs

module purge
module load gcc/11.3.0

g++ -O2 -std=c++17 -o run_reference run_reference.cpp
./run_reference ../SyntheticData/inputs SequentialData