#!/bin/bash
#SBATCH --account=mp309
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --job-name=pickle
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.error
#SBATCH --mem=64G

source activate KMCenv
python -u runprocessor.py $ARG1 $ARG2 $ARG3