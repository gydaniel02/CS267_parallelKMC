#!/bin/bash
#SBATCH --account=mp309
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --job-name=run_htcmc
#SBATCH --output=htcmc-%j.out
#SBATCH --error=htcmc-%j.error
#SBATCH --mem=128G

source activate KMCenv
python -u runhtcmc.py $ARG1 $ARG2 $ARG3