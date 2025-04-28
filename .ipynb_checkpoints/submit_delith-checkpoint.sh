#!/bin/bash
#SBATCH --account=mp309
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=run_kmc
#SBATCH --output=kmc-%j.out
#SBATCH --error=kmc-%j.error
#SBATCH --mem=128G

source activate KMCenv
python -u rundelith.py $ARG1 $ARG2 $ARG3