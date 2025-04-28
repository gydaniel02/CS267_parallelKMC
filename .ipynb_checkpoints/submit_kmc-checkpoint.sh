#!/bin/bash
#SBATCH --account=mp309
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name=kmc_fin
#SBATCH --output=fin-%j.out
#SBATCH --error=fin-%j.error
#SBATCH --mem=128G

source activate KMCenv
python -u runkmc.py $ARG1 $ARG2 $ARG3