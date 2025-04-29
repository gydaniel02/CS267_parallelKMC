#!/bin/bash
#SBATCH --account=mp309
#SBATCH --constraint=cpu
#SBATCH --qos=shared
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --job-name=kmc_fin
#SBATCH --output=fin-%j-%x.out
#SBATCH --error=fin-%j-%x.error

#SBATCH --mem=128G

conda activate KMCenv

SIZES=(32 64 256 384 864 112)

LI_CONC=1.1
TI_CONC=0.2
NSTEPS=500000
T_HT=1500
RT_LI_CONC=0.7
RT_STEPS=10000
T_KMC=300
TRAJ_STEPS=1000

for SIZE in "${SIZES[@]}"; do
  OUTFILE="output_size${SIZE}.log"
  echo "Running with size=${SIZE}... Output -> ${OUTFILE}"

  python runner.py \
    --size $SIZE \
    --Li_conc $LI_CONC \
    --Ti_conc $TI_CONC \
    --nsteps $NSTEPS \
    --T_HT $T_HT \
    --RTCMC_Li_conc $RT_LI_CONC \
    --RTCMC_nsteps $RT_STEPS \
    --T_KMC $T_KMC \
    --traj_steps $TRAJ_STEPS \
    > "$OUTFILE" 2>&1
done