# setup.py
import argparse
import time
from MultiTimescaleKMC.processor_maker import Processor_Maker
from MultiTimescaleKMC.ht_fully_lithiated_cmc import DRX_CMC
from MultiTimescaleKMC.rt_delithiated_cmc import Delithiated_RT_CMC
from MultiTimescaleKMC.multi_timescale_kmc import Multi_Time_Scale_KMC

parser = argparse.ArgumentParser(description="Run multi-timescale KMC pipeline.")
parser.add_argument("--size", type=int, default=4)
parser.add_argument("--Li_conc", type=float, default=1.1)
parser.add_argument("--Ti_conc", type=float, default=0.2)
parser.add_argument("--nsteps", type=int, default=500000)
parser.add_argument("--T_KMC", type=float, default=300)
parser.add_argument("--traj_steps", type=int, default=1000)
parser.add_argument("--RTCMC_Li_conc", type=float, default=0.7)
parser.add_argument("--RTCMC_nsteps", type=int, default=10000)
parser.add_argument("--T_HT", type=float, default=1500)

args = parser.parse_args()

print("\n================== KMC Pipeline Parameters ==================")
print(f"Size                : {args.size}")
print(f"Li_conc (HT)        : {args.Li_conc}")
print(f"Ti_conc (HT)        : {args.Ti_conc}")
print(f"HT_CMC steps        : {args.nsteps}")
print(f"T_HT (HT temp)      : {args.T_HT}")
print(f"RTCMC_Li_conc       : {args.RTCMC_Li_conc}")
print(f"RTCMC_nsteps        : {args.RTCMC_nsteps}")
print(f"T_KMC (KMC temp)    : {args.T_KMC}")
print(f"KMC traj steps      : {args.traj_steps}")
print("============================================================\n")

# Set up processor
start_time = time.time()
processor = Processor_Maker(args.size, args.size, args.size)
processor.Processor_Maker()
Processor_filename = f'Processor_{processor.cell_size}_O.pickle'
duration = time.time() - start_time
print(f"TIMERTIMER: Processor_Maker for processor.cell_size {processor.cell_size}: {duration:.2f} seconds")

# Run HT_CMC
start_time = time.time()
ht_cmc = DRX_CMC(args.Li_conc, args.Ti_conc, Processor_filename, args.nsteps, args.T_HT)
ht_cmc.HT_CMC()
duration = time.time() - start_time
print(f"TIMERTIMER: DRX_CMC HT_CMC for Processor_filename {Processor_filename}, "
      f"Li_conc {args.Li_conc}, Ti_conc {args.Ti_conc}, nsteps {args.nsteps}, T {args.T_HT}: {duration:.2f} seconds")

# Run Delithiated_RT_CMC
start_time = time.time()
rt_delith_cmc = Delithiated_RT_CMC(args.RTCMC_Li_conc, Processor_filename, args.RTCMC_nsteps)
rt_delith_cmc.Write_Configuration()
duration = time.time() - start_time
print(f"TIMERTIMER: Delithiated_RT_CMC HT_CMC for Processor_filename {Processor_filename}, "
      f"Li_conc {args.RTCMC_Li_conc}, nsteps {args.RTCMC_nsteps}, RT: {duration:.2f} seconds")

# Run KMC
start_time = time.time()
kmc = Multi_Time_Scale_KMC(args.T_KMC, args.traj_steps, Processor_filename)
kmc.run_KMC()
duration = time.time() - start_time
print(f"TIMERTIMER: KMC for T_KMC {args.T_KMC}, traj_steps {args.traj_steps}, "
      f"Processor_filename {Processor_filename}: {duration:.2f} seconds")
