from MultiTimescaleKMC.processor_maker import Processor_Maker
from MultiTimescaleKMC.ht_fully_lithiated_cmc import DRX_CMC
from MultiTimescaleKMC.rt_delithiated_cmc import Delithiated_RT_CMC
from MultiTimescaleKMC.multi_timescale_kmc import Multi_Time_Scale_KMC
import time

# Set up processor
size = 4
cell_size = 4 * size**3
start_time = time.time()
processor = Processor_Maker(size, size, size)  # Creates 256 sites
processor.Processor_Maker()
Processor_filename = f'Processor_{processor.cell_size}_O.pickle'
duration = time.time() - start_time
print(f"TIMERTIMER: Processor_Maker for processor.cell_size {processor.cell_size}: {duration:.2f} seconds")


# Run HT_CMC
Processor_filename = f'Processor_{cell_size}_O.pickle'
Li_conc = 1.1
Ti_conc = 0.2
nsteps = 500000
T = 1500
start_time = time.time()
ht_cmc = DRX_CMC(Li_conc, Ti_conc, Processor_filename, nsteps, T)
ht_cmc.HT_CMC()
duration = time.time() - start_time
print(f"TIMERTIMER: DRX_CMC HT_CMC for Processor_filename {Processor_filename}, Li_conc {Li_conc}, Ti_conc {Ti_conc}, nsteps {nsteps}, T {T}: {duration:.2f} seconds")


# Run Delithiated_RT_CMC
RTCMC_nsteps = 10000
RTCMC_Li_conc = 0.7
start_time = time.time()
rt_delith_cmc = Delithiated_RT_CMC(RTCMC_Li_conc, Processor_filename, RTCMC_nsteps)
rt_delith_cmc.Write_Configuration()
duration = time.time() - start_time
print(f"TIMERTIMER: Delithiated_RT_CMC HT_CMC for Processor_filename {Processor_filename}, Li_conc {RTCMC_Li_conc}, nsteps {RTCMC_nsteps}, RT: {duration:.2f} seconds")


# Run KMC
T_KMC = 300
traj_steps = 1000
start_time = time.time()
kmc = Multi_Time_Scale_KMC(T_KMC, traj_steps, Processor_filename)
kmc.run_KMC()
duration = time.time() - start_time
print(f"TIMERTIMER: KMC for T_KMC {T_KMC}, traj_steps {traj_steps}, Processor_filename {Processor_filename}: {duration:.2f} seconds")
