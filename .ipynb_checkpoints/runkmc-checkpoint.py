import time
import sys
import numpy as np
from MultiTimescaleKMC.processor_maker import Processor_Maker
from MultiTimescaleKMC.ht_fully_lithiated_cmc import DRX_CMC
from MultiTimescaleKMC.rt_delithiated_cmc import Delithiated_RT_CMC
from MultiTimescaleKMC.multi_timescale_kmc import Multi_Time_Scale_KMC


# Set up
dim1 = int(sys.argv[1])
dim2 = int(sys.argv[2])
dim3 = int(sys.argv[3])

print("Dimensions: ", dim1, dim2, dim3)
n_sites = dim1 * dim2 * dim3 * 4

start_total = time.time()
Processor_filename = 'Processor_'+str(n_sites)+'_O.pickle'

kmc = Multi_Time_Scale_KMC(300, 1000, Processor_filename, 'Evolution_300K_'+str(n_sites)+'_O.pickle', 'Delithiated_RT_DRX_'+str(n_sites)+'_O.pickle')
kmc.run_KMC()

end_kmc = time.time()
t_kmc = end_kmc - start_total
print("Time for KMC: ", t_kmc)