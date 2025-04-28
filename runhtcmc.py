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

ht_cmc = DRX_CMC(1.1, 0.2, Processor_filename, 500000, 1500, 'HT_DRX_'+str(n_sites)+'_O.pickle') # HT CMC for Li1.1Mn0.7Ti0.2O2 for 500000 steps at 1500K, needs to have greater than 100000 steps for equilibration
ht_cmc.HT_CMC()

end_ht_cmc = time.time()
t_ht_cmc = end_ht_cmc - start_total
print("Time for High Temp CMC: ", t_ht_cmc)