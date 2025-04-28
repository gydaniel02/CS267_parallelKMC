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

rt_delith_cmc = Delithiated_RT_CMC(0.7, Processor_filename, 10000, 'HT_DRX_'+str(n_sites)+'_O.pickle', 'Delithiated_RT_DRX_'+str(n_sites)+'_O.pickle') # Delithiate to Li0.7 at RT for 10000 steps
rt_delith_cmc.Write_Configuration() # Outputs file Delithiated_RT_DRX.pickle

end_delith = time.time()
t_delith = end_delith - start_total
print("Time for Delithiation: ", t_delith)