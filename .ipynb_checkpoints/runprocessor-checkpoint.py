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
processor = Processor_Maker(dim1, dim2, dim3) # Creates the desired number of sites
processor.Processor_Maker()
Processor_filename = 'Processor_'+str(processor.cell_size)+'_O.pickle'

end_make_processor = time.time()
t_make_processor = end_make_processor - start_total
print("Time Making Processor: ", t_make_processor)