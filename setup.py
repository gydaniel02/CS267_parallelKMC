from MultiTimescaleKMC.processor_maker import Processor_Maker
from MultiTimescaleKMC.ht_fully_lithiated_cmc import DRX_CMC
from MultiTimescaleKMC.rt_delithiated_cmc import Delithiated_RT_CMC
from MultiTimescaleKMC.multi_timescale_kmc import Multi_Time_Scale_KMC

# Set up

# processor = Processor_Maker(4,4,4) # Creates 256 sites
# processor.Processor_Maker()
# Processor_filename = 'Processor_'+str(processor.cell_size)+'_O.pickle'
Processor_filename = 'Processor_256_O.pickle'
ht_cmc = DRX_CMC(1.1,0.2,Processor_filename,500000,1500) # HT CMC for Li1.1Mn0.7Ti0.2O2 for 500000 steps at 1500K, needs to have greater than 100000 steps for equilibration
ht_cmc.HT_CMC()
# Outputs file HT_DRX.pickle


rt_delith_cmc = Delithiated_RT_CMC(0.7,Processor_filename,10000) # Delithiate to Li0.7 at RT for 10000 steps
rt_delith_cmc.Write_Configuration() # Outputs file Delithiated_RT_DRX.pickle

# Run KMC

kmc = Multi_Time_Scale_KMC(300,1000,Processor_filename)
kmc.run_KMC()