import os
import random
import pickle
import numpy as np
from random import sample
from pymatgen.core import Species
from collections import defaultdict
import smol.cofe.space.domain as ForVac
from custom_io import Custom_IO


class Fast_Processes_MC():

    """
    Class to perform canonical monte-carlo (CMC) between transition metal (TM) hops in Kinetic Monte carlo (KMC) simulations 
    for the relatively faster kinetic processes such as Li-Vac swaps and --- presumably --- the charge transfer processes 
    between TM valence states (Mn2+, Mn3+, Mn4+). The class deliberately avoids defining a constructor method so that it only 
    has to be instantiated once outside the for loop of the run_KMC method in the the Multi_Time_Scale_KMC class. Instead, an 
    instance of the Multi_Time_Scale_KMC class () is passed in as arguement to the Fast_Configurations_Initialization 
    method which initializes the Fast_Configs dictionary relevant for this class. This dictionary is edited and returned by 
    multiple methods in this class.
    
    """
    #TODO; make an init for the fast processes that handles the Processor

    def Select_Fast_Configuration(self, energy, Energy_All, Species_Lists, T_sample, Tet_Oct_Updater, Occupancy_Resetter, occ, e_cut, kB, T_KMC, indices, Li, Vac, Mn2, Mn3, Mn4, site_encodings, processor_file, s: int) -> dict:

        """
        This method calls upon other methods within the class to list multiple (Li/ Vac/ TM charge) configurations by perfoming 
        CMC and choose one based on boltzman statistics.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            s (int): step number
        
        """

        #if not isinstance(, Multi_Time_Scale_KMC):
        #    raise TypeError(f"{} must be an instance of the Multi_Time_Scale_KMC class")
        
        Fast_Configs = self.Fast_CMC(energy, kB, T_sample, occ, Species_Lists, indices,  Li, Vac, Mn2, Mn3, Mn4, site_encodings, processor_file, s)

        minimum = np.min(Fast_Configs['Energy_All'])

        eq_idx = np.where((Fast_Configs['Energy_All']-minimum)<e_cut)[0]

        av_energy = np.mean(Fast_Configs['Energy_All'][eq_idx])
        Energy_All = np.append(Energy_All,av_energy)

        prob_Li_Vac = np.exp(-(Fast_Configs['Energy_All']-minimum)/(kB*T_KMC))

        probs = [np.sum(prob_Li_Vac[0:i+1])/np.sum(prob_Li_Vac) for i in range(len(prob_Li_Vac))]
        r = random.uniform(0, 1)
        idx = probs.index([i for i in probs if i > r][0])

        energy = Fast_Configs['Energy_All'][idx]

        Species_Lists['Vac'] = list(Fast_Configs['Vac_configs'][idx])
        Species_Lists['Li'] = [j for j in Species_Lists['Li_Vac'] if j not in Species_Lists['Vac']]

        Species_Lists['Mn2'] = list(Fast_Configs['Mn2_configs'][idx])
        Species_Lists['Mn3'] = list(Fast_Configs['Mn3_configs'][idx])
        Species_Lists['Mn4'] = list(Fast_Configs['Mn4_configs'][idx])

        Tet_Oct_Updater(Species_Lists)
        
        occ = Occupancy_Resetter(Species_Lists)

        return Species_Lists, occ, energy
            
    def Fast_CMC(self, energy, kB, T_sample, occ, Species_Lists, indices,  Li, Vac, Mn2, Mn3, Mn4, site_encodings, processor_file, s: int) -> dict:    

        """
        Method performs (Li/ Vac/ TM charge) CMC.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            s (int): step number
        """
        
        if s==0:
            sampling_steps = 50000
        else:
            sampling_steps = 500
        
        Fast_Configs = self.Fast_Configurations_Initialization(Species_Lists, indices, energy)

        for i in range(sampling_steps):             #Metropolis algorithm for canonical Li-Vac swaps accross sublattices.
            swaps = self.Swap_List_Calculator(Species_Lists, Fast_Configs)
            Perturbation = self.Perturbation_Calculator(swaps, Species_Lists, Li, Vac, Mn2, Mn3, Mn4, site_encodings, processor_file, occ, Fast_Configs)
            r1 = random.uniform(0, 1)
            p = np.exp(-(Perturbation['e_change'])/(kB*T_sample))
            
            if (Perturbation['e_change'] < 0) or ((Perturbation['e_change'] > 0) and (p > r1)):
                energy += Perturbation['e_change']                     #Fast Process Energy change
                a_Swap,b_Swap = Perturbation['swap_pair']
                occ[a_Swap], occ[b_Swap] = Perturbation['occ_exchange_pairs']
                Fast_Configs = self.Fast_Process_Executer(Species_Lists, Perturbation, Fast_Configs)
            if energy not in Fast_Configs['Energy_All']:
                Fast_Configs = self.Fast_Config_Updater(Species_Lists, energy, Fast_Configs)
            
        return Fast_Configs
    
    def Fast_Configurations_Initialization(self, Species_Lists, indices, energy) -> dict:

        """
        Method initializes and returns a dictionary which is updated with the (Li/ Vac/ TM charge) configurations during CMC.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
        """
        
        Energy_All = np.array([energy])

        Mn2_configs = [Species_Lists['Mn2'].copy()]
        Mn3_configs = [Species_Lists['Mn3'].copy()]
        Mn4_configs = [Species_Lists['Mn4'].copy()]
        Vac_configs = [Species_Lists['Vac'].copy()]

        Mn2_oct = [x for x in Species_Lists['Mn2'] if x in indices['oct']]                                  #list(np.intersect1d(Mn2_l, tet_oct_ind['oct']))
        Mn3_oct = [x for x in Species_Lists['Mn3'] if x in indices['oct']]                                  #list(np.intersect1d(Mn3_l, tet_oct_ind['oct']))  

        Fast_Configs = {
            'Energy_All':Energy_All,
            'Mn2_configs':Mn2_configs,
            'Mn3_configs':Mn3_configs,
            'Mn4_configs':Mn4_configs,
            'Vac_configs':Vac_configs,
            'Mn2_oct':Mn2_oct,
            'Mn3_oct':Mn3_oct
        }
        
        return Fast_Configs 

    def Swap_List_Calculator(self, Species_Lists, Fast_Configs: dict) -> list:

        """
        This method returns a list of the type of perturbations possible.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            swaps (list): List of numbers representing the type of perturbations allowed in the current configuration of the structure.
        """
        
        if (len(Species_Lists['Mn2'])==0) and (len(Species_Lists['Mn3'])==0):
            swaps = [0]

        elif (len(Species_Lists['Mn2'])==0) and (len(Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0]

        #elif (len(Species_Lists['Mn2'])==0) and (len(Fast_Configs['Mn3_oct'])>0):     ###CORRECT
            #swaps = [0, 2, 4]

        elif (len(Species_Lists['Mn2'])==0) and (len(Fast_Configs['Mn3_oct'])>0):     ###CORRECT
            if len(Fast_Configs['Mn3_oct']) == 1:
                swaps = [0, 2]
            else: 
                swaps = [0, 2, 4]

        elif (len(Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(Species_Lists['Mn3'])==0):
            swaps = [0, 5]

        elif (len(Species_Lists['Mn3'])==0) and (len(Fast_Configs['Mn2_oct'])>0):
            swaps = [0, 3, 5]

        elif (len(Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0, 1, 5]

        elif (len(Fast_Configs['Mn2_oct'])>0) and (len(Species_Lists['Mn3'])>0) and (len(Fast_Configs['Mn3_oct'])==0):
            swaps = [0, 1, 3, 5]

        elif (len(Species_Lists['Mn2'])>0) and (len(Fast_Configs['Mn2_oct'])==0) and (len(Fast_Configs['Mn3_oct'])>0):
            if len(Fast_Configs['Mn3_oct']) == 1:
                swaps = [0, 1, 2, 5]
            else: 
                swaps = [0, 1, 2, 4, 5]

        elif (len(Fast_Configs['Mn2_oct'] )>0) and (len(Fast_Configs['Mn3_oct'])>0):
            if len(Fast_Configs['Mn3_oct']) == 1:
                swaps = [0, 1, 2, 3, 5]
            else: 
                swaps = [0, 1, 2, 3, 4, 5]

        return swaps
    
    def Perturbation_Calculator(self, swaps, Species_Lists, Li, Vac, Mn2, Mn3, Mn4, site_encodings, processor_file, occ, Fast_Configs: dict) -> dict:

        """
        This method returns a dictionary containing information regarding the swap chosen for proposing as a KMC step and the 
        energy change associated with it.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            swaps (list): List of numbers representing the type of perturbations allowed in the current configuration of the structure. 
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            Perturbation (dict): Information about the perturbation which will actually be proposed.     
        """
        
        a, b = 0, 0
        swap = random.sample(swaps,1)[0]

        if swap == 0:
            a = Species_Lists['Li']
            b = Species_Lists['Vac']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = Li
            b_specie = Vac

        elif swap == 1:                         #Swap
            a = Species_Lists['Mn2']
            b = Species_Lists['Mn3']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = Mn2
            b_specie = Mn3

        elif swap == 2:                         #Swap
            a = Species_Lists['Mn3']
            b = Species_Lists['Mn4']

            a_Swap = sample(Fast_Configs['Mn3_oct'],1)[0]
            b_Swap = sample(b,1)[0]

            a_specie = Mn3
            b_specie = Mn4

        elif swap == 3:                          #Swap
            a = Species_Lists['Mn4']
            b = Species_Lists['Mn2']

            a_Swap = sample(a,1)[0]
            b_Swap = sample(Fast_Configs['Mn2_oct'],1)[0]

            a_specie = Mn4
            b_specie = Mn2

        elif swap == 4:                               ##############Disproportionation Reaction

            a_Swap = sample(Fast_Configs['Mn3_oct'],1)[0]
            if len([x for x in Species_Lists['Mn3'] if x != a_Swap])==0:
                print(Fast_Configs['Mn3_oct'])
                print(a_Swap)
                print(Fast_Configs['Mn3_oct'], Species_Lists['Mn3'], [x for x in Species_Lists['Mn3'] if x != a_Swap])
            b_Swap = sample([x for x in Species_Lists['Mn3'] if x != a_Swap],1)[0]

            a_specie = Mn2
            b_specie = Mn4

        elif swap == 5:                               ##############Reverse Disproportionation Reaction

            a_Swap = sample(Species_Lists['Mn2'],1)[0]
            b_Swap = sample(Species_Lists['Mn4'],1)[0]

            a_specie = Mn3
            b_specie = Mn3

        occ_a_to_b = site_encodings[a_Swap].index(b_specie)
        occ_b_to_a = site_encodings[b_Swap].index(a_specie)
        processor = Custom_IO.load_processor(processor_file)
        energy_change = processor.compute_property_change(occ,[(a_Swap, occ_a_to_b), (b_Swap, occ_b_to_a)])[0]

        Perturbation = {
            'swap':swap,
            'swap_pair':[a_Swap,b_Swap],
            'swap_pair_lists':[a, b],
            'occ_exchange_pairs':[occ_a_to_b, occ_b_to_a],
            'e_change':energy_change
        }

        return Perturbation

    def Fast_Process_Executer(self, Species_Lists, Perturbation: dict, Fast_Configs: dict) -> dict:

        """
        This method executes the CMC perturbation by randomly choosing one and updates the relevant dictionaries/ lists.

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            Perturbation (dict): List of numbers representing the type of perturbations allowed in the current configuration of the structure. 
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes. 

        Returns:
            Fast_Configs (dict): Updates the dictionary passed in as arguement.    
        """
        
        r = Perturbation['swap'] 
        a_Swap, b_Swap = Perturbation['swap_pair'] 
        a, b = Perturbation['swap_pair_lists']
        
        if r in [0,1,2,3]:

            a_idx = a.index(a_Swap)
            b_idx = b.index(b_Swap)

            a[a_idx] = b_Swap
            b[b_idx] = a_Swap

            if r==1:
                if a_Swap in Fast_Configs['Mn2_oct']:
                    Fast_Configs['Mn2_oct'].remove(a_Swap)
                    Fast_Configs['Mn3_oct'].append(a_Swap)

                if b_Swap in Fast_Configs['Mn3_oct']:
                    Fast_Configs['Mn3_oct'].remove(b_Swap)
                    Fast_Configs['Mn2_oct'].append(b_Swap)

            if r==2:
                a_idx = Fast_Configs['Mn3_oct'].index(a_Swap)
                Fast_Configs['Mn3_oct'][a_idx] = b_Swap

            elif r==3:
                b_idx = Fast_Configs['Mn2_oct'].index(b_Swap)
                Fast_Configs['Mn2_oct'][b_idx] = a_Swap

        elif (r == 4):
            Species_Lists['Mn4'].append(a_Swap)
            Species_Lists['Mn2'].append(b_Swap)
            Species_Lists['Mn3'].remove(a_Swap)
            Species_Lists['Mn3'].remove(b_Swap)

            Fast_Configs['Mn3_oct'].remove(a_Swap)

            if b_Swap in Fast_Configs['Mn3_oct']:
                Fast_Configs['Mn3_oct'].remove(b_Swap)
                Fast_Configs['Mn2_oct'].append(b_Swap)

        elif (r == 5):
            Species_Lists['Mn2'].remove(a_Swap)
            Species_Lists['Mn4'].remove(b_Swap)
            Species_Lists['Mn3'].append(a_Swap)
            Species_Lists['Mn3'].append(b_Swap)

            if a_Swap in Fast_Configs['Mn2_oct']:
                Fast_Configs['Mn2_oct'].remove(a_Swap)
                Fast_Configs['Mn3_oct'].append(a_Swap)

            Fast_Configs['Mn3_oct'].append(b_Swap)

        return Fast_Configs
        
    def Fast_Config_Updater(self, Species_Lists, energy, Fast_Configs: dict) -> dict:            
        
        """
        This method updates the dictionary initialized in Fast_Configurations_Initialization

        Args:
             (Multi_Time_Scale_KMC): instance of the kmc class.
            Fast_Configs (dict): Configurations sampled in the Canonical Monte-carlo of the fast-processes.

        Returns:
            Fast_Configs (dict): Updates the dictionary passed in as arguement.  
        
        """
        
        Fast_Configs['Energy_All'] = np.append(Fast_Configs['Energy_All'], energy)
        Fast_Configs['Mn2_configs'].append(Species_Lists['Mn2'].copy())
        Fast_Configs['Mn3_configs'].append(Species_Lists['Mn3'].copy())
        Fast_Configs['Mn4_configs'].append(Species_Lists['Mn4'].copy())
        Fast_Configs['Vac_configs'].append(Species_Lists['Vac'].copy())

        return Fast_Configs
