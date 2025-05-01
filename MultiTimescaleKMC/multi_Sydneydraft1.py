### TIMING RELATED TASKS
import cProfile
import pstats
import io
import functools
import time

def profiled(name=None, print_stats=True, save_to_file=None):
    """
    Decorator to profile a Ray remote function or actor method.
    
    Args:
        name (str): Optional name for the profile output.
        print_stats (bool): Whether to print the stats to stdout.
        save_to_file (str): If given, saves the profile stats to this file.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()

            result = func(*args, **kwargs)

            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
            ps.print_stats(30)  # top 30 functions

            profile_name = name or func.__name__
            if print_stats:
                print(f"\n=== Profile Results for {profile_name} ===\n{s.getvalue()}")

            if save_to_file:
                with open(save_to_file, 'w') as f:
                    f.write(s.getvalue())

            return result
        return wrapper
    return decorator

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Function {func.__name__} took {end_time - start_time} seconds to execute.")
        return result
    return wrapper

# example use cases: #1) a task
@ray.remote 
@profiled(name="my_task", print_stats=True, save_to_file="my_task.prof")
def my_task(x):
    # Expensive computation
    total = 0
    for i in range(1000000):
        total += (i * x) % 7
    return total

# example use cases: #2) an actor
@ray.remote
class MyActor:
    @profiled(name="actor_method")
    def heavy_compute(self, x):
        return sum((i * x) % 7 for i in range(1000000))

### HELPER FUNCTIONS FOR KMC
import os
import random
import pickle
import numpy as np
from MultiTimescaleKMC.custom_io import Custom_IO
from pymatgen.core import Species
from collections import defaultdict,namedtuple
from MultiTimescaleKMC.fast_processes import Fast_Processes_MC #lithium shuffling
from MultiTimescaleKMC.common_mc_initialization import Common_Class
from MultiTimescaleKMC.initial_structures import Initial_Structure_Makers
from MultiTimescaleKMC.local_structure_details import Local_Structure_Details
import smol.cofe.space.domain as ForVac


import ray

ray.init()


def KMC(T_KMC: int, traj_steps: int, processor_file: str, EVOFILE: str, RT_CMC_results_file:str = "Delithiated_RT_DRX.pickle"):
    
    ### SETUP KMC
    # formerly super().__init__(processor_file)
    Li = Species.from_str("Li+")
    Vac = ForVac.Vacancy()
    Mn2 = Species.from_str("Mn2+")
    Mn3 = Species.from_str("Mn3+")
    Mn4 = Species.from_str("Mn4+")
    Ti4 = Species.from_str("Ti4+")
    O2 = Species.from_str("O2-")
    
    kB = 8.617*10**-5
    
    processor = Custom_IO.load_processor(processor_file)
    n_sites = processor.num_sites
    
    site_encodings = processor.allowed_species
    
    indices = Local_Structure_Details.struct_indicies(processor)
    
    nns = Local_Structure_Details.Nearest_Neighbor_Calculator(processor, indices)

    Species_Lists = Custom_IO.load_pickle(RT_CMC_results_file)         #You should have this file within the current directory
    Species_Lists.pop('Energy_All')
    Species_Lists["Mn2"]=[]
    Species_Lists["O2"]=indices['O2']
    ### Defines composite species sets
    Species_Lists['Li_Vac'] = Species_Lists['Vac'].copy()+Species_Lists['Li'].copy()
    ### Counts number of atoms excluding Li_Vac and Vac
    n_atoms = np.sum([len(Species_Lists[species]) for species in Species_Lists if (species!='Li_Vac') and (species!='Vac')])

    Tet_Oct_Updater()
    spec_type = [Li, Vac, Mn3, Mn4, Ti4, Mn2, O2] #TODO: this is only used for Occupancy Resetter
    occ = Occupancy_Resetter(processor, Species_Indices(Species_Lists))

    ### Update which sites are in tet and oct based on the Common Class definition of indices
    energy = processor.compute_property(occ)[0]
    av_energy = 0
    Energy_All = np.array([])
    
    T_KMC = T_KMC
    e_cut = 6.96*10**-3 * T_KMC #TODO: what is this used for??
    T_sample = 2000 #TODO: is this different than T_KMC. why?
    
    traj_steps = traj_steps
    #disorder_fraction = disorder_fraction

    #Redox_Neighbors = Redox_Center_Calculator()
    
    Conf = defaultdict(dict)           
    evolution_filename = EVOFILE
    Time = 0 
    step_file_name = "Step_number.txt"


    All_Hops = {
        'counter':-1,
        'Hops':defaultdict(dict),
        'Activation_Barriers':[],
        'Energy_Changes':[]
    }       

    Hop_Mechanisms = {
        "Tri-Vac":{},
        "Di-Vac":{},
        "Mono-Vac":{},
    }
    
    BarrierParams = namedtuple("BarrierParams", ["kra", "end_state", "encoding"])
    
    # based off of number of vacancies adjacent, species name. Why Mn4 and Ti4 included?
    barrier_map = {
        "Tri-Vac": {
            "Mn3_oct": BarrierParams(0.67, Mn3, 1),
            "Mn3_tet": BarrierParams(0.67, Mn3, 2),
            "Mn2_oct": BarrierParams(0.3, Mn2, 3),
            "Mn2_tet": BarrierParams(0.3, Mn2, 4),  
            "Mn4": BarrierParams(1.5, Mn4, 5),
            "Ti4": BarrierParams(0.87, Ti4, 6),
        },
        "Di-Vac": {
            "Mn3_oct": BarrierParams(0.75, Mn3, 7),
            "Mn3_tet": BarrierParams(0.75, Mn3, 8),
            "Mn2_oct": BarrierParams(0.3, Mn2, 9),
            "Mn2_tet": BarrierParams(0.3, Mn2, 10),
        },
        "Mono-Vac": {
            "Mn2_oct": BarrierParams(0.45, Mn2, 11),
            "Mn2_tet": BarrierParams(0.45, Mn2, 12),
        },
    }

    hop_types = {
        "mn3_mn3": [1, 2, 7, 8],
        "mn2_mn2": [3, 4, 9, 10, 11, 12],
        "mn4_mn4": [5],
        "ti4_ti4": [6]
    }


    ### RUN KMC
    Fast_Processes = Fast_Processes_MC()
    for s in range(traj_steps):    
        Fast_Processes.Select_Fast_Configuration(s)      #JUST TO PROVE THAT WE HAVE DONE OUR DUE DILLIGENCE
        Hop(s)
        # if (s!=0) and (s%2000==0): #TODO: probably not needed.
        #     plot_energy_evolution()




# import matplotlib.pyplot as plt
# #TODO: not used
# def plot_energy_evolution(self):
#     """
#     Method to plot how the energy of the system has evolved in the MC simulation upto this point.
#     """
    
#     plt.figure(figsize=(12,8))
    
#     Steps = [x for x in range(len(Energy_All))]
#     Energy = (Energy_All-Energy_All[0])/n_atoms
    
#     plt.scatter(Steps,Energy)
    
#     plt.tight_layout()
    
#     fontsize = 35
#     plt.xlabel('Steps',fontsize = fontsize)
#     plt.ylabel('Energy (eV/atom)', fontsize = fontsize)
#     plt.xticks([0,int(np.max(Steps)/2),np.max(Steps)], fontsize = fontsize)
#     plt.yticks(fontsize = fontsize)        
#     plt.show()

def Species_Indices(Species_Lists):
    
    unwanted_lists = ['Li_Vac', "Mn3_tet", "Mn3_oct", "Mn2_tet", "Mn2_oct"]
    spec_indices = [Species_Lists[species] for species in Species_Lists if species not in unwanted_lists]
    return spec_indices

# helper function for runner KMC
def Occupancy_Resetter(processor, spec_indices = None, spec_type = None):

        """
        Method to calculate the occupancies of all sites within the structure, given the types of species in the structure and their 
        corresponding indices.

        Args:
            spec_type (list[Species | Vacancy], optional): List of Species present in the structure. Deaults to stealing the spec_type 
                                                           attribute from its child class (see if statement below).
            spec_indices (list[list[int]], optional): A 2D list of integer type representing indices in the supercell structure. Indices 
                                                      in the inner lists correspond to the Species defined in spec_type. First dimension 
                                                      of spec_indices and length of spec_indices must be the same. Deaults to building it 
                                                      using the Species_Indices method of its child class (see if statement below).
        
        """
        
        species_list = [0 for x in range(n_sites)]
        
        if (spec_type==None) and (spec_indices==None):
            spec_indices = Species_Indices()
            spec_type = spec_type

        for idx, si in enumerate(spec_indices):
            for ind in si:
                species_list[ind] = spec_type[idx]

        occ = processor.encode_occupancy(species_list)

        return occ
    

### GOAL IS TO MAKE THESE DISTRIBUTED TASKS

#modified to take in step_file_name
def Hop(s: int, step_file_name: str):
        
    """
    This method calls upon other methods to list all possible hops, advance the time based on the list, 
    find the relevant hop to be executed, save infomation regarding this hop, execute the hop and write the step number of the 
    current hop in the file named in the step_file_name attribute. 

    Args:
        s (int): current KMC step number
    """
    
    All_Possible_Hops()       
    # Time_Advancement() #incorporated time adv into all_possible_hops
    the_hop, encoding, energychg = Hop_Finder()                     #The TM Hop energy change was updated
    Write_Evolution_File(s, the_hop, encoding)                   #encoding is for keeping track of the mechanism
    Hop_Executer(the_hop, encoding)  
    Custom_IO.write_step_file(s, step_file_name)

#modified to take in new variables and include the time advancement
# still has no return value for now
def All_Possible_Hops(Species_Lists, nns, barrier_map):
    #TODO this is the main thing
    
    """
    Method to list all possible Transition metal hops.
    """
    All_Hops = {
        'counter':-1,
        'Hops':defaultdict(dict),
        'Activation_Barriers':[],
        'Energy_Changes':[]
    }       

    Hop_Mechanisms = {
        "Tri-Vac":{},
        "Di-Vac":{},
        "Mono-Vac":{},
    }
    
    # Find the mobile ions. Are these lists static or only changing briefly? unfortunately, no because of Li relaxation.
    # However, I would expect that the Mn and O stay stationary while only the Li moves during the CMC
    Mobile_Mn_tet = Species_Lists["Mn2_tet"]+Species_Lists["Mn3_tet"]
    Mobile_Mn_oct = Species_Lists["Mn2_oct"]+Species_Lists["Mn3_oct"]
    # Mobile_Mn = Species_Lists['Mn2']+Species_Lists['Mn3']

    tet_vac = [x for x in Species_Lists['Vac'] if x in indices['tet']]
    oct_vac = [x for x in Species_Lists['Vac'] if x in indices['oct']]
    print(f"There are {len(tet_vac)} tet_vac and {len(oct_vac)} vacancies = {sum([len(tet_vac), len(oct_vac)])} total vacancies.")
    print(f"There are {len(Mobile_Mn_tet)} Mn_tet and {len(Mobile_Mn_oct)} Mobile_Mn_oct = {sum([len(Mobile_Mn_oct), len(Mobile_Mn_tet)])} total vacancies.")
    Pristine_oct_vacs = []
    """✅ Can be parallelized (with some effort):

Memory independent per iteration → each o is processed independently.

Requires nns[o] and Species_Lists['Vac'] to be passed in a numba-friendly format (e.g., np.ndarray, set).

Appending to Pristine_oct_vacs inside parallel code needs to be done with a temporary buffer per thread and reduced afterward."""
    for o in oct_vac:
        oct_pristine_vacs = len([x for x in nns[o] if x in Species_Lists['Vac']])
        if oct_pristine_vacs==8:
            Pristine_oct_vacs.append(o)
    

    """⚠️ Not trivially parallelizable:

Calls Mechanism_Update() which likely involves mutable state.

Writes to shared dictionary (Hop_Mechanisms), which is not safe in parallel.

You’d need to refactor the loop to first collect data in thread-safe containers, then apply updates sequentially.

Verdict: Rewrite for two-pass logic (gather → update) if you want to parallelize.

"""

    ### VACANCY CENTRIC LOOP ABOUT TRANSITION STATE
    for vac in tet_vac:                    #Differentiating vacant tetraherdrals for hopping mechanism
        
        v_int = len([x for x in nns[vac] if x in Species_Lists['Vac']]) #number of vacancies adjacent to the tetrahedral
        if (v_int in [1,2,3]):
            FS_TMs = [x for x in nns[vac] if x in Mobile_Mn_oct] #which Mn are nearby 
            if (v_int==3) and (len(FS_TMs)==1):    #Trivac
                Mechanism_Update(FS_TMs[0], "Tri-Vac")
                Hop_Mechanisms["Tri-Vac"][FS_TMs[0]].append(vac) 
            elif (v_int==2):
                li_int = len([x for x in nns[vac] if x in Species_Lists['Li']])
                if (len(FS_TMs)==1) and (li_int==1):    #Di-vac
                    Mechanism_Update(FS_TMs[0], "Di-Vac")
                    Hop_Mechanisms["Di-Vac"][FS_TMs[0]].append(vac)         
            elif (v_int==1):
                li_int = len([x for x in nns[vac] if x in Species_Lists['Li']])
                if (len(FS_TMs)==1) and (li_int==2):    #Mono-vac mechanism
                    Mechanism_Update(FS_TMs[0], "Mono-Vac")
                    Hop_Mechanisms["Mono-Vac"][FS_TMs[0]].append(vac) 

        FS_Mn4s = [x for x in nns[vac] if x in Species_Lists['Mn4']] #Mn4 only looking for oct-oct transitions
        if (len(FS_Mn4s)==1):           
            v_octs = [x for x in nns[vac] if x in Pristine_oct_vacs]
            if (len(v_octs)==3):
                Mechanism_Update(FS_Mn4s[0], "Tri-Vac")
                for v_oct in v_octs:                
                    Hop_Mechanisms["Tri-Vac"][FS_Mn4s[0]].append(v_oct) 
                    
        FS_Tis = [x for x in nns[vac] if x in Species_Lists['Ti4']] #Ti4 only looking for oct-oct transitions
        if (len(FS_Tis)==1):           
            v_octs = [x for x in nns[vac] if x in Pristine_oct_vacs]
            if (len(v_octs)==3):
                Mechanism_Update(FS_Tis[0], "Tri-Vac")
                for v_oct in v_octs:                
                    Hop_Mechanisms["Tri-Vac"][FS_Tis[0]].append(v_oct)
    
    """⚠️ Same situation as above:

Dependent on nns[mn], Species_Lists, and writes to Hop_Mechanisms.

Not trivially parallelizable without extracting state updates from loop.

Verdict: Can be parallelized only if split into analysis + assignment phases."""

    ## LOOPS OVER NEARBY VACANCIES TO SEE IF MN SITS IN A FAVORABLE ENVIRONMENT
    for mn in Mobile_Mn_tet:               #Differentiating Mn tetraherdrals for hopping mechanism
        FS_Vacs = [x for x in nns[mn] if x in Species_Lists['Vac']]
        if (len(FS_Vacs)==4):                                   #Trivac
            Hop_Mechanisms["Tri-Vac"][mn] = FS_Vacs
        elif (len(FS_Vacs)==3):                                   #Divac
            li_int = len([x for x in nns[mn] if x in Species_Lists['Li']])
            if li_int==1:
                Hop_Mechanisms["Di-Vac"][mn] = FS_Vacs
        elif (len(FS_Vacs)==2):                                   #Monovac
            li_int = len([x for x in nns[mn] if x in Species_Lists['Li']])
            if li_int==2:
                Hop_Mechanisms["Mono-Vac"][mn] = FS_Vacs 


    """🚫 Not directly parallelizable:

Deeply nested with heterogeneous function calls.

Calls Barrier_Calculator, likely updating shared state.

Dictionary and dynamic function calling don’t map cleanly to GPU/parallel execution.

Verdict: You could precompute a flattened list of (mn, vac, mechanism) and pass it to a parallel kernel — but this requires restructuring."""
    print(f"Number of mechanisms: {len(Hop_Mechanisms)}, Number of Mn considered {sum([len(Hop_Mechanisms[mech]) for mech in Hop_Mechanisms])}")
    
    for mechanism in Hop_Mechanisms:
        for mn in Hop_Mechanisms[mechanism]:
            for vac in Hop_Mechanisms[mechanism][mn]:
                for cation_description, hop_info in barrier_map[mechanism].items():
                    if mn in Species_Lists[cation_description]:  # Dynamically fetch the correct set (Mn3_oct, etc.) #globals()[mn_type]
                        Barrier_Calculator(hop_info.kra, int(mn), vac, hop_info.end_state, hop_info.encoding)
                        break  # Exit the loop once a match is found

    """
    Method to advance the time associated with the TM hop. 
    """
    #Gillespie method=first-reaction method, exponential distribution of waiting times in Poisson process
    
    attempt_frequency = 1e13
    rate_consts = np.exp(-np.array(All_Hops['Activation_Barriers'])/(kB*T_KMC))*attempt_frequency
    r_time = random.uniform(0, 1) #TODO: this is not deterministic, need to keep deterministic for accuracy checks.
    Time = Time - (np.log(r_time)/np.sum(rate_consts))


    

# modified to add argument parameters and returned the energychg to prevent mutation
# also vectorized cumsum and added fast binary search
def Hop_Finder(All_Hops, kB, T_KMC) -> tuple[dict , int, float]:
    
    """
    Method to find the hop to be executed from a the dictionary All_Hops attribute of all possible hops.
    """
    if not All_Hops['Activation_Barriers']:
        return {-1:(0,0)},-1

    hop_probs = np.exp(-(np.array(All_Hops['Activation_Barriers']))/(kB*T_KMC))
    cum_probs = np.cumsum(hop_probs)
    probs = cum_probs / cum_probs[-1] #normalizes
    # probs = [np.sum(hop_probs[0:i+1])/np.sum(hop_probs) for i in range(len(hop_probs))] 
    #constructs cumulative probability distribution, we can probably improve this algorithm ourselves
    
    r = random.uniform(0, 1) #TODO: this is not deterministic, need to keep deterministic for accuracy checks.
    # rng = random.Random(seed_value=7) #this is deterministic but the rng is only used 1x so it would be hte same repeated constant r
    # r = rng.uniform(0, 1)

    idx = np.searchsorted(probs, r)
    # idx = probs.index([i for i in probs if i > r][0]) #can probably also speed up this lookup function??

    the_hop = All_Hops['Hops'][idx]
    encoding = list(the_hop.keys())[0] #why only the first encoding??

    energychg = All_Hops['Energy_Changes'][idx] #this isnt super data local 

    return the_hop, encoding, energychg
    

#will not be used in ray.remote because only 1 task
def Write_Evolution_File(Conf, Species_Lists, av_energy, Time, evolution_filename, s: int, the_hop: dict, encoding: int):            

    """
    Method to write the evolution file for tracking the hop information, average energy and time associated with the hop.

    Args:
        s (int): current KMC step number
        the_hop (dict): information for regarding the hop occuring
        encoding (int): code for the type of hop occuring
    """
    
    if s%20==0:
        
        print("Trajectory Step Number:  " + str(s) + "\n")
        
        Conf[s] = {
            'Mn2_l':Species_Lists['Mn2'].copy(),
            'Mn3_l':Species_Lists['Mn3'].copy(),
            'Mn4_l':Species_Lists['Mn4'].copy(),
            'Li_l':Species_Lists['Li'].copy(),
            'Ti_l':Species_Lists['Ti4'].copy(),
            'Av_Energy':av_energy, #TODO make this work
            'Hop': the_hop[encoding],
            'Encoding': encoding,
            'time':Time
        } 
        
        Custom_IO.write_pickle(Conf, evolution_filename)

    else:
        Conf[s] = {
            'Av_Energy':av_energy,
            'time':Time,
            'Hop': the_hop[encoding],
            'Encoding': encoding
        }

  
# will have to do this differently
# def Li_Vac_Updater(self, mn: int, vac: int):

#     """
#     Method to update the Li and vacancy species lists.

#     Args:
#         mn (int): index of the TM site involved in the hop
#         vac (int): index of the Vac site involved in the hop
    
#     """
    
#     idx_Vac = Species_Lists['Li_Vac'].index(vac)
#     Species_Lists['Li_Vac'][idx_Vac] = mn

#     idx_Vac = Species_Lists['Vac'].index(vac)
#     Species_Lists['Vac'][idx_Vac] = mn 


def Hop_Executer(hop_types, Species_Lists, Mn2, Mn3, Mn4, Ti4, Vac, site_encodings, occ, Li_Vac_Updater, the_hop: dict, encoding: int):
    
    """
    Method to execute the TM hop.

    Args:
        the_hop (dict): information for regarding the hop occuring
        encoding (int): code for the type of hop occuring
    """
    if encoding != -1: 
        tm, vac = the_hop[encoding]

        if encoding in hop_types["mn3_mn3"]:                              ### Mn3+ ----> Mn3+

            idx_Mn3 = Species_Lists['Mn3'].index(tm)
            Species_Lists['Mn3'][idx_Mn3] = vac

            post_hop_specie = Mn3

            if encoding in [1,7]:
                Species_Lists['Mn3_oct'].remove(tm)
                Species_Lists['Mn3_tet'].append(vac)
            elif encoding in [2,8]:
                Species_Lists['Mn3_tet'].remove(tm)
                Species_Lists['Mn3_oct'].append(vac)               

        if encoding in hop_types["mn2_mn2"]:                              ### Mn2+ ----> Mn2+

            idx_Mn2 = Species_Lists['Mn2'].index(tm)
            Species_Lists['Mn2'][idx_Mn2] = vac

            post_hop_specie = Mn2

            if encoding in [3, 9, 11]:
                Species_Lists['Mn2_oct'].remove(tm)
                Species_Lists['Mn2_tet'].append(vac)
            elif encoding in [4, 10, 12]:
                Species_Lists['Mn2_tet'].remove(tm)
                Species_Lists['Mn2_oct'].append(vac)               

        if encoding in hop_types["mn4_mn4"]:
            idx_Mn = Species_Lists['Mn4'].index(tm)
            Species_Lists['Mn4'][idx_Mn] = vac

            post_hop_specie = Mn4

        if encoding in hop_types["ti4_ti4"]:
            idx_Ti = Species_Lists['Ti4'].index(tm)
            Species_Lists['Ti4'][idx_Ti] = vac

            post_hop_specie = Ti4

        occ[tm] = site_encodings[tm].index(Vac)
        occ[vac] = site_encodings[vac].index(post_hop_specie)

        Li_Vac_Updater(tm, vac)    


def Barrier_Calculator(self, kra: float, mn: int, vac: int, end1: Species, ec: int) -> dict:

    """
    Method to calculate the migration barrier associated with a PROPOSED hop using its KRA and end point energies.

    Args:
        kra (float): Kinetically resolved activation barrier for the PROPOSED hop.
        mn (int): index of the TM site involved in the PROPOSED hop.
        vac (int): index of the Vac site involved in the PROPOSED hop.
        end1 (Species): the type of species of the TM in the final position if the PROPOSED hop were to occur.
        ec (int): code for the type of hop PROPOSED.
    """
    
    All_Hops['counter']+=1

    All_Hops['Hops'][All_Hops['counter']][ec] = mn,vac
    change = processor.compute_property_change(occ,[(mn, site_encodings[mn].index(Vac)), (vac, site_encodings[vac].index(end1))])[0]

    All_Hops['Energy_Changes'].append(change)
    barrier = (change/2)+ kra
    All_Hops['Activation_Barriers'].append(barrier)


def Redox_Center_Calculator(self, Redox_cutoff_dist=3):

    Redox_Neighbors = defaultdict(list)
    all_sites = indices['tet'] + indices['oct']

    for site1 in all_sites:
        for site2 in all_sites:
            if (site1!=site2) and (processor.structure[site1].distance(processor.structure[site2]) <= Redox_cutoff_dist):
                Redox_Neighbors[site1].append(site2)

    return Redox_Neighbors



def Mechanism_Update(self, tm, mechanism):
    
    if tm not in Hop_Mechanisms[mechanism]:
        Hop_Mechanisms[mechanism][tm]=[]





def Tet_Oct_Updater(self):

    Species_Lists["Mn3_tet"] = [x for x in Species_Lists['Mn3'] if x in indices['tet']]
    Species_Lists["Mn3_oct"] = [x for x in Species_Lists['Mn3'] if x in indices['oct']]
    Species_Lists["Mn2_tet"] = [x for x in Species_Lists['Mn2'] if x in indices['tet']]   
    Species_Lists["Mn2_oct"] = [x for x in Species_Lists['Mn2'] if x in self.indices['oct']]