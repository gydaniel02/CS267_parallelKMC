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

def Tet_Oct_Updater(x, indices, tet_True):
    interstitial = "oct"
    if tet_True:
        interstitial = "tet"
    if x in indices["interstitial"]:
        return x
    # return x in indices[interstitial]

    # Species_Lists["Mn3_tet"] = [x for x in Species_Lists['Mn3'] if x in indices['tet']]
    # Species_Lists["Mn3_oct"] = [x for x in Species_Lists['Mn3'] if x in indices['oct']]
    # Species_Lists["Mn2_tet"] = [x for x in Species_Lists['Mn2'] if x in indices['tet']]   
    # Species_Lists["Mn2_oct"] = [x for x in Species_Lists['Mn2'] if x in indices['oct']]

def fill_species_list_partial(indices, species_value, n_sites):
    partial_list = [0] * n_sites
    for i in indices:
        partial_list[i] = species_value
    return partial_list


def GetHops(TM, nndict, Species_Lists, Mobile_Mntet, Mobile_MnOct, site_encodings, occ, barrier_map, processor):
    #biggest anticipated challenge is the latency in passing in refrences and memory issues
    ## https://chatgpt.com/share/6815009c-1458-8011-8fd4-e0c8713118b7
    #TODO: still need to format the return value
    # return a list of dict from metal --> vacancy, activation barriers, energy changes, and hop_mechanism
    All_Hops = {
            'counter': 0,
            'Hops': {},
            'Energy_Changes': [],
            'Activation_Barriers': []
        }
    nns = [v for v in nndict(TM) if v in Species_Lists["Vac"]]
    for V in nns:
        Hop_Mechanism = None
        result = None
        the_vacs = None

        nn_vacs = [x for x in nndict[V] if x in Species_Lists["Vac"]]
        num_vac = len(nn_vacs)

        ###I AM ASSUMING THAT NN[0] TAKES THE CLOSEST NEIGHBOR OF THE SPECIES TO BE THE CLOSEST VACANCY TO HOP INTO
        if num_vac == 0 or num_vac > 4:
            continue #move to next V
            # return Hop_Mechanism, result, the_vac    
        elif TM in Mobile_Mntet:
            FS_vacs = nn_vacs #face-sharing vacancies
            the_vacs = nn_vacs
            if (num_vac == 4):
                Hop_Mechanism = "Tri-Vac"
                result = FS_vacs
            elif (num_vac == 3):
                li_ct = len([x for x in nns[TM] if x in Species_Lists['Li']])
                if li_ct == 1:
                    Hop_Mechanism = "Di-Vac"
                    result = FS_vacs
            elif num_vac == 2:
                li_ct = len([x for x in nns[TM] if x in Species_Lists['Li']])
                if li_ct == 2:
                    Hop_Mechanism = "Di-Vac"
                    result = FS_vacs
            # return Hop_Mechanism, result, the_vac
            if Hop_Mechanism:
                hop_candidates = barrier_map[Hop_Mechanism][TM]
                
                initial_changes = [
                    (TM, site_encodings[TM].index(V)),
                    (V, site_encodings[V].index(hop_candidates.end_state))
                ]
                All_Hops["counter"]+=1
                hop_id = All_Hops['counter']
                All_Hops['Hops'][hop_id] = {hop_candidates.encoding: (TM, V)}
                change = processor.compute_property_change(occ, initial_changes)[0]
                All_Hops['Energy_Changes'].append(change)
                barrier = (change / 2) + hop_candidates.kra
                All_Hops['Activation_Barriers'].append(barrier)
        else: #refers to octahedral transition metals Mn4 and Ti4
            FS_TMs = [x for x in nns[TM] if x in Mobile_MnOct]
            num_FS_TMs = len(FS_TMs)
            the_vacs = nn_vacs
            if (num_vac==3) and (num_FS_TMs==1):    #Trivac
                Hop_Mechanism = "Tri-Vac"
                result = FS_TMs
            elif num_vac==2:
                li_ct = len([x for x in nns[TM] if x in Species_Lists['Li']])
                if (li_ct==1) and (num_FS_TMs==1):
                    Hop_Mechanism = "Di-Vac"
                    result = FS_TMs
            elif num_vac == 1:
                li_ct = len([x for x in nns[TM] if x in Species_Lists['Li']])
                if (li_ct==2) and (num_FS_TMs==1):
                    Hop_Mechanism = "Mono-Vac"
                    result = FS_TMs
                # return Hop_Mechanism, result, the_vac
            if Hop_Mechanism:
                hop_candidates = barrier_map[Hop_Mechanism][TM]
                
                initial_changes = [
                    (TM, site_encodings[TM].index(V)),
                    (V, site_encodings[V].index(hop_candidates.end_state))
                ]
                All_Hops["counter"]+=1
                hop_id = All_Hops['counter']
                All_Hops['Hops'][hop_id] = {hop_candidates.encoding: (TM, V)}
                change = processor.compute_property_change(occ, initial_changes)[0]
                All_Hops['Energy_Changes'].append(change)
                barrier = (change / 2) + hop_candidates.kra
                All_Hops['Activation_Barriers'].append(barrier)

    
    # Barrier Calculator
    return All_Hops
    # if All_Hops["counter"]>0: #then this hop exists

        

    #     return All_Hops[]

#the issue is that multiple transition metals may have different hop mechanisms, that is why they looped through the vacancies first
        
                

# #TODO: break apart the issues in what barrier_calculator, barrier_map, Hop_mechanisms look like~
## TODO: also make the kra data local
# 
#  for mechanism in Hop_Mechanisms:
#             for mn in Hop_Mechanisms[mechanism]:
#                 for vac in Hop_Mechanisms[mechanism][mn]:
#                     for cation_description, hop_info in barrier_map[mechanism].items():
#                         if mn in Species_Lists[cation_description]:  # Dynamically fetch the correct set (Mn3_oct, etc.) #globals()[mn_type]
#                             Barrier_Calculator(hop_info.kra, int(mn), vac, hop_info.end_state, hop_info.encoding)
#                             break  # Exit the loop once a match is found
    # Get NN list
    # Count number of vacancies, clarify type of vacancies?





def main(num_cpus: int, T_KMC: int, traj_steps: int, processor_file: str, EVOFILE: str, RT_CMC_results_file:str = "Delithiated_RT_DRX.pickle"):
    """You're correct that if you launch your Python file as a job (e.g., from a shell script, SLURM, or a job queue), you can't pass num_cpus directly into the file via Ray decorators at runtime unless you structure your code to accept it as input before defining the @ray.remote functions.

    But there is a clean way to still do this: use ray.remote as a function, not as a decorator. This lets you pass num_cpus dynamically at runtime, even from a job.
    """
    ray.init()


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

    #Update Species_Lists to have tetrahedra and octahedral sites labeled
    remote_Tet_Oct_Updater = ray.remote(num_cpus=num_cpus)(Tet_Oct_Updater)
    remote_Tet_Oct_Updater = ray.remote(num_cpus=num_cpus)(Tet_Oct_Updater)

    # Map species names to their lists
    species_tasks = {
        "Mn3": Species_Lists["Mn3"],
        "Mn2": Species_Lists["Mn2"],
        "Vac": Species_Lists["Vac"]
    }

    # Prepare tasks and labels dynamically
    futures = {}
    for species, items in species_tasks.items():
        for is_tet in [True, False]:
            label = f"{species}_{'tet' if is_tet else 'oct'}"
            futures[label] = [remote_Tet_Oct_Updater.remote(i, indices, is_tet) for i in items]

    # Gather results and filter out NaNs
    for label, future_list in futures.items():
        result = np.array(ray.get(future_list))
        Species_Lists[label] = result[~np.isnan(result)]

    # Mn3list = Species_Lists["Mn3"]
    # Mn2list = Species_Lists["Mn2"]
    # Vaclist = Species_Lists["Vac"]
    # future_Mn3tet = [remote_Tet_Oct_Updater.remote(i, indices, True) for i in Mn3list]
    # future_Mn3oct = [remote_Tet_Oct_Updater.remote(i, indices, False) for i in Mn3list]
    # future_Mn2tet = [remote_Tet_Oct_Updater.remote(i, indices, True) for i in Mn2list]
    # future_Mn2oct = [remote_Tet_Oct_Updater.remote(i, indices, False) for i in Mn2list]
    # future_Vactet = [remote_Tet_Oct_Updater.remote(i, indices, True) for i in Vaclist]
    # future_Vacoct = [remote_Tet_Oct_Updater.remote(i, indices, False) for i in Vaclist]
    # Mn3_tet = np.array(ray.get(future_Mn3tet))
    # Mn3_oct = np.array(ray.get(future_Mn3oct))
    # Mn2_tet = np.array(ray.get(future_Mn2tet))
    # Mn2_oct = np.array(ray.get(future_Mn2oct))
    # Vac_tet = np.array(ray.get(future_Vactet))
    # Vac_oct = np.array(ray.get(future_Vacoct))
    # Species_Lists["Mn3_tet"] = Mn3_tet[~np.isnan(Mn3_tet)]
    # Species_Lists["Mn3_oct"] = Mn3_oct[~np.isnan(Mn3_oct)]
    # Species_Lists["Mn2_tet"] = Mn2_tet[~np.isnan(Mn2_tet)]
    # Species_Lists["Mn2_oct"] = Mn2_oct[~np.isnan(Mn2_oct)]
    # Species_Lists["Vac_tet"] = Vac_tet[~np.isnan(Vac_tet)]
    # Species_Lists["Vac_oct"] = Vac_oct[~np.isnan(Vac_oct)]

    species_type_map = {
        "Li": Li,
        "Vac": Vac,
        "Mn3": Mn3,
        "Mn4": Mn4,
        "Ti4": Ti4,
        "Mn2": Mn2,
        "O2": O2
    }

    unwanted_lists = {'Li_Vac', "Mn3_tet", "Mn3_oct", "Mn2_tet", "Mn2_oct"}

    # Launch Ray tasks
    futures = []
    for species, value in species_type_map.items():
        if species in Species_Lists and species not in unwanted_lists:
            futures.append(fill_species_list_partial.remote(Species_Lists[species], value, n_sites))

    # Gather partial results
    partial_lists = ray.get(futures)

    # Reduce to a single species list
    species_list = [0] * n_sites
    for partial in partial_lists:
        for i in range(n_sites):
            if partial[i] != 0:
                species_list[i] = partial[i]

    # Encode occupancy
    occ = processor.encode_occupancy(species_list)

    # #Reset occupancies
    # spec_type = [Li, Vac, Mn3, Mn4, Ti4, Mn2, O2] #TODO: this is only used for Occupancy Resetter
    # # occ = Occupancy_Resetter(processor, Species_Indices(Species_Lists))
    # species_list = [0 for x in range(n_sites)]
    # unwanted_lists = ['Li_Vac', "Mn3_tet", "Mn3_oct", "Mn2_tet", "Mn2_oct"]
    # spec_indices = [Species_Lists[species] for species in Species_Lists if species not in unwanted_lists]
    # for idx, si in enumerate(spec_indices):
    #     for ind in si:
    #         species_list[ind] = spec_type[idx]
    # occ = processor.encode_occupancy(species_list)

    energy = processor.compute_property(occ)[0] #TODO: subclassing of processor
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
        'Activation_Barriers':np.array([]),
        'Energy_Changes':np.array([])
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

    # do any ray remote initializations here
    remote_HeavyGetHops = ray.remote(num_cpus=num_cpus)(GetHops)
    remote_LightGetHops = ray.remote(num_cpus = num_cpus/4)(GetHops)

    Fast_Processes = Fast_Processes_MC()
    for s in range(traj_steps):    
        Fast_Processes.Select_Fast_Configuration(s)      #JUST TO PROVE THAT WE HAVE DONE OUR DUE DILLIGENCE
        # Formerly Hop(s)
        # Get all possible hops
        Mobile_Mn2_tet = Species_Lists["Mn2_tet"]
        Mobile_Mn2_oct = Species_Lists["Mn2_oct"]
        Mobile_Mn3_tet = Species_Lists["Mn3_tet"]
        Mobile_Mn3_oct = Species_Lists["Mn3_oct"]
        Mn4_oct = Species_Lists["Mn4_oct"] #TODO
        Ti4_oct = Species_Lists["Ti4_oct"]
        Mobile_Mntet = Mobile_Mn2_tet + Mobile_Mn3_tet
        Mobile_MnOct = Mobile_Mn2_oct + Mobile_Mn3_oct
        future_heavyhops = [remote_HeavyGetHops(TM, nns, Species_Lists, Mobile_Mntet, Mobile_MnOct, site_encodings, occ, barrier_map, processor) for TM in Mobile_Mn2_tet + Mobile_Mn2_oct + Mobile_Mn3_tet + Mobile_Mn3_oct]
        future_lighthops = [remote_LightGetHops(TM, nns, Species_Lists, Mobile_Mntet, Mobile_MnOct, site_encodings, occ, barrier_map, processor) for TM in Mn4_oct + Ti4_oct]
        both_futures = future_heavyhops + future_lighthops
        #TODO: decide if dictionary aggregation after all futures is done is more efficient than periodically checking
        

        for future in both_futures:
            result = future.result()
            # Merge All_Hops
            if result["counter"]>0:

                current_counter = All_Hops["counter"]
                reindexed_hops = {}
                for i, (old_key, hop_info) in enumerate(result['Hops'].items(), start=1):
                    new_key = current_counter + i
                    reindexed_hops[new_key] = hop_info
                
                # Update base with reindexed entries
                All_Hops['Hops'].update(reindexed_hops)

                All_Hops["counter"] += result["counter"]
                All_Hops["Activation_Barriers"] = np.concatenate(All_Hops["Activation_Barriers"], result["Activation_Barriers"])
                All_Hops["Energy_Changes"]= np.concatenate(All_Hops["Energy_Changes"], result["Energy_Changes"])
                All_Hops["Hops"]

        # Time Advancement
        attempt_frequency = 1e13
        hop_probs = np.exp(-np.array(All_Hops['Activation_Barriers'])/(kB*T_KMC))
        r_time = random.uniform(0, 1) #TODO: this is not deterministic, need to keep deterministic for accuracy checks.
        Time = Time - (np.log(r_time)/np.sum(attempt_frequency * hop_probs))

        # Hop Finder
        if not All_Hops['Activation_Barriers']:
            return {-1:(0,0)},-1
        
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

        energy += All_Hops['Energy_Changes'][idx] #this isnt super data local 

        #Write Evolution File
        if s%20==0:
            print("Trajectory Step Number:  " + str(s) + "\n")
            Conf[s] = {
                'Mn2_l':Species_Lists['Mn2'].copy(),
                'Mn3_l':Species_Lists['Mn3'].copy(),
                'Mn4_l':Species_Lists['Mn4'].copy(),
                'Li_l':Species_Lists['Li'].copy(),
                'Ti_l':Species_Lists['Ti4'].copy(),
                'Av_Energy':av_energy,
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


        #Hop executer
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

            # update species lists
            idx_Vac = Species_Lists['Li_Vac'].index(vac)
            Species_Lists['Li_Vac'][idx_Vac] = tm

            idx_Vac = Species_Lists['Vac'].index(vac)
            Species_Lists['Vac'][idx_Vac] = tm 



