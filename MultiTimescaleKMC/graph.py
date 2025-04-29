import random
import numpy as np
from custom_io import Custom_IO
import matplotlib.pyplot as plt
from pymatgen.core import Species
from local_structure_details import Local_Structure_Details
import scipy.sparse as sparse

class Node():
    """ Atom node"""
    def __init__(self, index: int, specie: str, polyhedron: str):
        """
        Initialize node
        Args:
            index: site index
            species: elemental species (Mn2, Mn3, Mn4, Ti4, Li, Vac)
            polyhedron: tet or oct
        """
        self.index = index
        self.species = specie
        self.polyhedron = polyhedron
    

class Graph():
    def __init__(self, processor_file: str):
        """
        Initialization of graph
        Args:
            processor_file (str): Location of the pre-built processor file.
        """
        
        self.Li = Species.from_str("Li+")
        self.Vac = ForVac.Vacancy()
        self.Mn2 = Species.from_str("Mn2+")
        self.Mn3 = Species.from_str("Mn3+")
        self.Mn4 = Species.from_str("Mn4+")
        self.Ti4 = Species.from_str("Ti4+")
        self.O2 = Species.from_str("O2-")
        
        self.kB = 8.617*10**-5
        
        self.processor = Custom_IO.load_processor(processor_file)
        
        self.n_sites = self.processor.num_sites
        self.site_encodings = self.processor.allowed_species
        self.indices = Local_Structure_Details.struct_indicies(self.processor)
        self.nns = Local_Structure_Details.Nearest_Neighbor_Calculator(self.processor, self.indices)

        # Graph objects
    def make_graph(self,structure):
        """
        Makes graph
        """

    