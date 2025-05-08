import numpy as np
from convenientindexing import findCellFromCoords, findPosFromType

Nx = 8
Ny = 8
Nz = 8

def makeKey (coords):
    '''
    A FUNCTION TO CONVERT PYMATGEN INDEXING INTO CONVENIENT INDEXING
    USAGE: 
    frac_coords_list = []
    for site in kmc.processor.structure.sites:
        frac_coords_list.append(site.frac_coords)
    coords = np.array(frac_coords_list)
    INPUT coords AS ARGUMENT
    ---------
    FIELDS
    coords : a numpy array of dimension nsites x 3, containing the fractional position of every site 
    ---------
    RETURNS
        keytypes : a numpy array of length (12 * number of unit cells), keytypes[i] returns the type 
            of the ith site in convenient indexing
        key : a numpy array of length (12 * number of unit cells), key[i] returns the index of the ith
            site in convenient indexing
    '''
    coords[:,0] = coords[:,0] * Nx * 4
    coords[:,1] = coords[:,1] * Ny * 4
    coords[:,2] = coords[:,2] * Nz * 4
    coords = np.round(coords,0)
    coords = coords.astype(int)
    cellcoords = np.floor(coords/4)
    cellcoords = cellcoords.astype(int)
    typecoords = coords % 4
    
    keytypes = [None] * (ntets+nocts)
    key = [None] * (ntets+nocts)
    for i in range(0, ntets):
        cell = cellcoords[i]
        entry = typecoords[i]
        typing = 4
        if (entry[0] == 1):
            typing = typing + 2
        if (entry[1] == 1):
            typing = typing + 1
        if (entry[2] == 1):
            typing = typing + 4
        keytypes[i] = typing
        key[i] = findPosFromType(findCellFromCoords(cell[0],cell[1],cell[2]), typing)
    for i in range(ntets, ntets + nocts):
        cell = cellcoords[i]
        entry = typecoords[i]
        typing = 0
        if (entry[0] == 2):
            typing = typing + 0
        if (entry[1] == 2):
            typing = typing + 1
        if (entry[2] == 2):
            typing = typing + 2
        keytypes[i] = typing
        key[i] = findPosFromType(findCellFromCoords(cell[0],cell[1],cell[2]), typing)
    return keytypes, key

def speciesListToConvenient (speciesdict, key):
    '''
    FIELDS
    speciesdict : a dict containing the pymatgen indexes of different species (typically kmc.Species_Lists)
    key : a 1D of length (12 * number of unit cells) translating pymatgen indices to convenient indices
    ---------
    RETURNS
        allsites : a list of length (12 * number of unit cell) indexed in convenient indices
            allsites[i] will contain the name of the species in site i (convenient index)
    '''
    allsites = [None] * (12 * Nx * Ny * Nz)
    for spec in ['Li', 'Vac', 'Mn3', 'Mn4', 'Mn4', 'Ti4', 'Mn2']:
        for place in speciesdict[spec]:
            allsites[key[place]] = spec
    return allsites
