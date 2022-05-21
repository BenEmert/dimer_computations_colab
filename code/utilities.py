
import itertools
import numpy as np
import eqtk

def make_nXn_dimer_reactions(n_types = 2):
    """
    Generate all pairwise reactions for n monomer species to be parsed by EQTK. Assumes symmetric heterodimers are equivalent (D12=D21)

    Parameters
    ----------
    n_types : int
        Number of monomer species. Default is 2. 
    
    Returns
    -------
    reactions : string
        Set of reactions for specified numbers of species, one reaction
        per line.
    """
    combinations = itertools.combinations_with_replacement(range(n_types), 2)
    reactions = [f'M{i+1} + M{j+1} <=> D{i+1}{j+1}\n' for i,j in combinations]
    return ''.join(reactions).strip('\n')

def make_nXn_species_names(n_types = 2):
    """
    Enumerate names of monomers and dimers for ordering stoichiometry matrix of nXn rxn network
    
    Parameters
    ----------
    n_types : int
        Number of monomer species. Default is 2. 
    
    Returns
    -------
    names : list, length (n_types(n_types+1)/2 + n_types) 
        List where each element (string) represents a species. 
    """
    monomers = [f'M{i+1}' for i in range(n_types)]
    combinations = itertools.combinations_with_replacement(range(n_types), 2)
    dimers = [f'D{i+1}{j+1}' for i,j in combinations]
    return monomers + dimers

def make_nXn_stoich_matrix(n_types = 2):
    """
    Generate stochiometry matrix for dimerization reactions of indiciated number (n_types) of monomers.
    Use for nXn dimer rxn network
    Parameters
    ----------
    n_types : int
        Number of monomer species. Default is 2. 
    
    Returns
    -------
    N : array_like, shape (n_types * (n_types-1)/2 + n_types, n_types(n_types+1)/2 + n_types)
        Array where each row corresponds to distinct dimerization reaction and each column correspond to reaction components (ordered monomers then dimers)
    """
    reactions = make_nXn_dimer_reactions(n_types=n_types)
    names = make_nXn_species_names(n_types = n_types)
    N = eqtk.parse_rxns(reactions)
    return N[names]

def number_of_dimers(n_types):
    """
    Calculate the number of distinct dimers from input number (n_types) of monomers. 
    Use for nXn rxn network
    """
    return int(n_types*(n_types+1)/2)

def number_of_heterodimers(n_types):
    """
    Calculate the number of distinct dimers from input number (n_types) of monomers. 
    Use for nXn rxn network
    """
    return int(n_types*(n_types-1)/2)
    
def number_of_species(n_types):
    """
    Calculate the number of monomers + dimers from input number (n_types) of monomers
    Use for nXn rxn network
    """
    return n_types + number_of_dimers(n_types)

def make_M0_grid(n_types=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial monomer concentrations
    
    Parameters
    ----------
    n_types : int
        Number of monomer species. Default is 2. 
    M0_min : array_like, shape (n_types,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (n_types,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (n_types,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 10.
    
    Returns
    -------
    M0 :  array_like, shape (numpy.product(n_conc), n_types) or (n_conc ** n_types, n_types)
        Each row corresponds to distinct set of monomer concentrations.
        
    Raises
    ------
    ValueError
        Incorrect size of M0_min, M0_max, or n_conc.
    """

    if np.size(M0_min) == 1 and np.size(M0_max) == 1 and np.size(num_conc) == 1:
        titration = [np.logspace(M0_min, M0_max, num_conc)]*n_types
    elif np.size(M0_min) == n_types and np.size(M0_max) == n_types and  np.size(num_conc) == n_types:
        titration = [np.logspace(M0_min[i], M0_max[i], num_conc[i]) for i in range(n_types)]
    else:
        raise ValueError('Incorrect size of M0_min, M0_max, or num_conc.')
    
    titration = np.meshgrid(*titration, indexing='ij')
    return np.stack(titration, -1).reshape(-1,n_types)

def make_C0_grid(n_types=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial concentrations. Outputs titration for monomers (see parameters below) and 0 for possible dimers
    Use for nXn rxn network

    Parameters
    ----------
    n_types : int
        Number of monomer species. Default is 2. 
    M0_min : array_like, shape (n_types,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (n_types,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (n_types,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 10.
    
    Returns
    -------
    C0 :  array_like, shape (numpy.product(n_conc), (n_types(n_types+1)/2 + n_types)) or (n_conc ** n_types, (n_types(n_types+1)/2 + n_types))
        Each row corresponds to distinct set of species concentrations.
        
    """
    num_dimers = number_of_dimers(n_types)
    M0 = make_M0_grid(n_types=n_types, M0_min=M0_min, M0_max=M0_max, num_conc=num_conc)
    return np.hstack((M0, np.zeros((M0.shape[0], num_dimers))))

def Kij_names(n_input = 2, n_accesory = 2, n_types = None, rxn_ordered = True):
    """
    Create Kij names for ordering parameters
    """
    #Check input
    if n_types is None: 
        n_types = n_input + n_accesory
    elif n_types != n_input+n_accesory:
        raise ValueError('n_types must equal n_input + n_accesory.\n Check input values')
        
    if rxn_ordered:
        return [f'K{i[0]}{i[1]}' for i in itertools.combinations_with_replacement(range(1, n_types+1), 2)]
    else:
        names = []
        #add input homodimers
        names.extend([f'K{i}{i}' for i in range(1, n_input+1)])
        #add input heterodimers
        names.extend([f'K{i[0]}{i[1]}' for i in itertools.combinations(range(1, n_input+1), 2)])
        #add input-acc heterodimers
        names.extend([f'K{i[0]}{i[1]}' for i in itertools.product(range(1, n_input+1), 
                                                                  range(n_input+1, n_input+n_accesory+1))])
        #add accessory homodimers
        names.extend([f'K{i}{i}' for i in range(n_input+1, n_input+n_accesory+1)])
        #add accessory heterodimers
        names.extend([f'K{i[0]}{i[1]}' for i in itertools.combinations(range(n_input+1, n_input+n_accesory+1), 2)])

        return names
