
import itertools
import numpy as np
import scipy.stats
from eqtk import parse_rxns
import math
from pdb import set_trace as bp
from time import strftime, gmtime


def sec2friendly(t):
    return strftime("%H:%M:%S", gmtime(t))

def make_opt_settings(x):
    '''strips _* from dictionary keywords'''
    xnew = {key.split('_')[0]: x[key] for key in x}
    return xnew

def sample_concentrations(n_dims, n_samples, lb, ub, centered=True, seed=42, do_power=True):
    lhs_sampler =  scipy.stats.qmc.LatinHypercube(d=n_dims, centered=centered, seed=seed)
    param_sets = lhs_sampler.random(n=n_samples)
    param_sets = scipy.stats.qmc.scale(param_sets, lb, ub)
    if do_power:
        param_sets = np.power(10, param_sets)
    return param_sets

def make_nXn_dimer_reactions(m = 2):
    """
    Generate all pairwise reactions for n monomer species to be parsed by EQTK.
    Assumes symmetric heterodimers are equivalent (D12=D21)

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.

    Returns
    -------
    reactions : string
        Set of dimerization reactions for specified numbers of monomers, one reaction
        per line.
    """
    combinations = itertools.combinations_with_replacement(range(m), 2)
    reactions = [f'M{i+1} + M{j+1} <=> D{i+1}{j+1}\n' for i,j in combinations]
    return ''.join(reactions).strip('\n')

def make_nXn_species_names(m = 2):
    """
    Enumerate names of monomers and dimers for ordering stoichiometry matrix of nXn rxn network

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.

    Returns
    -------
    names : list, length (m(m+1)/2 + m)
        List where each element (string) represents a reacting species.
    """
    monomers = [f'M{i+1}' for i in range(m)]
    combinations = itertools.combinations_with_replacement(range(m), 2)
    dimers = [f'D{i+1}{j+1}' for i,j in combinations]
    return monomers + dimers

def make_nXn_stoich_matrix(m = 2):
    """
    For the indicated number of monomers (m), generate stochiometry matrix for dimerization reactions.
    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.

    Returns
    -------
    N : array_like, shape (m * (m-1)/2 + m, m(m+1)/2 + m)
        Array where each row corresponds to distinct dimerization reaction
        and each column corresponds to a reaction component (ordered monomers then dimers)
    """
    reactions = make_nXn_dimer_reactions(m=m)
    names = make_nXn_species_names(m = m)
    N = parse_rxns(reactions)
    return N[names].to_numpy()

def number_of_dimers(m):
    """
    Calculate the number of distinct dimers from input number (m) of monomers.
    """
    return int(m*(m+1)/2)

def number_of_heterodimers(m):
    """
    Calculate the number of distinct heterodimers from input number (m) of monomers.
    """
    return int(m*(m-1)/2)

def number_of_species(m):
    """
    Calculate the number of monomers + dimers from input number (m) of monomers
    """
    return m + number_of_dimers(m)

def make_M0_grid(m=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial monomer concentrations

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.
    M0_min : array_like, shape (m,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all monomers. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (m,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all monomers. Default is -3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (m,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all monomers. Default is 10.

    Returns
    -------
    M0 :  array_like, shape (numpy.product(n_conc), m) or (n_conc ** m, m)
        Each row corresponds to distinct set of monomer concentrations.

    Raises
    ------
    ValueError
        Incorrect size of M0_min, M0_max, or n_conc.
    """

    if np.size(M0_min) == 1 and np.size(M0_max) == 1 and np.size(num_conc) == 1:
        titration = [np.logspace(M0_min, M0_max, num_conc)]*m
    elif np.size(M0_min) == m and np.size(M0_max) == m and  np.size(num_conc) == m:
        titration = [np.logspace(M0_min[i], M0_max[i], num_conc[i]) for i in range(m)]
    else:
        raise ValueError('Incorrect size of M0_min, M0_max, or num_conc.')

    titration = np.meshgrid(*titration, indexing='ij')
    return np.stack(titration, -1).reshape(-1,m)

def make_C0_grid(m=2, M0_min=-3, M0_max=3, num_conc=10):
    """
    Construct grid of initial monomer and dimer concentrations.
    Initial dimer concentrations set to 0.

    Parameters
    ----------
    m : int
        Number of monomer species. Default is 2.
    M0_min : array_like, shape (m,) or (1,)
        Lower limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^-3.
    M0_max : array_like, shape (m,) or (1,)
        Upper limit of the monomer concentration in log10 scale.
        Scalar values set the same limit for all ligands. Default is -3,
        corresponding to a lower limit of 10^3.
    num_conc : array_like, shape (m,) or (1,)
        Number of concentrations for each monomer, sampled logarithmically.
        Scalar values set the same limit for all ligands. Default is 10.

    Returns
    -------
    C0 :  array_like, shape (numpy.product(n_conc), (m(m+1)/2 + m)) or (n_conc ** m, (m(m+1)/2 + m))
        Each row corresponds to distinct set of species concentrations.

    """
    num_dimers = number_of_dimers(m)
    M0 = make_M0_grid(m=m, M0_min=M0_min, M0_max=M0_max, num_conc=num_conc)
    return np.hstack((M0, np.zeros((M0.shape[0], num_dimers))))

def make_K_matrix(K_vec, m):
    """
    Create Kij names for ordering parameters
    """
    K = np.zeros((m,m))
    cc = -1
    for i in range(m):
        for j in range(m-i):
            cc += 1
            K[i,j+i] = K_vec[cc]
    return K

def make_Kij_names(n_input = 2, n_accesory = 2, m = None, rxn_ordered = True):
    """
    Create Kij names for ordering parameters
    """
    #Check input
    if m is None:
        m = n_input + n_accesory
    elif m != n_input+n_accesory:
        raise ValueError('m must equal n_input + n_accesory.\n Check input values')

    if rxn_ordered:
        return [f'K{i[0]}{i[1]}' for i in itertools.combinations_with_replacement(range(1, m+1), 2)]
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

def pointsInCircum(x, y, r=0.5, n=99):
    return np.array([[x+np.cos(2*np.pi/n*i)*r,y+np.sin(2*np.pi/n*i)*r] for i in range(0,n+1)])

def interp_points(x_pos, y_pos, n = 3):
    return np.stack((np.linspace(x_pos[0], x_pos[1], n), np.linspace(y_pos[0], y_pos[1], n)), axis =1)

def get_poly_vertices(n, r = 1, dec = 3, start = math.pi/4):
    """
    Get x and y coordinates of n-polygon with radius r.
    """
    #This could be broadcast with numpy
    #i.e. x = r * np.cos(2*np.pi * np.arange(1,n+1)/n)
    #but I think it's easier to follow as list comprehension
    x = np.array([round(r * math.cos(2*math.pi*i/n+start), dec) for i in range(n)])
    y = np.array([round(r * math.sin(2*math.pi*i/n+start), dec) for i in range(n)])
    return x,y
