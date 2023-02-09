import os, sys
import itertools
import pickle
import numpy as np
import scipy.stats
from sklearn.linear_model import Ridge
from eqtk import parse_rxns
import math
from time import strftime, gmtime, time
import eqtk
import json
import networkx as nx


from pdb import set_trace as bp

import itertools

class DimerNets:
    def __init__(self, n_monomers=3, n_input=1):

        assert(n_monomers > n_input)

        self.n_input = n_input
        self.n_monomers = n_monomers
        self.n_dimers = n_monomers*(n_monomers-1) / 2
        self.n_acc = n_monomers - n_input

        self.acc_inds = np.arange(n_input,n_monomers)
        self.x_input_inds = np.arange(n_input)

        self.Knames = np.array(make_Kij_names(n_input=self.n_input, n_accesory=self.n_acc))
        self.N = make_nXn_stoich_matrix(self.n_monomers)
        self.num_rxns = self.N.shape[0]


    def generate_random_params(self, K_bounds=(-5,7), c0_acc_bounds=(-3,3)):

        K = np.random.uniform(low=K_bounds[0], high=K_bounds[1], size=(self.num_rxns))
        c0_acc = np.random.uniform(low=c0_acc_bounds[0], high=c0_acc_bounds[1], size=(self.n_acc))

        # convert parameters from log to linear
        self.K = np.float_power(10, K)
        self.c0_acc = np.float_power(10, c0_acc)

    def sparsify(self, num_edges=110):

        m = self.n_monomers

        # Create graph
        G = nx.Graph()
        G.add_node(0)
        # Add nodes one at a time, connecting them with a random edge, to guarantee connectedness
        for monomer in range(1,m):
            G.add_node(monomer)
            G.add_edge(np.random.choice([x for x in G.nodes if x!=monomer]),monomer)

        # Add extra edges
        all_possible_edges = list(itertools.combinations(range(m),r=2))
        possible_new_edges = [x for x in all_possible_edges if x not in list(G.edges)]
        num_edges = 110
        chosen_new_edge_ids = np.random.choice(np.arange(len(possible_new_edges)),size=(num_edges-(m-1),),replace=False)
        chosen_new_edges = [possible_new_edges[i] for i in chosen_new_edge_ids]
        G.add_edges_from(chosen_new_edges)

        # Now export to list of K values
        edge_key = list(itertools.combinations_with_replacement(range(m),r=2)) # List of monomer combinations
        edge_indices_inG = [edge_key.index(x) for x in G.edges] # Indices of K matrix to fill

        # This generates a list of dimer indices, out of all possible combinations of monomers, that are actually used in the network

        N = self.N
        K_mask = np.zeros(N.shape[0])
        K_mask[edge_indices_inG] = 1
        dimers_to_delete = np.array(np.where(K_mask==0))
        N2 = np.delete(N, dimers_to_delete, axis=0)
        N3 = np.delete (N2, m + dimers_to_delete, axis=1)

        # assign new values to K
        self.K = self.K[edge_indices_inG]
        self.N = N3

        assert(self.N.shape[0]==num_edges)
        assert(self.N.shape[1]==(num_edges+m))

        print('Reduced stoichiometry matrix from {} to {}'.format(N.shape, self.N.shape))

    def make_dimer_features(self, x_input):
        C0 = np.zeros((x_input.shape[0], self.N.shape[1]))
        C0[:,self.acc_inds] = self.c0_acc # each row should get c0_acc
        C0[:,self.x_input_inds] = x_input

        sols = eqtk.solve(c0=C0, N=self.N, K=self.K)
        dimers = sols[:,self.n_monomers:]

        return dimers


def my_regressor(X, y, alpha=1e-5, positive=False):
    clf = Ridge(alpha=alpha, fit_intercept=False, positive=positive)
    # note on solvers: solver{‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’, ‘lbfgs’}, default=’auto’
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html

    clf.fit(X, y)
    y_pred = clf.predict(X)
    residuals = y_pred - y

    print('\tTraining MSE:', np.mean((residuals)**2))
    print('\tTraining R^2:', clf.score(X,y))

    return clf, y_pred, residuals


def make_tanh_features(n_input_dim, n_features, x_input):
    '''n_input_dim [scalar]: This is the dimension of the input space.
       n_features [scalar]: This is the number of random features to generate.
       x_input [vector]: "N_data_points" x n_input_dim. This is the set of inputs (i.e. discretizing a domain)
    '''
    # Generate random tanh features
    A = np.random.randn(n_features, n_input_dim)
    b = np.random.uniform(low=-1, high=1, size=n_features)
    F = np.tanh((A @ x_input.T).T + b)
    return F

def dict_combiner(mydict):
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v)) for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class DotDict(dict):
    """dot.notation access to dictionary attributes
    From https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    def __getattr__(*args):
        # Allow nested dicts
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __dir__ = dict.keys

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def dict_to_file(mydict, fname):
	dumped = json.dumps(mydict, cls=NumpyEncoder)
	with open(fname, 'w') as f:
		json.dump(dumped, f, indent=3)
	return

def file_to_dict(fname):
    with open(fname) as f:
        my_dict = json.load(f)
    return my_dict

def load_data(fname='input.pkl'):
    with open(fname, "rb") as f:
        return pickle.load(f)

def dump_data(out, fname='output.pkl', to_dict=True):
    with open(fname, 'wb') as f:
        pickle.dump(out, f)
    try:
        dict_to_file(out, os.path.splitext(fname)[0]+'.txt')
    except:
        pass
    return

def make_new_dir(x):
    c = 0
    while c >=0 :
        try:
            newdir = x + '_{}'.format(c)
            os.makedirs(newdir)
            c = -1
        except:
            c += 1
    return newdir

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

def get_diag_inds(n_input = 2, n_accesory = 2, m = None, rxn_ordered = True):
    '''Get indices for the diagonal elements of an upper triangular matrix
    when the whole upper-triangular matrix is re-written as flattened 1d-array.'''
    Knames = np.array(make_Kij_names(n_input, n_accesory, m, rxn_ordered))
    ind_list = []
    for j in range(m):
        Knm = 'K{}{}'.format(j+1,j+1)
        ind_list.append(np.where(Knames==Knm))

    return np.array(ind_list).squeeze()

def make_K_matrix(K_vec, m):
    """
    Create Kij names for ordering parameters
    """
    K_vec = np.squeeze(np.array(K_vec)).tolist() # be flexible to array / list inputs
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

def sort_K_ascending(K, m, n_input=1):
    # re-indexes K matrix so that diagonals (after inputs) are ordered in descending.

    if K.ndim==1:
        K = make_K_matrix(K, m)
        reshape = True
    else:
        reshape = False

    # make K symm
    for i in range(m):
        for j in range(i,m):
            K[j,i] = K[i,j]

    # get the non-input reordered indices
    new_inds = np.flip(np.argsort(np.diag(K)[n_input:])) + n_input

    # concatenate w/ the permanent input indices
    new_inds = np.hstack((np.arange(n_input), new_inds))

    Knew = np.copy(K)
    for i in range(m):
        for j in range(m):
            Knew[i,j] = K[new_inds[i], new_inds[j]]

    if reshape:
        Knew = Knew[np.triu_indices(m)]

    return Knew, new_inds
