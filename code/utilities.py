import os, sys
import pandas as pd
import itertools
import pickle
import numpy as np
import scipy.stats
import eqtk
from eqtk import parse_rxns
import math
from time import strftime, gmtime
import json

# Plotting
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from pdb import set_trace as bp

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



################################################
# Plotting: Feed-forward-like network diagram
################################################


def semicircle_points(x, y, n_points, r=0.5):
    step = math.pi/(n_points-1)
    return np.array([[round(x+r*np.cos(theta),3),round(y+r*np.sin(theta),3)] for theta in np.arange(math.pi/2,(3*math.pi/2)+step,step)])

def make_network_nodes(n, n_input, acc_levels, titrated_param = None):
    species = make_nXn_species_names(n)
    num_dimers = number_of_dimers(n)
    #For now, assuming single layer network
    # Which layer each species is in
    layers = ['layer 1' if 'M' in x else 'layer 2' for x in species]
    # Whether each species is input, accessory, or dimer
    species_type = ['input'] * n_input + ['accessory'] * (n - n_input) + ['dimer'] * num_dimers

    # Node sizes (abundances)
    weights = np.hstack(([1] * n_input, acc_levels , [1]*num_dimers))

    param_titrated = np.array(species) == titrated_param

    # Df of node info
    node_df = pd.DataFrame({'species': species,  'species_type': species_type,
                            'layer': layers, 'weight': weights,
                           'titrated_param': param_titrated})
    node_df = pd.concat([node_df, 
                         pd.DataFrame({'species': ['M1','out'], 
                                       'species_type': ['input','output'],
                                       'layer': ['input','output'],
                                       'weight': [1,1],
                                       'titrated_param': [False,False]})])

    node_df.reset_index(inplace=True, drop = True)
    # Set x positions
    circle_coords = semicircle_points(x=1, y=0, n_points=n, r=1)
    node_df['x'] = node_df.apply(lambda row: 0 if row['layer']=='input' else\
                                 (2 if row['layer']=='layer 2' else (\
                                 (3 if row['layer']=='output' else (\
                                 (1-(0.1*n*(1-circle_coords[species.index(row['species'])][0])) if row['layer']=='layer 1' else None)
        )))),axis=1)
    #Set y positions
    node_df['position'] = node_df.groupby('layer').cumcount() # Rank within layer
    node_df['position2'] = node_df.groupby(['layer'], group_keys=False).apply(lambda x: x.position - x.position.median()) # Position relative to middle of layer
    node_df['y'] = node_df.apply(lambda row: row['position2'] if row['layer'] in ['input','layer 2','output']  else\
                                 (circle_coords[species.index(row['species'])][1]*((n-1)/2) if row['layer']=='layer 1' else None),axis=1
        )
    return node_df

def make_Kij_edge_df(n, node_df, K, edge_scale = 1, titrated_param = None):
    x_start = []
    x_end = []
    y_start = []
    y_end = []
    param = []
    weight = []
    for i, comb in enumerate(itertools.combinations_with_replacement(range(1,n+1), 2)):
        # For combinations of monomers
        if comb[0] == comb[1]:
            # For homodimers
            param.append(f'K{comb[0]}{comb[1]}')
            weight.append(K[i])
            start = f'M{comb[0]}'
            end = f'D{comb[0]}{comb[1]}'
            x_start.append(node_df.query('layer == "layer 1" & species == @start')['x'].item())
            x_end.append(node_df.query('layer == "layer 2" & species == @end')['x'].item())
            y_start.append(node_df.query('layer == "layer 1" & species == @start')['y'].item())
            y_end.append(node_df.query('layer == "layer 2" & species == @end')['y'].item())
        else:
            # For heterodimers
            param.extend([f'K{comb[0]}{comb[1]}']*3)
            weight.extend([K[i]]*3)
            start1 = f'M{comb[0]}'
            start2 = f'M{comb[1]}'
            end = f'D{comb[0]}{comb[1]}'
            x_start.extend([node_df.query('layer == "layer 1" & species == @start1')['x'].item()]*2) # Including inter-accessory edge
            x_start.append(node_df.query('layer == "layer 1" & species == @start2')['x'].item())
            x_end.append(node_df.query('layer == "layer 2" & species == @end')['x'].item())
            x_end.append(node_df.query('layer == "layer 1" & species == @start2')['x'].item()) 
            x_end.append(node_df.query('layer == "layer 2" & species == @end')['x'].item())
            y_start.extend([node_df.query('layer == "layer 1" & species == @start1')['y'].item()]*2) # Including inter-accessory edge
            y_start.append(node_df.query('layer == "layer 1" & species == @start2')['y'].item())
            y_end.append(node_df.query('layer == "layer 2" & species == @end')['y'].item())
            y_end.append(node_df.query('layer == "layer 1" & species == @start2')['y'].item())
            y_end.append(node_df.query('layer == "layer 2" & species == @end')['y'].item())


    param_titrated = np.array(param) == titrated_param
    edge_df = pd.DataFrame({'x_start':x_start, 'x_end':x_end, 
                            'y_start':y_start, 'y_end':y_end, 
                            'Kij_name':param, 'weight': np.array(weight)*edge_scale,
                           'titrated_param': param_titrated})
    return edge_df


def make_input_edge_df(n, input_species, node_df):
    x_start = []
    x_end = []
    y_start = []
    y_end = []
    for input_i in range(n_input):
        x_start.append(node_df.query('layer == "input" & species_type == "input"')['x'].reset_index(drop=True)[input_i])
        x_end.append(node_df.query('layer == "layer 1" & species_type == "input"')['x'].reset_index(drop=True)[input_i])
        y_start.append(node_df.query('layer == "input" & species_type == "input"')['y'].reset_index(drop=True)[input_i])
        y_end.append(node_df.query('layer == "layer 1" & species_type == "input"')['y'].reset_index(drop=True)[input_i])

    input_df =  pd.DataFrame({'x_start': x_start, 'x_end': x_end, 
                               'y_start': list(y_start), 'y_end': list(y_end), 
                               'name':input_species})
    return input_df


def make_out_edge_df(n, n_layers, out_weights, node_df, edge_scale):
    """
    Note, out_species and out_weights must be in the same order as node_df.species. 
    Otherwise, need to sort out_weights. 
    """
    x_start = [n_layers+1]*len(out_weights)
    x_end = [n_layers+2]*len(out_weights)

    y_start = node_df.query('layer == @final_layer')['y']
    y_end = [0]*len(out_weights) #assuming 1 output node for now
    output_df =  pd.DataFrame({'x_start': x_start, 'x_end': x_end, 
                               'y_start': list(y_start), 'y_end': list(y_end), 
                                'weight':np.array(out_weights)*edge_scale
                              })
    return output_df


def make_forward_network_plots(n, n_input, univs_to_plot, param_sets=None, param_set_file=None, out_weights = None, \
                               ncols = 1, node_scales = {'input':2,'accessory':2,'dimer':2,'output':2},\
                               edge_scales={'input':4,'K':50,'output':15}, \
                               rotate = False, figsize = (12,12),K_input_cmap = 'Set1'):
    """
    Load a subset of networks from a parameter file and plot them in the style of a feedforward neural network.
    Note that this representation is partly inaccurate, as it suggests that the monomer concentrations are fixed,
    whereas in reality one free monomer concentration depends on all the starting monomer concentrations.
    
    Parameters
    --------------
    n: Int
        Number of total monomers in the network.
    n_input: Int
        Number of input monomers
    univ_to_plots: array-like
        Array of which universes (parameter sets) to plot.
    param_sets: Array-like, shape (n_univ, n_parameters)
        Array of parameters to plot. Alternatively, can use:
        param_set_file: Str
            File path to the .npy file containing the saved parameter sets.
    out_weights: array-like or None
        Weights describing the output activity of each dimer species. Either specify a weight
        for ALL dimers (including zeros for inactive dimers) or pass None for no output layer.
    ncols: Int
        Number of columns to use for multiple subplots.
    node_scales: Dict
        Scale factors for the sizes of the nodes. Keys: ['input','accessory','dimer','output']
    edge_scales: Dict
        Scale factors for the widths of the edges. Keys: ['input','K','output']
    rotate: Bool
        Whether to rotate the plots 90 degrees clockwise to be top-down.
    figsize: Tuple of len 2
        Width and height of the final figure in inches.
    K_input_cmap: Str or matplotlib colormap
        Matplotlib colormap, or name of colormap, to use to color dimerization K's.
    
    Returns:
        fig, axs: Created plot
    """
    if out_weights is None:
        plot_output_layer = False
    else:
        plot_output_layer = True
    if param_sets is None:
        param_sets = np.load(param_set_file)
    param_sets = param_sets[univ_to_plot,:]
    num_plots = len(univ_to_plot)
    species_names = np.array(make_nXn_species_names(n))
    dimer_names = species_names[n:]
    Kij_labels = make_Kij_names(n_input = n_input, n_accesory=(n-n_input))
    num_rxns = len(Kij_labels)
    
    #Make dataframe containing node positions. Color accessory monomer nodes and scale size by parameter value
    
    node_df_list = [0]*num_plots
    for i in range(num_plots):
        #scale acc monomer weights
        acc_weights = np.log10(param_sets[i,num_rxns:]) + 3
        node_df_list[i] = make_network_nodes(n=n, n_input=n_input, acc_levels=acc_weights)
    node_df_combined = pd.concat(node_df_list, keys=univ_to_plot).reset_index()
    node_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)

    edge_df_list = [0]*num_plots
    for i in range(num_plots):
        #scale edge weights
        edge_weights = np.log2(param_sets[i,:num_rxns]/1e-6)
        edge_df_list[i] = make_Kij_edge_df(n, node_df_list[0], edge_weights, 
                                           edge_scale=1, titrated_param = None)
    edge_df_combined = pd.concat(edge_df_list, keys=univ_to_plot).reset_index()
    edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)

    input_df = make_input_edge_df(n=n, input_species=species_names[0:n_input], node_df=node_df_list[0])
    input_df_combined = pd.concat([input_df]*num_plots, keys=univ_to_plot).reset_index()
    input_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    
    if plot_output_layer:
        if np.ndim(out_weights) == 1:
            output_df = make_out_edge_df(n=n, n_layers=1, 
                                     out_weights=out_weights, 
                                     node_df=node_df_list[0], edge_scale=1)
            output_df_combined = pd.concat([output_df]*num_plots, keys=univ_to_plot).reset_index()
        elif np.ndim(out_weights) == 2:
            output_list = [0]*num_plots
            for univ in range(num_plots):
                output_list[univ] = make_out_edge_df(n=n, n_layers=1, 
                                     out_weights=out_weights[univ,:], 
                                     node_df=node_df_list[0], edge_scale=1)
            output_df_combined = pd.concat(output_list, keys=univ_to_plot).reset_index()
        
        output_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)

    if len(univ_to_plot)==1:
        ncols = 1
    fig, axs = plt.subplots(nrows=math.ceil(len(univ_to_plot)//ncols), ncols=ncols,figsize=figsize)
    if len(univ_to_plot)==1:
        axs = [axs]
    color_types ={
        'input':'black',
        'accessory':'red',
        'dimer':'#636363',
        'output':'blue'
    }     

    if type(K_input_cmap)==str:
        K_cmap = cm.get_cmap(K_input_cmap)
    else:
        K_cmap = K_input_cmap

    for univ, ax in zip(list(univ_to_plot),axs):
        ax.axis('off') # Hide axes
        node_df = node_df_combined.query('univ==@univ')
        edge_df = edge_df_combined.query('univ==@univ')
        input_df = input_df_combined.query('univ==@univ')
        if plot_output_layer:
            output_df = output_df_combined.query('univ==@univ')
            all_edge_df = pd.concat([edge_df,input_df,output_df])
        else:
            all_edge_df = pd.concat([edge_df,input_df])
        layers = node_df['layer'].unique()
        layers = ['input'] + sorted([x for x in layers if x not in ['input','output']]) + ['output']
        colors = node_df['species_type'].apply(lambda x: color_types[x])

        for i, edge in all_edge_df.iterrows():
            if not pd.isnull(edge['weight']):
                if not pd.isnull(edge['Kij_name']):
                    weight = edge['weight']*edge_scales['K']
                    color = K_cmap(sorted(list(set(edge_df['Kij_name']))).index(edge['Kij_name']))
                    alpha = 0.7
                else:
                    weight = edge['weight']*edge_scales['output']
                    color = '#7F7F7F'
                    alpha = 0.4

            else:
                weight = edge_scales['input']
                color = 'k'
                alpha = 1
            if rotate:
                ax.plot([edge['y_start'], edge['y_end']], [-edge['x_start'],-edge['x_end']],\
                       color = color,\
                       lw=weight,alpha=alpha,zorder=1)
            else:
                ax.plot([edge['x_start'],edge['x_end']],\
                        [edge['y_start'], edge['y_end']], color = color,\
                       lw=weight,alpha=alpha,zorder=1)

        input_node_df = node_df.query('species_type=="input"')
        accessory_node_df = node_df.query('species_type=="accessory"')
        dimer_node_df = node_df.query('species_type=="dimer"')
        if rotate:
            ax.scatter(input_node_df['y'],-input_node_df['x'],\
                   s = node_scales['input']*input_node_df['weight'],c=input_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            ax.scatter(accessory_node_df['y'],-accessory_node_df['x'],\
                   s = node_scales['accessory']*accessory_node_df['weight'],c=accessory_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            ax.scatter(dimer_node_df['y'],-dimer_node_df['x'],\
                   s = node_scales['dimer']*dimer_node_df['weight'],c=dimer_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            if plot_output_layer:
                output_node_df = node_df.query('species_type=="output"')
                ax.scatter(output_node_df['y'],-output_node_df['x'],\
                       s = node_scales['output']*output_node_df['weight'],c=output_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
        else:
            ax.scatter(input_node_df['x'],input_node_df['y'],\
                   s = node_scales['input']*input_node_df['weight'],c=input_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            ax.scatter(accessory_node_df['x'],accessory_node_df['y'],\
                   s = node_scales['accessory']*accessory_node_df['weight'],c=accessory_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            ax.scatter(dimer_node_df['x'],dimer_node_df['y'],\
                   s = node_scales['dimer']*dimer_node_df['weight'],c=dimer_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)
            if plot_output_layer:
                output_node_df = node_df.query('species_type=="output"')
                ax.scatter(output_node_df['x'],output_node_df['y'],\
                       s = node_scales['output']*output_node_df['weight'],c=output_node_df['species_type'].apply(lambda x: color_types[x]),zorder=2)

    patches={}
    patches['Input'] = mlines.Line2D([0,1],[0,1],ls='',marker='o',color='k',label='Input')
    patches['Accessory'] = mlines.Line2D([0,1],[0,1],ls='',marker='o',color='r',label='Accessory')
    patches['Dimer'] = mlines.Line2D([0,1],[0,1],ls='',marker='o',color='#636363',label='Dimer')
    if plot_output_layer:
        patches['Output'] = mlines.Line2D([0,1],[0,1],ls='',marker='o',color='blue',label='Output')
    font = mpl.font_manager.FontProperties(family='Gill Sans MT', size=16)
    leg = fig.legend(bbox_to_anchor=(1, 0.5), loc='center left',facecolor='white',\
        handles=list(patches.values()),\
        edgecolor='white',prop=font)
    return fig, axs

################################################
# Plotting: Regular Network Diagram
################################################

def make_network_nodes_polygon(n, r, n_input, titrated_param = None):
    x, y = get_poly_vertices(n, r=r)
    species = [f'M{i}' for i in range(1,n+1)]
    species_type = ['input']*n_input + ['accessory']*(n - n_input)
    param_titrated = np.array(species) == titrated_param
    node_df = pd.DataFrame({'species': species, 'species_type':species_type, 
                            'x': x, 'y': y, 'titrated_param': param_titrated})
    
    return node_df

def make_self_edges(n, r_node, r_edge, circ_points = 99, titrated_param = None):
    edge_df_list = [0] * n
    x, y = get_poly_vertices(n, r=r_node+r_edge)
    # weights_scaled = np.array(K)*edge_scale
    for i in range(n):
        # Set center of self-edge to be r_edge further from the origin
        if x[i]>=0:
            angle = np.arctan(y[i]/x[i])
        else:
            angle = np.arctan(y[i]/x[i])+np.radians(180)
        y_new = (1+r_edge)*np.sin(angle)
        x_new = (1+r_edge)*np.cos(angle)
        center = [[x_new, y_new]]
        tmp_df = pd.DataFrame(center, columns=['x', 'y'])
        tmp_df['Kij_names'] = [f'K{i+1}{i+1}']
        tmp_df['titrated_param'] = tmp_df['Kij_names'] == titrated_param 
        edge_df_list[i] = tmp_df
        
    return pd.concat(edge_df_list)

def make_heterodimer_edges(n, node_df, titrated_param = None):
    pairs = itertools.combinations(range(n), 2)
    n_heterodimers = number_of_heterodimers(n)
    x = [0]*n_heterodimers
    x_end = [0]*n_heterodimers
    y = [0]*n_heterodimers
    y_end = [0]*n_heterodimers
    names = [0]*n_heterodimers
    for i, comb in enumerate(pairs):
        x[i] = node_df.loc[comb[0],'x']
        x_end[i] = node_df.loc[comb[1],'x']
        y[i] = node_df.loc[comb[0],'y']
        y_end[i] = node_df.loc[comb[1],'y']
        names[i] = f'K{comb[0]+1}{comb[1]+1}'
    
    edge_df = pd.DataFrame({'Kij_names': names,
                          'x': x, 'x_end': x_end,
                          'y': y, 'y_end': y_end})
    edge_df['titrated_param'] = edge_df['Kij_names'] == titrated_param 
    return edge_df

def make_network_plots_polygon(n, n_input, univs_to_plot, param_sets=None, param_set_file=None, ncols = 1, r_node = 1, r_loop = 0.5,
                            node_scale = 20, K_edge_scale = 30,figsize=(12,12),input_cmap='Pastel1',fontsize=16):
    """
    Load a subset of networks from a parameter file and plot the affinity parameters between monomers.
    
    Parameters
    --------------
    n: Int
        Number of total monomers in the network.
    n_input: Int
        Number of input monomers
    univs_to_plot: array-like
        Array of which universes (parameter sets) to plot.
    param_sets: Array-like, shape (n_univ, n_parameters)
        Array of parameters to plot. Alternatively, can use:
        param_set_file: Str
            File path to the .npy file containing the saved parameter sets.
    out_weights: array-like or None
        Weights describing the output activity of each dimer species. Either specify a weight
        for ALL dimers (including zeros for inactive dimers) or pass None for no output layer.
    ncols: Int
        Number of columns to use for multiple subplots.
    node_scales: Dict
        Scale factors for the sizes of the nodes. Keys: ['input','accessory','dimer','output']
    edge_scales: Dict
        Scale factors for the widths of the edges. Keys: ['input','K','output']
    rotate: Bool
        Whether to rotate the plots 90 degrees clockwise to be top-down.
    figsize: Tuple of len 2
        Width and height of the final figure in inches.
    input_cmap: Colormap, or name of matplotlib colormap to use for node colors.
    fontsize: Int. Size of the labels.
    
    Returns:
        fig, axs: Created plot
    """
    if param_sets is None:
        param_sets = np.load(param_set_file)
    param_sets = param_sets[univ_to_plot,:]
    num_plots = len(univ_to_plot)
    species_names = np.array(make_nXn_species_names(n))
    dimer_names = species_names[n:]
    Kij_labels = make_Kij_names(n_input = n_input, n_accesory=(n-n_input))
    num_rxns = len(Kij_labels)
    
    #Make dataframe containing node positions. Color accessory monomer nodes and scale size by parameter value
    
    node_df_list = [0]*num_plots
    for i in range(num_plots):
        #scale acc monomer weights
        acc_weights = np.log10(param_sets[i,num_rxns:]) + 3
        node_df_list[i] = make_network_nodes_polygon(n=n, r=r_node, n_input=n_input)
    node_df_combined = pd.concat(node_df_list, keys=univ_to_plot).reset_index()
    node_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    
    node_weights = param_sets[np.arange(num_plots), num_rxns:]
    node_weights = np.hstack((np.ones((num_plots, n_input))*1e-3, node_weights))
    node_weights = np.log2(node_weights/1e-4)*node_scale

    node_df_combined['weight'] = node_weights.flatten()
    
    #Make dataframe for self-loops. Scale width by Kii value
    self_edge_df = make_self_edges(n, r_node, r_loop) 
    self_edge_df_combined = pd.concat([self_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index()
    self_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    self_edge_labels = [f'K{i}{i}' for i in range(1,n+1)]
    self_edge_index = np.where(np.isin(Kij_labels, self_edge_labels))[0]

    self_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], self_edge_index]
    self_edge_weights = np.log2(self_edge_weights/1e-6)*edge_scale

    self_edge_df_combined['weight'] = np.repeat(self_edge_weights.flatten(), self_edge_df_combined.level_1.max()+1)

    #Make dataframe for heterodimer edges. Scale width by Kij value
    hetero_edge_df = make_heterodimer_edges(n, node_df_combined)
    hetero_edge_df_combined = pd.concat([hetero_edge_df]*num_plots, keys=np.arange(num_plots)).reset_index()
    hetero_edge_df_combined.rename(columns={'level_0': 'univ'}, inplace=True)
    hetero_edge_index = np.where(~np.isin(Kij_labels, self_edge_labels))[0]
    hetero_edge_weights = param_sets[np.arange(num_plots)[:,np.newaxis], hetero_edge_index]
    hetero_edge_weights = np.log2(hetero_edge_weights/1e-6)*edge_scale
    hetero_edge_df_combined['weight'] = hetero_edge_weights.flatten()
    
    if type(input_cmap)==str:
        cmap = cm.get_cmap(input_cmap)
    else:
        cmap = input_cmap

    if len(univ_to_plot)==1:
        ncols = 1

    fig, axs = plt.subplots(nrows=math.ceil(len(univ_to_plot)//ncols), ncols=ncols,figsize=figsize)
    if len(univ_to_plot)==1:
        axs = [axs]

    for univ, ax in enumerate(axs):
        ax.axis('off') # Hide axes
        ax.set_xlim([-2-r_loop,2+r_loop])
        ax.set_ylim([-2-r_loop,2+r_loop])
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        xrange = ax.get_xlim()[1] - ax.get_xlim()[0]
        yrange = ax.get_ylim()[1] - ax.get_ylim()[0]
        x_points_per_data = (bbox.width*fig.get_dpi())/xrange
        y_points_per_data = (bbox.height*fig.get_dpi())/yrange
        points_per_data = np.mean([x_points_per_data,y_points_per_data])
        for i, edge in self_edge_df_combined.iterrows():
            weight = edge['weight']*K_edge_scale
            ax.plot([edge['x']],[edge['y']], ls='',marker='o',markeredgecolor = 'k',\
                   markersize=(2*((r_loop*points_per_data)+weight)),\
                   markeredgewidth=weight,markerfacecolor='none',\
                   zorder=1)
            ax.text(edge['x'],edge['y'],'M{}'.format(edge['Kij_names'][1]),c='k',ha='center',va='center',fontsize=fontsize,fontweight='bold',fontname='Gill Sans MT')
            # Draw arrow
            arrow_size = edge['weight']*K_edge_scale*0.05
            node_name = "M{}".format(edge["Kij_names"][1])
            node_coord = np.array(node_df_combined.query('species == @node_name').reset_index(drop=True).loc[0,['x','y']])
            if node_coord[0]>=0:
                angle = np.arctan(node_coord[1]/node_coord[0])
            else:
                angle = np.arctan(node_coord[1]/node_coord[0])+np.radians(180)
            arrow_coord = np.array([(1+(2*r_loop)+(weight/points_per_data/2))*np.cos(angle),(1+(2*r_loop)+(weight/points_per_data/2))*np.sin(angle)])
            arrow_vector = np.array([node_coord[1],-node_coord[0]])
            arrow_vector = arrow_vector/np.linalg.norm(arrow_vector,ord=2)
            arrow_tip = arrow_coord+(arrow_vector*(-arrow_size/2))
            arrow_base = arrow_coord+(arrow_vector*(arrow_size/2))
            plt.arrow(arrow_base[0], arrow_base[1], (arrow_tip-arrow_base)[0], (arrow_tip-arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*1.3, head_length=arrow_size*1.3, overhang=0.25,color='k')
        for i, edge in hetero_edge_df_combined.iterrows():
            ax.plot([edge['x'],edge['x_end']],[edge['y'],edge['y_end']], color = 'k',\
                   lw=edge['weight']*K_edge_scale,marker=None,zorder=1)
            # Draw arrows
            arrow_size = edge['weight']*K_edge_scale*0.05
            start_node_coord = np.array([edge['x'],edge['y']])
            end_node_coord = np.array([edge['x_end'],edge['y_end']])
            start_node_name = "M{}".format(edge["Kij_names"][1])
            end_node_name = "M{}".format(edge["Kij_names"][2])
            start_node_radius_dataunits = (np.sqrt(node_df_combined.query('species == @start_node_name')['weight']\
                                                  .reset_index(drop=True)[0]*node_scale)/2/points_per_data)
            end_node_radius_dataunits = (np.sqrt(node_df_combined.query('species == @end_node_name')['weight']\
                                                 .reset_index(drop=True)[0]*node_scale)/2/points_per_data)
            start_end_vector = end_node_coord-start_node_coord
            start_end_vector = start_end_vector/np.linalg.norm(start_end_vector,ord=2)
            start_arrow_tip = start_node_coord+start_end_vector*start_node_radius_dataunits
            start_arrow_base = start_node_coord+start_end_vector*(start_node_radius_dataunits+arrow_size)
            end_arrow_tip = end_node_coord-(start_end_vector*end_node_radius_dataunits)
            end_arrow_base = end_node_coord-(start_end_vector*(end_node_radius_dataunits+arrow_size))
            plt.arrow(start_arrow_base[0], start_arrow_base[1], (start_arrow_tip-start_arrow_base)[0], (start_arrow_tip-start_arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size, head_length=arrow_size, overhang=0.25,color='k')
            plt.arrow(end_arrow_base[0], end_arrow_base[1], (end_arrow_tip-end_arrow_base)[0], (end_arrow_tip-end_arrow_base)[1], \
                      length_includes_head=True, head_width=arrow_size*1.3, head_length=arrow_size*1.3, overhang=0.25,color='k')

        ax.scatter(node_df_combined['x'],node_df_combined['y'],\
               s = node_df_combined['weight']*node_scale,color=[cmap(i) for i in range(n)],zorder=2)

    patches={}
    for i in range(n):
        patches[str(i)] = mpatches.Patch(color=cmap(i),label=f'M{i+1}')
        
    font = mpl.font_manager.FontProperties(family='Gill Sans MT', size=fontsize)

    leg = fig.legend(bbox_to_anchor=(1, 0.5), loc='center left',facecolor='white',\
        handles=list(patches.values()),\
        edgecolor='white',prop=font)
    return fig, axs


