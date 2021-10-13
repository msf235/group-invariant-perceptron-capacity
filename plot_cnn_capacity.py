
"""Script for plotting cnn capacity experiments.

This script calls cnn_capacity.py.

Sets of parameters used for running simulations can be found in the file 
cnn_capacity_params.py.

Datasets and the relevant group-shifted versions of datasets can be found
in datasets.py."""

import os
import math
import itertools
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_palette('colorblind')
import pickle as pkl

import cnn_capacity_params as cp
import cnn_capacity

plt.rcParams.update({
    'axes.labelsize': 'xx-large',
    'xtick.labelsize': 'x-large',
    'ytick.labelsize': 'x-large',
    "text.usetex": True,
})

fig_dir = 'figs'
# rerun = True # If True, rerun the simulation even if a matching simulation is
               # found saved to disk
rerun = False

# Number of processor cores to use for multiprocessing. Recommend setting to 1
# while debugging
n_cores = 5
seeds = [3, 4, 5]


def cover_theorem(P, N):
    frac_dich = 0
    for k in range(min(P,N)):
        frac_dich += math.factorial(P-1) / math.factorial(P-1-k) / math.factorial(k)
    frac_dich = 2**(1-1.0*P) * frac_dich
    return frac_dich

def make_plot(param_set_names):
    print('Running {}'.format('  '.join(param_set_names)))
    
    param_set = []
    for name in param_set_names:
        param_set += cp.param_sets[name]
    
    plot_vars = ['n_channels', 'n_inputs', 'layer_idx']

    results_table = pd.DataFrame()
    for seed in seeds:
        for i0, params in enumerate(param_set):
            print(f"Starting param set {i0+1}/{len(param_set)} with seed {seed}")
            capacity = cnn_capacity.get_capacity(seed=seed, n_cores=n_cores, 
                                    rerun=rerun, **params)
            layer = params['layer_idx']
            n_input = params['n_inputs']
            n_channel = params['n_channels']
            net_style = params['net_style']
            if net_style == 'grid':
                factor = 2
            else:
                factor = 1
            offset = int(params['fit_intercept'])
            alpha = n_input / (factor*n_channel + offset)
            cover_capacity = cover_theorem(n_input, n_channel)
            d1 = {'seed': seed, 'alpha': alpha, 'n_inputs': n_input,
                  'n_channels': n_channel, 'n_channels_offset':
                  n_channel + offset, 'fit_intercept': params['fit_intercept'],
                  'layer': layer, 'net_style': net_style, 'capacity': capacity,
                 }
            d1 = pd.DataFrame(d1, index=[0])
            results_table = results_table.append(d1, ignore_index=True)

    for catcol in ('layer',):
        results_table[catcol] = results_table[catcol].astype('category')

    if len(results_table['net_style'].unique()) > 1:
        style = 'net_style'
    else:
        style = None
    os.makedirs('figs', exist_ok=True)
    alpha_table = results_table.drop(
        columns=['n_channels', 'n_inputs', 'n_channels_offset',
                 'fit_intercept'])
    fig, ax = plt.subplots(figsize=(5,4))
    g = sns.lineplot(ax=ax, x='alpha', y='capacity', data=alpha_table,
                 hue='layer')
    # g.legend_.remove()
    nmin = results_table['n_channels_offset'].min()
    nmax = results_table['n_channels_offset'].max()
    pmin = results_table['n_inputs'].min()
    pmax = results_table['n_inputs'].max()
    alphamin = results_table['alpha'].min()
    alphamax = results_table['alpha'].max()
    if net_style == 'grid':
        factor = 2
    else:
        factor = 1
    cover_cap = {p/(factor*n): cover_theorem(p, factor*n) for n in range(nmin, nmax+1)
                for p in range(pmin, pmax+1) if alphamin <= p/(factor*n) <= alphamax}
    ax.plot(list(cover_cap.keys()), list(cover_cap.values()), linestyle='dotted',
           color='blue', label='theory')
    P = param_set[0]['n_inputs']
    if param_set[0]['net_style'] == 'grid':
       ax.set_xlabel(r'$\alpha = P/N_0$')
    else:
       ax.set_xlabel(r'$\alpha = P/N_0$')
    ax.set_ylim([-.01, 1.01])
    figname = '__'.join(param_set_names)
    fig.savefig(f'figs/{figname}.pdf', bbox_inches='tight')
    results_table.to_pickle(f'figs/{figname}.pkl')


## Run script by calling get_capacity 
if __name__ == '__main__':

    # Figure 2a
    param_set_names = ['random_2d_conv_exps']
    make_plot(param_set_names)

    # Figure 2b
    param_set_names = ['vgg11_cifar10_circular_exps']
    make_plot(param_set_names)

    # Figure 2c
    param_set_names = ['grid_2d_conv_exps']
    make_plot(param_set_names)

    # Figure 3
    param_set_names = ['vgg11_cifar10_exps']
    make_plot(param_set_names)

