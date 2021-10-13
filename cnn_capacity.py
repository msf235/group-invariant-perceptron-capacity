
"""Script for running cnn capacity experiments.

Sets of parameters used for running simulations can be found in the file 
cnn_capacity_params.py.

Datasets and the relevant group-shifted versions of datasets can be found
in datasets.py."""

import os
import sys
import inspect
import math
import torch
import torchvision
import itertools
import torchvision.transforms as transforms
from sklearn import svm, linear_model
from sklearn.exceptions import ConvergenceWarning
from joblib import Parallel, delayed
import pickle as pkl
import numpy as np
import warnings
from typing import *

import models
import model_output_manager as mom
import cnn_capacity_params as cp
import datasets
import cnn_capacity_utils as utils

output_dir = 'output'

# % Main function for capacity. This function is memoized based on its
# parameters.
def get_capacity(
    n_channels, n_inputs, seed=3, n_dichotomies=100, max_epochs=500,
    max_epochs_no_imp=None, improve_tol=1e-3, batch_size=256, img_size_x=10,
    img_size_y=10, img_channels=3, net_style='rand_conv', layer_idx=0,
    dataset_name='gaussianrandom', shift_style='2d', shift_x=1, shift_y=1,
    pool_over_group=False, perceptron_style='standard',
    pool=None, pool_x=None, pool_y=None, fit_intercept=True,
    center_response=True, n_cores=1, rerun=False):
    """Take number of channels of response (n_channels) and number of input
    responses (n_inputs) and a set of hyperparameters and return the capacity
    of the representation.

    This function checks to see how large the dataset and representation
    would be in memory. If this value, times n_cores, is <= 30GB, then
    the code calls linear_model.LinearSVC on the entire dataset.
    If not, the code calls linear_model.SGDClassifier on batches of
    the data.

    Parameters
    ----------
    n_channels : int
		Number of channels in the network response.
    n_inputs : int
		Number of input samples to use.
    n_dichotomies : int
		Number of random dichotomies to test
    max_epochs : int
		Maximum number of epochs. This is also the max number of iterations
        when using LinearSVC.
    batch_size : Optional[int]
		Batch size. If None, this is set to the size of the dataset.
    img_size_x : int
		Size of image x dimension.
    img_size_y : int
		Size of image y dimension.
    net_style : str 
        Style of network. Valid options are 'vgg11', 'grid', 'rand_conv', and
        'randpoints'.
    layer_idx : int
        Index for layer to get from conv net.
    dataset_name : str 
		The dataset. Options are 'cifar10', and 'gaussianrandom'
    shift_style : str 
        Shift style. Options are 1d (shift in only x dimension) and 2d (use
        input shifts in both x and y dimensions).
    shift_x : int
		Number of pixels by which to shift in the x direction
    shift_y : int
		Number of pixels by which to shift in the y direction
    pool_over_group : bool 
        Whether or not to average (pool) the representation over the group
        before fitting the linear classifier.
    perceptron_style : str {'efficient', 'standard'}
        How to train the output weights. If 'efficient' then use the trick
        of finding a separating hyperplane for the centroids
        and then applying the average group projector to this hyperplane.
    pool : Optional[str] 
		Pooling to use for representation. Options are None, 'max', and 'mean'.
        Only currently implemented for net_style in ('rand_conv', 'grid').
    pool_x : Optional[int] 
		Size in pixels of pool in x direction. Set to None if pool is None.
    pool_y : Optional[int] 
		Size in pixels of pool in y direction. Set to None if pool is None.
    fit_intercept : bool 
		NOT USED. Whether or not to fit the intercept in the linear classifier.
        This currently throws an error when set to True since I haven't
        got the intercept working with perceptron_style = 'efficient' yet.
    center_response : bool 
        Whether or not to mean center each representation response.
    seed : int
		Random number generator seed. 
    n_cores : int
        Number of processes to spawn for parallel processing.
    rerun : bool
        If False, look for a previous run that matches the passed parameters
        and load the result if found. If True, rerun the simulation even
        if a matching simulation is found on disk. This overwrites the
        previous save of the simulation.
    """
    if pool is None:
        pool_x = None
        pool_y = None
    if max_epochs_no_imp is None:
        improve_tol = None
    loc = locals()
    args = inspect.getfullargspec(get_capacity)[0]
    params = {arg: loc[arg] for arg in args}
    del params['n_cores']
    del params['rerun']
    if mom.run_exists(params, output_dir) and not rerun: # Memoization
        run_id = mom.get_run_entry(params, output_dir)
        run_dir = output_dir + f'/run_{run_id}'
        try:
            with open(run_dir + '/get_capacity.pkl', 'rb') as fid:
                savedict = pkl.load(fid)
            return savedict['capacity']
        except FileNotFoundError:
            pass

    if perceptron_style == 'efficient':
        if pool_over_group:
            raise AttributeError("""perceptron_style=efficient not implemented
                                with group pooling.""")
    if fit_intercept:
        raise AttributeError("fit_intercept=True not currently implemented.")
    torch.manual_seed(seed)

    pool_efficient_shift = 0

    if net_style[:5] == 'vgg11':
        if net_style[6:] == 'circular':
            net = models.vgg('vgg11_bn', 'A', batch_norm=True, pretrained=True,
                            circular_conv=True)
        else:
            net = models.vgg('vgg11_bn', 'A', batch_norm=True, pretrained=True)
        net.eval()
        def feature_fn(inputs):
            with torch.no_grad():
                feats = net.get_features(inputs, layer_idx)
                feats = feats[:, :n_channels]
                return feats
        if perceptron_style == 'efficient' or pool_over_group:
            if layer_idx > 2:
                if layer_idx > 6:
                    raise AttributeError("""This parameter combination not
                                         supported.""")
                pool_efficient_shift = 1
    elif net_style == 'grid':
        convlayer = torch.nn.Conv2d(img_channels, n_channels, (img_size_x, img_size_y),
                                    padding='same', padding_mode='circular',
                                    bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        net = torch.nn.Sequential(
            convlayer,
            torch.nn.ReLU(),
            models.MultiplePeriodicAggregate2D(((10, 10), (8, 8))),
        )
        net.eval()
        def feature_fn(input):
            with torch.no_grad():
                hlist = net(input)
            hlist = [h.reshape(*h.shape[:2], -1) for h in hlist]
            h = torch.relu(torch.cat(hlist, dim=-1))
            return h
    elif net_style == 'rand_conv':
        convlayer = torch.nn.Conv2d(img_channels, n_channels,
                                    (img_size_x, img_size_y),
                            padding='same', padding_mode='circular',
                            bias=False)
        torch.nn.init.xavier_normal_(convlayer.weight)
        layers = [convlayer, torch.nn.ReLU()]
        if pool is not None:
            if pool == 'max':
                pool_layer = torch.nn.MaxPool2d((pool_x, pool_y),
                                                (pool_x, pool_y)) 
            elif pool == 'mean':
                pool_layer = torch.nn.AvgPool2d((pool_x, pool_y),
                                                (pool_x, pool_y)) 
            if pool is not None or pool_over_group:
                if layer_idx > 1:
                    pool_efficient_shift = 1
            layers.append(pool_layer)
        layers = layers[:layer_idx+1]
        net = torch.nn.Sequential(*layers)
        net.eval()
        def feature_fn(input):
            with torch.no_grad():
                h = net(input)
                if center_response:
                    hflat = h.reshape(*h.shape[:2], -1)
                    hmean = hflat.mean(dim=-1, keepdim=True)
                    hms = hflat - hmean
                    hms_rs = hms.reshape(*h.shape)
                    return hms_rs
                return h
    elif net_style == 'randpoints':
        net = torch.nn.Module()
        net.eval()
        def feature_fn(inputs):
            return inputs
    else:
        raise AttributeError('net_style option not recognized')


    if net_style == 'randpoints':
        inp_channels = n_channels
    else:
        inp_channels = img_channels
    if dataset_name.lower() == 'cifar10':
        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                              std=(0.2023, 0.1994, 0.2010))])
        img_dataset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False, download=True,
            transform=transform)
        random_samples = torch.randperm(len(img_dataset))[:n_inputs]
        core_dataset = datasets.SubsampledData(img_dataset, random_samples)
    elif dataset_name.lower() == 'gaussianrandom':
        def zero_one_to_pm_one(y):
            return 2*y - 1
        core_dataset = datasets.FakeData(n_inputs,
                            (inp_channels, img_size_x, img_size_y),
                            target_transform=zero_one_to_pm_one)
    else:
        raise AttributeError('dataset_name option not recognized')

    if n_cores == 1:
        num_workers = 4
    else:
        num_workers = 0

    # if pool_over_group:
        # dataset = core_dataset
    if shift_style == '1d':
        dataset = datasets.ShiftDataset1D(core_dataset, shift_y)
    elif shift_style == '2d':
        dataset = datasets.ShiftDataset2D(core_dataset, shift_x, shift_y)
    else:
        raise AttributeError('Unrecognized option for shift_style.')
    if perceptron_style == 'efficient' or pool_over_group:
        datasetfull = dataset
        dataset = core_dataset
        inputsfull = torch.stack([x[0] for x in datasetfull])
        coreidxfull = [x[2] for x in datasetfull]
        dataloaderfull = torch.utils.data.DataLoader(
            datasetfull, batch_size=batch_size, num_workers=num_workers,
            shuffle=False)

    if batch_size is None or batch_size == len(dataset):
        batch_size = len(dataset)
        inputs = torch.stack([x[0] for x in dataset])
        if pool_over_group or perceptron_style == 'efficient':
            core_idx = list(range(len(dataset)))
        else:
            core_idx = [x[2] for x in dataset]
        dataloader = [(inputs, None, core_idx)]
    else:
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 num_workers=num_workers,
                                                 shuffle=True)



    # test_input, test_label = datasetfull[:2]
    test_input, test_label = next(iter(dataloader))[:2]
    # plt.figure(); plt.imshow(dataset[100][0].transpose(0,2).transpose(0,1)); plt.show()
    h_test = feature_fn(test_input)
    if h_test.shape[1] < n_channels:
        raise AttributeError("""Error: network response produces fewer channels
                             than n_channels.""")

    if net_style == 'grid':
        P = utils.compute_pi_mean_reduced_grid_2D((10, 8))
    else:
        P = utils.compute_pi_mean_reduced_2D(h_test.shape[-2], h_test.shape[-1],
                                       shift_x, shift_y) 
    Pt = P.T.copy()
    
    def score(w, X, Y):
        Ytilde = X @ w
        return np.mean(np.sign(Ytilde) == np.sign(Y-.5))

    def dich_loop(process_id=None):
        """Generates random labels and returns the accuracy of a classifier
        trained on the dataset."""
        np.random.seed(seed)
        rndseed = np.random.randint(10000)
        torch.manual_seed(process_id + rndseed)  # Be very careful with this
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
        class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        while len(set(class_random_labels.tolist())) < 2:
            class_random_labels = 2*(torch.rand(len(core_dataset)) < .5) - 1
        if batch_size == len(dataset): # Train classifier on entire dataset
            perceptron = linear_model.LogisticRegression(
                tol=1e-18, C=1e8, fit_intercept=fit_intercept,
                max_iter=max_epochs, random_state=seed + rndseed)
            inputs = dataloader[0][0]
            core_idx = dataloader[0][2]
            h = feature_fn(inputs)
            if perceptron_style == 'efficient' or pool_over_group:
                if perceptron_style == 'efficient':
                    hfull = feature_fn(inputsfull)
                    Xfull = hfull.reshape(hfull.shape[0], -1).numpy()
                    Yfull = class_random_labels[coreidxfull].numpy()
                    Yfull = (Yfull + 1)/2
                Y = np.array(class_random_labels)
                if pool_efficient_shift == 1:
                    inputs21 = torch.roll(inputs, shifts=1, dims=-2) 
                    h21 = feature_fn(inputs21)
                    inputs12 = torch.roll(inputs, shifts=1, dims=-1) 
                    h12 = feature_fn(inputs12)
                    inputs22 = torch.roll(inputs21, shifts=1, dims=-1) 
                    h22 = feature_fn(inputs22)
                    h = torch.cat((h, h21, h12, h22), dim=0)
                    Y = np.concatenate((Y,Y,Y,Y), axis=0)
                hrs = h.reshape(*h.shape[:2], -1)
                centroids = hrs @ Pt
                X = centroids.reshape(centroids.shape[0], -1).numpy()
            else:
                X = h.reshape(h.shape[0], -1).numpy()
                Y = class_random_labels[core_idx].numpy()
            Y = (Y + 1)/2
            perceptron.fit(X, Y)
            if perceptron_style == 'efficient':
                wtemp = perceptron.coef_.copy()
                wtemp = wtemp.reshape(-1, P.shape[0])
                wtemp = wtemp @ P
                wtemp = wtemp.reshape(-1)
                curr_avg_acc = score(wtemp, Xfull, Yfull)
            else:
                curr_avg_acc = perceptron.score(X, Y)
        else: # Use minibatches -- not well tested
            perceptron = linear_model.SGDClassifier(
                tol=1e-18, alpha=1e-16, fit_intercept=fit_intercept,
                max_iter=max_epochs)
            for epoch in range(max_epochs):
                losses_epoch = []
                class_acc_epoch = []
                for k1, data in enumerate(dataloader):
                    inputs = data[0]
                    h = feature_fn(inputs)
                    if pool_over_group:
                        hrs = h.reshape(*h.shape[:2], -1)
                        centroids = hrs @ Pt
                        X = centroids.reshape(centroids.shape[0], -1).numpy()
                        Y = np.array(class_random_labels)
                    else:
                        core_idx_batch = data[2]
                        X = h.reshape(h.shape[0], -1).numpy()
                        Y = class_random_labels[core_idx_batch].numpy()
                    perceptron.partial_fit(X, Y, classes=(-1, 1))
                    class_acc_epoch.append(perceptron.score(X, Y).item())
                    curr_avg_acc = sum(class_acc_epoch)/len(class_acc_epoch)
                if perceptron_style == 'efficient':
                    raise AttributeError("""Efficient perceptron style and
                                         minibatching not supported together.""")
                if curr_avg_acc == 1.0:
                    break
        return curr_avg_acc

    if n_cores > 1:
        print(f"Beginning parallelized loop over {n_dichotomies} dichotomies.")
        class_acc_dichs = Parallel(n_jobs=n_cores, batch_size=1, verbose=10)(
            delayed(dich_loop)(k1) for k1 in range(n_dichotomies))
    else:
        print(f"Beginning serial loop over {n_dichotomies} dichotomies.")
        class_acc_dichs = []
        for k1 in range(n_dichotomies):
            class_acc_dichs.append(dich_loop(k1))
            print(f'Finished dichotomy: {k1+1}/{n_dichotomies}', end='\r')

    capacity = (1.0*(torch.tensor(class_acc_dichs) == 1.0)).mean().item()
    if fit_intercept:
        alpha = n_inputs / (n_channels + 1)
    else:
        alpha = n_inputs / n_channels
    print(f'alpha: {round(alpha,5)}, capacity: {round(capacity,5)}')
    
    ## Now save results of the run to a pickled dictionary
    run_id = mom.get_run_entry(params, output_dir)
    run_dir = output_dir + f'/run_{run_id}/'
    os.makedirs(run_dir, exist_ok=True)
    with open(run_dir + 'get_capacity.pkl', 'wb') as fid:
        savedict = pkl.dump({'capacity': capacity}, fid)

    return capacity

