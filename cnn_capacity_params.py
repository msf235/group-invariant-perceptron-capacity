import torch
import copy

## Convenience functions
def alphas_to_channels(alphas, n_inputs, fit_intercept):
    n_channels_temp = torch.round(n_inputs/alphas).int()
    n_channels_temp -= fit_intercept 
    n_channels_temp = n_channels_temp.tolist()
    n_channels = []
    [n_channels.append(x) for x in n_channels_temp if x not in n_channels]
    return n_channels

def exps_channels_and_layers(param_base, n_channels, layers=None):
    exps = []
    if layers is not None:
        for layer in layers:
            for n_channel in n_channels:
                temp = param_base.copy()
                temp['layer_idx'] = layer
                temp['n_channels'] = n_channel
                exps.append(temp)
    else:
        for n_channel in n_channels:
            temp = param_base.copy()
            temp['n_channels'] = n_channel
            exps.append(temp)
            
    return exps


param_sets = {}

## One-dimensional random CNN layer.
random_1d_conv = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    n_inputs = 40, # Number of input samples to test
    max_epochs = 4000, # Maximum number of epochs.
    batch_size = None, # Batch size if training with SGD
    img_size_x = 40, # Size of image x dimension.
    img_size_y = 1, # Size of image y dimension.
    net_style = 'rand_conv', # Random convolutional layer.
    dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
    img_channels = 3,
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    pool_over_group = False, # group before fitting the linear classifier.
    pool='max', 
    fit_intercept = False,
    center_response = False,
)
alphas = torch.linspace(0.5, 3.0, 15)
layer_idx = [1, 2]
n_channels = alphas_to_channels(alphas, random_1d_conv['n_inputs'],
    int(random_1d_conv['fit_intercept']))
param_sets.update(
random_1d_conv_exps = exps_channels_and_layers(random_1d_conv, n_channels,
                                              layers=layer_idx)
)

## Random inputs
randpoint = random_1d_conv.copy()
randpoint['net_style'] = 'randpoints'
n_channels = alphas_to_channels(
    alphas, randpoint['n_inputs'], int(randpoint['fit_intercept']))
param_sets.update(
randpoint_exps = exps_channels_and_layers(randpoint, n_channels)
)

randpoint_efficient = randpoint.copy()
randpoint_efficient.update(
    perceptron_style = 'efficient'
)
param_sets.update(
randpoint_efficient_exps = exps_channels_and_layers(randpoint_efficient,
                                                    n_channels)
)

## Random CNN layer
random_2d_conv = random_1d_conv.copy()
random_2d_conv.update(
    perceptron_style='standard',
    img_size_x = 10,
    img_size_y = 10,
    pool_x = 2,
    pool_y = 2
)
alphas = torch.cat((torch.linspace(0.5, 1.0, 10), torch.linspace(1.0, 3.0, 10)),
                   dim=0)
layer_idx = [1, 2]
n_channels = alphas_to_channels(
    alphas, random_2d_conv['n_inputs'],
    int(random_2d_conv['fit_intercept']))
param_sets.update(
random_2d_conv_exps = exps_channels_and_layers(
    random_2d_conv, n_channels, layers=layer_idx)
)

random_2d_conv_gpool = random_2d_conv.copy()
random_2d_conv_gpool.update(
    pool_over_group=True,
)
param_sets.update(
    random_2d_conv_gpool_exps = exps_channels_and_layers(
            random_2d_conv_gpool, n_channels, layers=layer_idx)
)

random_2d_conv_efficient = random_2d_conv.copy()
random_2d_conv_efficient.update(
    perceptron_style='efficient',
)
param_sets.update(
random_2d_conv_efficient_exps = exps_channels_and_layers(
    random_2d_conv_efficient, n_channels, layers=layer_idx)
)

## Random CNN layer with 2 pixel shifts
random_2d_conv_shift2 = random_2d_conv.copy()
random_2d_conv_shift2.update(
    shift_x=2,
    shift_y=2,
)
alphas=torch.linspace(3.0, 8.0, 10)
n_channels = alphas_to_channels(
    alphas, random_2d_conv_shift2['n_inputs'],
    int(random_2d_conv_shift2['fit_intercept']))
param_sets.update(
random_2d_conv_shift2_exps = exps_channels_and_layers(
    random_2d_conv_shift2, n_channels)
)

## VGG11 on CIFAR10
vgg11_cifar10 = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    n_inputs = 20, # Number of input samples to test
    max_epochs = 500, # Maximum number of epochs.
    batch_size = None, # Batch size if training with SGD.
    img_size_x = 32, # Size of image x dimension.
    img_size_y = 32, # Size of image y dimension.
    net_style = 'vgg11',
    dataset_name = 'cifar10', # Use imagenet inputs.
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    pool_over_group = False, # group before fitting the linear classifier.
    pool=None,
    fit_intercept = False, 
    center_response = False,  
)
alphas = torch.linspace(0.5, 3.0, 15)
layer_idx = [2, 3, 6]
n_channels = alphas_to_channels(
    alphas, vgg11_cifar10['n_inputs'],
    int(vgg11_cifar10['fit_intercept']))
param_sets.update(
vgg11_cifar10_exps = exps_channels_and_layers(vgg11_cifar10, n_channels,
                                              layer_idx)
)
vgg11_cifar10_exps = exps_channels_and_layers(vgg11_cifar10, n_channels,
                                              layer_idx)

vgg11_cifar10_circular = vgg11_cifar10.copy()
vgg11_cifar10_circular.update(
    net_style='vgg11_circular',
)
param_sets.update(
vgg11_cifar10_circular_exps = exps_channels_and_layers(
    vgg11_cifar10_circular, n_channels, layer_idx)
)

vgg11_cifar10_efficient = vgg11_cifar10_circular.copy()
vgg11_cifar10_efficient.update(
    perceptron_style='efficient',
)
param_sets.update(
vgg11_cifar10_efficient_exps = exps_channels_and_layers(
    vgg11_cifar10_efficient, n_channels, layer_idx)
)

vgg11_cifar10_gpool = vgg11_cifar10_circular.copy()
vgg11_cifar10_gpool.update(
    pool_over_group=True,
)
param_sets.update(
vgg11_cifar10_gpool_exps = exps_channels_and_layers(
    vgg11_cifar10_gpool, n_channels, layer_idx)
)


## Grid cell net
grid_2d_conv = dict(
    n_dichotomies = 100, # Number of random dichotomies to test
    n_inputs = 16, # Number of input samples to test
    max_epochs = 500, # Maximum number of epochs.
    batch_size=None,
    img_size_x = 40, # Size of image x dimension.
    img_size_y = 40, # Size of image y dimension.
    net_style = 'grid', # Not fully implemented. Grid cell CNN.
    layer_idx=0,
    dataset_name = 'gaussianrandom', # Use Gaussian random inputs.
    shift_style = '2d', # Use input shifts in both x and y dimensions
    shift_x = 1, # Number of pixels by which to shift in the x direction
    shift_y = 1, # Number of pixels by which to shift in the y direction
    pool_over_group = False, # group before fitting the linear classifier.
    fit_intercept = False,
    center_response = False,  
)
alphas = torch.linspace(.8, 3.5, 20)
n_channels = alphas_to_channels(
    alphas, grid_2d_conv['n_inputs'],
    int(grid_2d_conv['fit_intercept']))
n_channels = [int(x/2) for x in n_channels]
param_sets.update(
grid_2d_conv_exps=exps_channels_and_layers(grid_2d_conv, n_channels)
)

