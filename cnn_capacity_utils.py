import numpy as np
import matplotlib.pyplot as plt

def compute_pi_mean_reduced(L, k):
    base_v = np.zeros(L)
    base_v[::k] = 1 / (int(L/k))
    pi = np.zeros((k, L))
    for i in range(k):
        pi[i] = np.roll(base_v, i)
    return pi

def compute_pi_mean_reduced_2D(L, W, kx, ky):
    pi_mean_1d_L = compute_pi_mean_reduced(L, kx)
    pi_mean_1d_W = compute_pi_mean_reduced(W, ky)
    return np.kron(pi_mean_1d_L, pi_mean_1d_W)

def compute_pi_mean_reduced_grid_2D(L_tuple):
    L2d = sum([L**2 for L in L_tuple])
    pi = np.zeros((len(L_tuple), L2d))
    prev_ind = 0
    for i0, L in enumerate(L_tuple):
        pi_L = compute_pi_mean_reduced_2D(L, L, 1, 1)
        pi[i0, prev_ind:prev_ind+L**2] = pi_L
        prev_ind = prev_ind + L**2
    return pi
