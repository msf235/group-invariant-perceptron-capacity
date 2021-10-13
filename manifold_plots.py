import torch
from matplotlib import pyplot as plt
import math

π = math.pi
torch.manual_seed(4)
X = torch.randn(2, 2)
Y = torch.zeros(2)
Y[0] = 1
s1 = 320
s2 = 26000

s1 = 160
s2 = 120

def R(θ):
    θ = torch.tensor(θ)
    return torch.tensor([[torch.cos(θ), -torch.sin(θ)],
                         [torch.sin(θ), torch.cos(θ)]])

# %% Figure 1a
n = 4
θs = torch.arange(0, 2*π, 2*π/n) 
RX = torch.stack([X@R(θ).T for θ in θs])
fig, ax = plt.subplots(figsize=(4,4))
ax.scatter(*RX[0,0].T, c='orange', marker='*', s=s1)
ax.scatter(*RX[1:,0].T, c='orange', s=s2)
ax.scatter(*RX[0,1].T, c='g', marker='*', s=s1)
ax.scatter(*RX[1:,1].T, c='g', s=s2)
ax.scatter([0], [0], s=s1, facecolors='none', edgecolors='b')
fig.show()
fig.savefig('rot_mat_orbits.pdf')

# %% Figure 1b
torch.manual_seed(5)
X = torch.randn(2, 3)
Y = torch.zeros(2)
Y[0] = 1

def R(θ):
    θ = torch.tensor(θ)
    return torch.tensor([[torch.cos(θ), -torch.sin(θ), 0],
                         [torch.sin(θ), torch.cos(θ), 0],
                         [0,0,1]])

RX = torch.stack([X@R(θ).T for θ in θs])
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*RX[0,0].T, c='orange', marker='*', s=s1)
ax.scatter(*RX[1:,0].T, c='orange', s=s2)
ax.scatter(*RX[0,1].T, c='g', marker='*', s=s1)
ax.scatter(*RX[1:,1].T, c='g', s=s2)
ax.plot([0, 0], [0, 0], [-1.8, 2])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-2, 2])
ax.view_init(azim=-38, elev=19)
ax.set_xticks([-1, -.5, 0, .5, 1])
ax.set_yticks([-1, -.5, 0, .5, 1])
ax.set_zticks([-2, -1, 0, 1, 2])
# ax.set_azim(-38)
# ax.set_elev(19)
ax.grid(False)
fig.savefig('rot_mat_orbits_3d.pdf')
fig.show()


# %% Figure 1c
torch.manual_seed(1)
X = torch.randn(2, 3)
def Ck(x, k):
    return torch.roll(x, k, dims=[-1])
RX = torch.stack([Ck(X,k) for k in range(3)])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(*RX[0,0].T, c='orange', marker='*', s=s1)
ax.scatter(*RX[1:,0].T, c='orange', s=s2)
ax.scatter(*RX[0,1].T, c='g', marker='*', s=s1)
ax.scatter(*RX[1:,1].T, c='g', s=s2)
ax.plot([-.3, .5], [-.3, .5], [-.3, .5])
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-2, 2])
ax.view_init(azim=-38, elev=19)
ax.set_xticks([-1, -.5, 0, .5, 1])
ax.set_yticks([-1, -.5, 0, .5, 1])
ax.set_zticks([-1, -.5, 0, .5, 1])
# ax.set_azim(-38)
# ax.set_elev(19)
ax.grid(False)
ax.axis('auto')
fig.savefig('cyc_mat_orbits_3d.pdf')
fig.show()
