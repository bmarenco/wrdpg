'''
Created on Jul 28, 2023

@author: bernardo
'''

import numpy as np
from graspologic.simulations import er_np, sbm
from graspologic.embed import AdjacencySpectralEmbed as ASE
import matplotlib.pyplot as plt
from utils.moments_utils import normal_moments
from utils.embedding_utils import align_Xs

plt.close(fig='all')
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['text.usetex'] = True
plt.rcParams['font.size'] = 28
plt.rcParams['lines.markersize'] = 15
plt.rcParams['axes.grid'] = True

#np.random.seed(42)

######### ER + N(1, 1) for all nodes ###########
n = 1000
p = 0.5
mu = 1
sigma = 0.1
sigma_sq = np.square(sigma)

max_k = 6

# Normal distribution moments
moments_seq = normal_moments(mu,sigma,max_k)
    
X_seq = []

# Compute theoretical latent positions
for moment in moments_seq[1:]:
    X_seq.append(np.sqrt(p*moment))
    
w_norm = np.random.normal
w_norm_args = dict(loc = mu, scale = sigma)

er_graph = er_np(n, p, wt=w_norm, wtargs=w_norm_args)

ase = ASE(n_components=1, diag_aug=True, algorithm='truncated')

Xhats = []

# Embed A^(k)
for k in np.arange(1,max_k+1):
    Xhat = ase.fit_transform(np.power(er_graph,k))
    if np.mean(Xhat) < 0:
        Xhat = -Xhat 
    Xhats.append(Xhat)

fig, axs = plt.subplots(figsize=(12,7),nrows=2, ncols=3, layout='constrained')
axs = axs.flatten()

for idx, ax in enumerate(axs):
    ax.hist(Xhats[idx], bins=20, density=True, rwidth=0.8)
    ax.axvline(X_seq[idx], color = 'maroon', linestyle = '--')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')


######### 2-block SBM + N(1, 1) for all nodes ###########
max_k = 6
N = 1000
ratio = 0.7
n = [int(N*ratio),int(N*(1-ratio))]

p_c1 = 0.7
p_c2 = 0.3
q = 0.1

mu = 1
sigma = 0.1
sigma_sq = np.square(sigma)

P = [[p_c1, q],
     [q, p_c2]]


# Normal distribution moments
moments_seq = normal_moments(mu,sigma,max_k)

X_seq = []

# Compute theoretical latent positions
for moment in moments_seq[1:]:
    X_c1 = np.array([np.sqrt(p_c1*moment),0])
    X_c2 = np.array([q*np.sqrt(moment/p_c1),np.sqrt(moment*(p_c2-np.square(q)/p_c1))])
    X_seq.append(np.vstack((X_c1,X_c2)))

w_norm_args = dict(loc = mu, scale = sigma)
sbm_graph = sbm(n=n, p=P, wt=w_norm, wtargs=w_norm_args)

ase = ASE(n_components=2, diag_aug=True, algorithm='truncated')

Xhats_sbm = []

for k in np.arange(1,max_k+1):
    # Embed A^(k)
    Xhat = ase.fit_transform(np.power(sbm_graph,k))
    # Align with theoretical latent positions
    X = np.empty_like(Xhat)
    X[:n[0]] = X_seq[k-1][0,:]
    X[n[0]:] = X_seq[k-1][1,:]
    rotated_Xhat = align_Xs(Xhat, X)
    Xhats_sbm.append(rotated_Xhat)

fig, axs = plt.subplots(figsize=(10,16),nrows=2, ncols=3, layout='constrained')
axs = axs.flatten() 
colors = n[0]*['maroon'] + n[1]*['royalblue']

for idx, ax in enumerate(axs):
    ax.scatter(Xhats_sbm[idx][:,0],Xhats_sbm[idx][:,1],c=colors,alpha=0.1)
    ax.scatter(X_seq[idx][:,0],X_seq[idx][:,1],c='black',marker='x')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')


######### 2-block SBM + N(5, 0.1) or Poisson(5) ###########
mu = 5
sigma = 0.1
sigma_sq = np.square(sigma)
lam = mu+0.1
N = 2000
ratio = 0.5
n = [int(N*ratio),int(N*(1-ratio))]
p = 0.5

P = [[p, p],
     [p, p]]

wt = [[np.random.normal, np.random.normal],
      [np.random.normal, np.random.poisson]]

w_norm_args = dict(loc = mu, scale = sigma)
w_poisson_args = dict(lam=lam)

wtargs = [[w_norm_args, w_norm_args],
           [w_norm_args, w_poisson_args]]


sbm_graph = sbm(n=n, p=P, wt=wt, wtargs=wtargs)


max_k = 3


# Normal distribution moments
moments_normal = np.empty((max_k+1,))
moments_normal[0] = 1
moments_normal[1] = mu
for k in range(2, max_k+1):
    moments_normal[k] = moments_normal[1]*moments_normal[k-1]+(k-1)*sigma_sq*moments_normal[k-2]
# Poisson distribution moments
moments_poisson = [1, lam, np.square(lam)+lam, np.power(lam,3)+3*np.square(lam)+lam]

X_seq = []

# Compute theoretical latent positions
for moment_normal, moment_poisson in zip(moments_normal[1:],moments_poisson[1:]):
    X_c1 = np.array([np.sqrt(p*moment_normal),0])
    X_c2 = np.array([np.sqrt(p*moment_normal),np.sqrt(p*(moment_poisson-moment_normal))])
    X_seq.append(np.vstack((X_c1,X_c2)))
    
Xhats_sbm = []

ase = ASE(n_components=2, diag_aug=True, algorithm='truncated')

for k in np.arange(1,max_k+1):
    # Embed A^(k)
    Xhat = ase.fit_transform(np.power(sbm_graph,k))
    # Align with theoretical latent positions
    X = np.empty_like(Xhat)
    X[:n[0]] = X_seq[k-1][0,:]
    X[n[0]:] = X_seq[k-1][1,:]
    rotated_Xhat = align_Xs(Xhat, X)
    Xhats_sbm.append(rotated_Xhat)

    
fig, axs = plt.subplots(figsize=(18,6),nrows=1, ncols=3, layout='constrained')
axs = axs.flatten() 
colors = n[0]*['maroon'] + n[1]*['royalblue']

for idx, ax in enumerate(axs):
    ax.scatter(Xhats_sbm[idx][:,0],Xhats_sbm[idx][:,1],c=colors,alpha=0.3)
    ax.scatter(X_seq[idx][:,0],X_seq[idx][:,1],c='black',marker='x')
    ax.set_title(f'$\hat{{\mathbf{{X}}}}[{idx+1}]$')

plt.show()



