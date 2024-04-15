import argparse
from mc import get_action, get_drift, get_hamiltonian, hmc
import copy
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm 
import numpy as np
import os

parser = argparse.ArgumentParser(description='Run Monte Carlo simulations.')
parser.add_argument('--L', type=int, required=True, default=16, help='L parameter')
parser.add_argument('--k', type=float, required=True, default=0.5, help='k parameter')
parser.add_argument('--l', type=float, required=True, default=0.022, help='l parameter')
parser.add_argument('--chains', type=int, required=True, default=1024, help='Number of chains')
parser.add_argument('--nk', type=int, required=True, default=10, help='nk parameter')
args = parser.parse_args()

# superparameters
L = args.L
k = args.k
l = args.l
chains = args.chains
nk = args.nk

filename = 'data/cfgs_L{}_k{}_l{}_{}k.npy'.format(L,k,l,nk)

def runhmc(chains):
    local_cfgs = []
    local_acc = []
    for chain_idx in tqdm(range(chains), desc="Chain Progress"):
        phi = np.random.randn(L,L)
        S = get_action(phi, k, l)
        for i in range(all_steps):
            phi, S, accepted = hmc(phi, S, k, l)
            local_acc.append(accepted)
            if i % eq_step == 0 and i > therm_step:
                local_cfgs.append(copy.deepcopy(phi))
    return local_cfgs

if os.path.exists(filename):
    print('load from file') 
    cfgs = np.load(filename)

else:
    print('generating new cfgs')
    therm_step = 100
    eq_step = 64
    all_steps = (therm_step + eq_step)
                
    # prepare training data-set
    chainlist = list(chains for i in range(nk))

    with ProcessPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(runhmc, chainlist))

    # Accumulating and reshaping results
    all_cfgs = [cfg for sublist in results for cfg in sublist]
    cfgs = np.array(all_cfgs).reshape(-1, L, L)
    print(cfgs.shape)

    np.save(filename, cfgs)