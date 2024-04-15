import copy
import numpy as np

def action(phi: np.ndarray, k, l):
    return np.sum(-2 * k * phi * (np.roll(phi, 1, 0) + np.roll(phi, 1, 1))+ (1 - 2 * l) * phi**2 + l * phi**4,axis=(1,2))

def get_action(phi, k, l):
    return np.sum(-2 * k * phi * (np.roll(phi, 1, 0) + np.roll(phi, 1, 1))
                  + (1 - 2 * l) * phi**2 + l * phi**4)

def get_drift(phi, k, l):
    return (2 * k * (np.roll(phi, 1, 0) + np.roll(phi, -1, 0)
                     + np.roll(phi, 1, 1) + np.roll(phi, -1, 1))
            + 2 * phi * (2 * l * (1 - phi**2) - 1))

def get_hamiltonian(chi, action):
    return 0.5 * np.sum(chi**2) + action

def hmc(phi_0, S_0, k, l, n_steps=100):
    dt = 1 / n_steps

    phi = phi_0
    chi = np.random.randn(*phi.shape)
    H_0 = get_hamiltonian(chi, S_0)

    chi += 0.5 * dt * get_drift(phi, k, l)
    for i in range(n_steps-1):
        phi += dt * chi
        chi += dt * get_drift(phi, k, l)
    phi += dt * chi
    chi += 0.5 * dt * get_drift(phi, k, l)

    S = get_action(phi, k, l)
    dH = get_hamiltonian(chi, S) - H_0

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, False
    return phi, S, True

def mc(phi_0, S_0, k, l):
    phi = phi_0
    chi = np.random.randn(*phi.shape)
    H_0 = get_hamiltonian(chi, S_0)

    phi += chi
    S = get_action(phi, k, l)
    dH = get_hamiltonian(chi, S) - H_0

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, False
    return phi, S, True

def dm_mc(phi_0, S_0,logq_0, k, l, cfgs_df,logq_df):

    index = np.random.choice(cfgs_df.shape[0])
    phi = cfgs_df[index]
    
    last_logp = - S_0
    last_logq = logq_0
    S = get_action(phi, k, l)
    new_logp = - S
    new_logq = logq_df[index]

    dH = (last_logp - last_logq) - (new_logp - new_logq)

    if dH > 0:
        if np.random.rand() >= np.exp(-dH):
            return phi_0, S_0, logq_0, False
    return phi, S, new_logq, True

