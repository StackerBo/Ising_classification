import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

def init(L, mode):
    """Initialize the Ising model"""
    if mode == 'hot':
        state = np.random.choice([1,-1], size = (L,L))
    elif mode == 'cold':
        state = np.ones((L,L), dtype = int)
    return state

def generate_next_step(state, T, L, J):
    """Perform a single Metropolis step"""
    # Choose a random site
    i = np.random.randint(L)
    j = np.random.randint(L)
    
    # Calculate Delta E
    dE = 2*state[i,j]*J*np.sum([state[(i+1)%L, j], state[i, (j+1)%L], state[(i-1)%L, j], state[i, (j-1)%L]])
    
    # Acceptance probability
    alpha = np.exp(-dE/T)
    
    # Accept or reject
    r = np.random.rand()
    if r < alpha:
        state[i,j] *= -1
    return state

def Metropolis(L, T, J, total_steps, sampling_steps, save_file, mode = 'hot'):
    """Perform the Metropolis algorithm"""
    state = init(L, mode)
    
    samples = []
    
    for i in range(total_steps):
        state = generate_next_step(state, T, L, J)
        
        if i % sampling_steps == 0 and i > (10**5):
            np_state = np.array(state)
            # state_flatten = np_state.flatten()

            samples.append(np_state)

    np_state = np.array(samples)
    np.save(save_file, samples)