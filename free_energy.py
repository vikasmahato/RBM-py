import numpy as np


def free_energy(v_sample, W, vb, hb):
    #Function to compute the free energy
    wx_b = np.dot(v_sample, W) + hb
    vbias_term = np.dot(v_sample, vb)
    hidden_term = np.sum(np.log(1 + np.exp(wx_b)), axis=1)
    return -hidden_term - vbias_term