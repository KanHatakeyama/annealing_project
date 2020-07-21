"""
Wrapper functon of blueqat

"""

from blueqat import opt
import numpy as np


def auto_bit_search_by_blueqat(mat, shots=100):
    """
    search ideal binary by blueqat

    Parameters
    ----------------------------
    mat: np array
        qubo matrix 
    shots: int
        number of shots for annealing
    Returns
    ---------------
    ideal_bit: np array of 01010101..
        best binary found by blueqat
    min_energy: np
        its energy    
    """

    an = opt.opt()
    an.qubo = -mat

    result = an.sa(shots=shots, sampler="fast")

    en = [i[-1] for i in an.E]
    min_energy = min(en)
    ideal_bit = result[np.where(en == min_energy)[0][0]]

    return ideal_bit, min_energy
