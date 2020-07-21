"""
Wrapper class of Fujitsu digital annealer 2 (DA)
"""


#from python_fjda import fjda_client, fjda_int
import numpy as np
import copy
from QUBOUtility import get_pertubated_coeff
from BlueqatWrapper import auto_bit_search_by_blueqat


class DAWrapper:
    def __init__(self, return_raw_data=False, return_minimum=True, return_all=False):
        """
        *** removed***
        this wrapper class can be used only with digital annealer environment
        """

def find_best_bit_DA(model, qubo_util, mix_ratio, act_best_bit):
    """
    automatic wrapper function to find best bit by DA

    Parameters
    -----------------
    model: sklearn model object
        linear-type, sklearn prediction model (e.g., lasso, SGD, ...). it should have ".coef_" as QUBO
    qubo_util: QUBOUtility class
        --
    mix_ratio: float:
        how much the effect of "act_best_bit" is provided to qubo
    act_best_bit: [list of int (01010111....)]
        binary data to affect qubo

    Returns
    ---------------
    ideal_bit: np array of [010101011...]
        best bit found by DA
    """

    # give pertubation effect to qubo, using act_best_bit
    mix_coeff = get_pertubated_coeff(model.coef_, act_best_bit[0], mix_ratio)
    scale = 10**5

    # find best val by DA
    mat = qubo_util.calculate_QUBO_matrix(mix_coeff)
    daw = DAWrapper(return_all=True)
    daw.setting_dict["scale"] = scale
    daw.setting_dict["init_state"] = "zeros"

    state, res = daw.run(-mat)
    en = res["eg_min_o_n.numpy"]

    min_energy = min(en)
    bin_len = mat.shape[0]
    ideal_bit = state[np.where(en == min_energy)[0][0]][:bin_len]

    return ideal_bit


def find_best_bit_BQ(model, qubo_util, mix_ratio, act_best_bit):
    """
    automatic wrapper function to find best bit by BQ

    Parameters
    -----------------
    model: sklearn model object
        linear-type, sklearn prediction model (e.g., lasso, SGD, ...). it should have ".coef_" as QUBO
    qubo_util: QUBOUtility class
        --
    mix_ratio: float:
        how much the effect of "act_best_bit" is provided to qubo
    act_best_bit: [list of int (01010111....)]
        binary data to affect qubo

    Returns
    ---------------
    ideal_bit: np array of [010101011...]
        best bit found by DA
    """

    # give pertubation effect to qubo, using act_best_bit
    mix_coeff = get_pertubated_coeff(model.coef_, act_best_bit[0], mix_ratio)
    scale = 10**5

    # find best val by DA
    mat = qubo_util.calculate_QUBO_matrix(mix_coeff)
    ideal_bit, min_energy=auto_bit_search_by_blueqat(mat)

    return ideal_bit
