import itertools
import numpy as np
from tqdm import tqdm

# utility class to prepare QUBO matrix,etc


class QUBOUtility:

    def __init__(self, verbose=False):
        self.var_dict = {}
        self.bit_length = 0
        self.verbose = verbose

    def prepare_coeff_dict(self, bit_list):
        """
        this function must be called before the main func, calc_interactions
        prepare dict to calculate interactions xi*xj

        Parameters
        --------------
        bit_list: list of bit (1101010...)

        """

        # prepare dict to calculate interactions, such as x1x2.
        # TODO: dict values are not used in this class. unnecessary
        if self.var_dict == {}:
            var_dict = {"val"+str(k): v for k, v in enumerate(bit_list)}
            interaction_dict = {v[0]+"*"+v[1]: var_dict[v[0]]*var_dict[v[1]]
                                for v in itertools.combinations(var_dict.keys(), 2)}
            var_dict.update(interaction_dict)
            self.var_dict = var_dict
            self.bit_length = len(bit_list)
        return

    def calc_interactions(self, ls):
        """
        calculate interactions xi*xj for list  of bit list

        Parameters
        ------------------
        ls: array of bit (1010101...)
            (x1, x2,...)

        Returns
        -------------------
        combined_array: array of bit (10101...)
            (x1, x2,...,x1x2, x1x3,....)
        """

        one_list = ls[0]
        self.prepare_coeff_dict(one_list)

        if self.verbose:
            print("total iteration {}".format(1/2*len(ls)**2))

        npList = np.array(ls)
        inter_list = [(npList[:, v[0]]*npList[:, v[1]])
                      for v in (itertools.combinations((range(len(one_list))), 2))]

        if self.verbose:
            print("recording to memory...")

        inter_list = np.array(inter_list).T
        combined_array = np.concatenate([npList, inter_list], -1)

        return combined_array

    def calculate_QUBO_matrix(self, coeffient_list):
        """
        calculate qubo matrix from the coefficient list of linear regression
        Parameters
        ---------------
        coeffient_list:  np array
            this is made by linear regression. (e.g., model.coef_)

        Returns
        ----------------
        QUBO_matrix: np array
            qubo matrix
        """

        coefficient_dict = {k: v for k, v in zip(
            self.var_dict.keys(), coeffient_list)}
        QUBO_matrix = np.zeros((self.bit_length, self.bit_length))

        for k, v in zip(coefficient_dict.keys(), coefficient_dict.values()):
            k = k.replace("val", "")

            if k.find("*") < 0:
                QUBO_matrix[int(k)][int(k)] = v
            else:
                i, j = k.split("*")
                QUBO_matrix[int(i)][int(j)] = v

        return QUBO_matrix

    def get_uninteracted_bit(self, bit_list):
        """
        get uninteracted bit

        Parameters
        ------------------
        bit_list: array of bit
            (x1, x2,...,x1x2, x1x3,....)

        Returns
        -------------------
        returny: array of bit
            (x1, x2,...)
        """
        return bit_list[:self.bit_length]


def get_pertubated_coeff(coeff, perturbation_bit, mix_ratio):
    """
    calculate pertubation for the model

    Parameters
    ----------------
    coeff: np array
        model.coeff_
    pertubation_bit: array of bit
        e.g., use actual fingerprint
    mix_ratio: float
        amount of pertubation

    Returns
    -----------------
    mix_coeff: np array
        pertubated coeffcient


    """

    perturbation_bit = [1 if i == 1 else -1 for i in perturbation_bit]
    pad_width = (0, coeff.shape[0]-len(perturbation_bit))
    bias = np.pad(perturbation_bit, pad_width, 'constant', constant_values=0)
    mix_coeff = coeff+mix_ratio*bias

    return mix_coeff
