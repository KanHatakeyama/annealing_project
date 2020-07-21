import itertools
import pandas as pd
import numpy as np
from BitCompressor import BitCompressor
from QUBOUtility import QUBOUtility
from joblib import Parallel, delayed

# prepare  binary array from smiles


class SMILESConverter:
    def __init__(self, fingerprint, threshold=0.1, verbose=False):
        """
        Parameters
        ----------------
        fingerprint: Fingerprint class
            use Fingerprint class
        threshold: float
            threshold to compress the binary (fingerprint) data. 
        """

        self.threshold = threshold
        self.fingerprint = fingerprint
        self.verbose = verbose

    def fit_transform(self, SMILES_list):
        """
        calculate FPs from SMILEs, compress, and calculate interactions (e.g, x1*x2)

        Parameters
        ---------------------
        SMILES_list: list of str
            list of smiles

        Returns
        --------------------
        return: np array
            np array of compressed fingerprints after calcualting xi_xj
        """

        fingerprint_list, _ = self.fingerprint.calc_fingerprint(SMILES_list)

        self.bit_compressor = BitCompressor(self.threshold)
        compound_bit_list = self.bit_compressor.fit_transform(fingerprint_list)

        self.qubo_utility = QUBOUtility()

        if self.verbose:
            print("calculating interactions...")

        X_list = self.qubo_utility.calc_interactions(compound_bit_list)

        if self.verbose:
            print("xshape: ", len(X_list[0]))

        return np.array(X_list)

    def transform(self, SMILES_list):
        """
        calculate FPs from SMILEs, compress, and calculate interactions (e.g, x1*x2). Compressor is not initialized.

        Parameters
        ---------------------
        SMILES_list: list of str
            list of smiles

        Returns
        --------------------
        return: np array
            np array of compressed fingerprints after calcualting xi_xj
        """

        fingerprint_list, _ = self.fingerprint.calc_fingerprint(SMILES_list)
        compound_bit_list = self.bit_compressor.transform(fingerprint_list)

        X_list = self.qubo_utility.calc_interactions(compound_bit_list)

        return np.array(X_list)
