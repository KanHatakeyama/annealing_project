import numpy as np
import re
from tqdm import tqdm
import pandas as pd
import random

from SimilarFPAnalyzer import SimilarFPAnalyzer
from LiPolymerDataBase import MAX_SMILES
from DAWrapper import find_best_bit_DA,find_best_bit_BQ
from Fingerprint import get_Tanimoto


class Dummy:
    def __init__(self):
        pass


class CompositeAnalyzer:
    """
    utility class to analyze the result of composite conductor exploration
    """

    def __init__(self, bit_compressor, qubo_util,
                 auto_bit_converter,
                 data_scaler,
                 column_list,
                 target_param):
        """
        Parameters
        -----------------
        bit_compressor: BitCompressor class
            use the one which was employed during dataset preparation
        qubo_util: QUBOYtiloty class
            use the one which was employed during dataset preparation
        data_scaler: LiPolymerDataScaler class
            use the one which was employed during dataset preparation
        column_list: list of string
            list of columns of a dataframe used for machine learning
        target_param: string
            column name of the target parameter (e.g., Conductivity)

        """
        self.bit_compressor = bit_compressor

        # TODO: for SimilarFPAnalyzer, a dummy class is used...
        dum = Dummy()
        dum.bit_compressor = bit_compressor
        self.qubo_util = qubo_util
        self.similar_amalyzer = SimilarFPAnalyzer(target_param, dum)
        self.target_param = target_param
        self.auto_bit_converter = auto_bit_converter
        self.data_scaler = data_scaler

        self.column_list = column_list
        self.column_list.remove(self.target_param)
        self.column_list = np.array(self.column_list)

    def find_actual_best_x(self, train_x, train_y):
        """
        explore database to find the best condition giving the highest y

        Parameters
        ---------------
        train_x: np array
            train dataset (x)
        train_y: np array
            train dataset (y)

        Returns
        -----------------
        max_x: np array (bit list?)
            found best X
        """

        max_y = max(np.array(train_y))
        ind = np.where(train_y == max_y)
        max_x = np.array(train_x)[ind]
        max_x = self.qubo_util.get_uninteracted_bit(max_x[0])

        self.act_best_x = max_x
        return max_x

    def find_da_best_x(self, model, mix_ratio=0.001):
        """
        find the best composition according to the result of DA

        Parameters
        -------------------------
        model: sklearn model
            trained model
        mix_ratio: float
            how much the actual best fingerprint is affected

        Returns
        --------------------
        ideal_bit: np array
            found best x by DA
        """

        """
        ideal_bit = find_best_bit_DA(
            model, self.qubo_util, mix_ratio, [self.act_best_x])
        """
        ideal_bit = find_best_bit_BQ(
            model, self.qubo_util, mix_ratio, [self.act_best_x])        
        
        return ideal_bit

    def set_ideal_bit(self, original_ideal_bit):
        """
        set ideal bit in this class. this should be called before analyzing the database

        Parameters
        ------------------
        original_ideal_bit: bit list
            decompressed fingerprint 
        """

        self.fp_dict = {}
        self.fp_range_dict = {}
        self.weight_dict = {}
        self.weight_range_dict = {}

        for i in range(MAX_SMILES):
            name = "fp_list"+str(i)
            fp_range = self.get_bit_column_range(self.column_list, name)
            self.fp_dict[i] = np.array(original_ideal_bit)[fp_range]
            self.fp_range_dict[i] = fp_range

            name = "SMILES_wt_list"+str(i)
            t_range = self.get_bit_column_range(self.column_list, name)
            bit = np.array(original_ideal_bit)[t_range]

            float_data = self.auto_bit_converter.bit_converter_dict[name].bit_to_float(
                bit)
            value = self.data_scaler.scaling_dict[name].inverse_transform(
                np.array(float_data).reshape(1, -1))
            self.weight_dict[name] = value
            self.weight_range_dict[name] = t_range

    def compare_with_database(self, compound_database):
        """
        explore similar compounds with the those in the ideal composite

        Parameters
        ------------------
        compound_database: CompoundDatabase class
            use the one which was employed during dataset preparation

        Returns
        --------------------
        found_comp_dict: dict of dataframe
            found_comp_dict[i] is a dataframe of compounds (sorted by similarity with the ideal bit). i=0 to MAX_SMILES
        best_comp_dict: dict of string
            dict of best compounds (with the highest similarity)

        """
        self.smiles_list = list(compound_database.fp_dict.keys())
        self.compound_database = compound_database

        self.found_comp_dict = {}
        self.best_comp_dict = {}
        for i in tqdm(range(MAX_SMILES)):
            fp = self.fp_dict[i]
            sim_dict = {k: get_Tanimoto(
                fp, v) for k, v in self.compound_database.fp_dict.items()}

            sim_df = pd.DataFrame.from_dict(sim_dict, orient="index")
            sim_df = sim_df.reset_index()
            sim_df.columns = ["SMILES", "sim"]
            sim_df = sim_df.sort_values(by="sim", ascending=False)
            sim_df = sim_df.reset_index()

            self.found_comp_dict[i] = sim_df
            self.best_comp_dict[i] = sim_df["SMILES"][0:1].values

        return self.found_comp_dict, self.best_comp_dict

    def get_best_condition(self):
        """
        get best condition according to the result of DA

        Returns
        ----------------
        condition: list of array
             best condition
        """
        condition = [(i[0], j[0][0]) for i, j in zip(
            self.best_comp_dict.values(), self.weight_dict.values())]
        return condition

    def get_random_da_condition(self, max_N=10):
        """
        get good condition according to the result of DA with a random process

        Parameters
        -------------
        max_N: int
            ca. (best ~ max_N)th most similar compounds will be selected randomly. (using Gaussian ditribution)

        Returns
        ---------------
        condition: list of array
            selected condition
        """
        condition = []
        for i, j in zip(self.found_comp_dict.values(), self.weight_dict.values()):
            n = int(abs(np.random.randn())*max_N)
            compound = list(i["SMILES"][n:n+1])[0]
            weight = j[0][0]
            condition.append((compound, weight))

        return condition

    def get_random_condition(self):
        """
        get random conditon

        Returns
        ----------------
        condition: list of array
            selected condition
        """
        rand_cond=[[random.choices(self.smiles_list)[0], random.random()] for i in range(MAX_SMILES)]
        rand_cond[0][1]=0
        return rand_cond

    def get_bit_column_range(self, column_list, name):
        """
        get range of the target parameter in the database
        """
        target_column_list = list(
            column_list[[True if re.match(name, i) else False for i in column_list]])
        target_range = np.where(np.isin(column_list, target_column_list))
        return target_range
