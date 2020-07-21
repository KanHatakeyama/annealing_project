from DecimalBitConverter import DecimalBitConverter
from DataUtility import unnest_dataframe
import copy
import numpy as np
import pandas as pd


class AutoBitConverter:
    """
    a class which convert float columns to binary form (and vice versa)
    """
    def __init__(self, n_bit_float=10):
        """
        Parameters
        ---------------
        n_bit_float: int
            float value will be converted into a bit array of "n_bit_float"
        """
        self.n_bit_float = n_bit_float

    def transform(self, df, use_columns, target_param):
        """
        convert selected float arrays to bit arrays

        Parameters
        ----------------
        df: dataframe
            target dataframe to be converted
        use_columns: list of string
            all colums to be used for machine learning 
        target_param: string
            name of the columns to be used as y
        Returns
        ------------------
        bit_df: dataframe
            converted dataframe
        """
        self.float_columns = [
            i for i in use_columns if df[i].dtype == np.float64]
        # check columns which are not binary
        self.float_columns.remove(target_param)
        self.bit_converter_dict = {}

        bit_df = copy.deepcopy(df[use_columns])

        for column in self.float_columns:
            self.bit_converter_dict[column] = DecimalBitConverter(
                bit_df[column].min(),
                bit_df[column].max(),
                self.n_bit_float)

            bit_df[column] = [self.bit_converter_dict[column].float_to_bit(
                val) for val in bit_df[column].values]

        bit_df = unnest_dataframe(bit_df, self.float_columns, axis=0)
        return bit_df
