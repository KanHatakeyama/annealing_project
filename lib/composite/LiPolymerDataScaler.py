import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from DataUtility import get_column_names


class LiPolymerDataScaler:
    """
    a special class to scale the lithium polymer database
    """

    def __init__(self):
        self.scaling_dict = {}
        self.main_val_params = ["SMILES_wt", "wt_ratio", "inorg_contain_ratio"]
        self.main_txt_params = ["structureList", "inorg_name"]
        self.main_params = self.main_val_params+self.main_txt_params
        self.target_param = "Conductivity"

    def mutual_process(self, df):
        """
        convert values (to log, etc)
        """
        df["Conductivity"] = np.log10(df["Conductivity"].astype('float'))
        df["Temperature"] = np.log10(df["Temperature"].astype('float')+273)

        # fill Nan by zero
        for c in self.main_params:
            target_columns = get_column_names(df, c)
            df[target_columns] = df[target_columns].fillna(0)

        # convert molecular weight
        self.mw_column_list = get_column_names(df, "MWList")
        for c in self.mw_column_list:
            df[c] = np.log10(df[c].astype('float'))

        return df

    def fit_transform(self, original_df):
        """
        scaling data, etc

        Parameters
        ----------------
        original_df: dataframe
            dataframe to be scaled

        Returns
        ----------------
        df: dataframe
            scaled dataframe
        """
        df = original_df.copy()
        df = self.mutual_process(df)

        # fill lacking Molecular weight with average value
        self.average_mw = sum(df[self.mw_column_list].sum()) / \
            sum(df[self.mw_column_list].count())

        for c in self.mw_column_list:
            df[c] = df[c].fillna(self.average_mw)

        # scaling
        for v in self.main_val_params + ["Conductivity", "Temperature"]+self.mw_column_list:
            for c in get_column_names(df, v):
                sc = StandardScaler()
                df[c] = sc.fit_transform(
                    df[c].astype('float').values.reshape(-1, 1))
                self.scaling_dict[c] = sc

        # onehot encoding
        for v in self.main_txt_params:
            df = pd.get_dummies(df, columns=get_column_names(df, v))

        self.use_columns = []

        for c in ["Conductivity", "Temperature"]+self.main_params + self.mw_column_list+["fp_list"]:
            self.use_columns.extend(get_column_names(df, c))

        """ 
        **********************************************************
        delete some columns for easiness of machine learning
        following parameters can be useful for machine learning (10.1021/jacs.9b11442), but ignored in this project.
        """
        for remove_targets in ["MWList", "wt_ratio", "inorg", "structure", "Temperature"]:
            del_columns = get_column_names(df, remove_targets)
            for i in del_columns:
                self.use_columns.remove(i)

        self.tr_df = df
        return df

    def transform(self, original_df):
        """
        scaling data, etc

        Parameters
        ----------------
        original_df: dataframe
            dataframe to be scaled

        Returns
        ----------------
        df: dataframe
            scaled dataframe
        """
        df = original_df.copy()
        df = self.mutual_process(df)

        for c in self.mw_column_list:
            df[c] = df[c].fillna(self.average_mw)

        # scaling
        for v in self.main_val_params + ["Conductivity", "Temperature"]+self.mw_column_list:
            for c in get_column_names(df, v):
                df[c] = self.scaling_dict[c].transform(
                    df[c].astype('float').values.reshape(-1, 1))

        # onehot encoding
        for v in self.main_txt_params:
            df = pd.get_dummies(df, columns=get_column_names(df, v))

        # for lacking columns, add the most frequent vals
        lacking_columns = set(self.use_columns)-set(df.columns)

        for i in lacking_columns:
            df[i] = self.tr_df[i].mode()

        return df
