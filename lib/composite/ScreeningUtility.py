import numpy as np
from tqdm import tqdm
import time
import joblib


class ScreeningUtility:
    """
    utility class to explore new lithium conducting composites
    """

    def __init__(self, composite_analyzer, model, log_file_path="",verbose=False):
        """
        Parameters
        ------------------
        composite_analyzer: CompositeAnalyzer class
            please initialize composite_analyzer before this class
        model: sklean model
            trained model
        log_gile_path: string
            if not "", prediction log will be recored in the file

        """
        self.composite_analyzer = composite_analyzer
        self.model = model
        self.bit_len = len(composite_analyzer.column_list)
        self.log_file_path = log_file_path
        self.verbose = verbose

        if self.log_file_path != "":
            self.time_list = []
            self.y_list = []
            self.x_list = []

    def prepare_bit(self, condition):
        """
        prepare input bit (= x) from the specific condition

        Parameters
        ------------------
        condition: list
            e.g., ("C", "CC","CCC","CCCC","CCCCC","CCCCCC",  : smiles
                    1,   0.5,  0.3,   0.24,  0.1,    0.05 )  : weight ratio (log scale)

        Returns
        ---------------------
        return: corresponding bit array
        """
        try_bit = np.array([9999]*self.bit_len)
        for num, rng in enumerate(self.composite_analyzer.fp_range_dict.values()):
            smiles = condition[num][0]
            fingerprint = self.composite_analyzer.compound_database.fp_dict[smiles]
            try_bit[rng[0]] = fingerprint

        for num, name in enumerate(self.composite_analyzer.auto_bit_converter.bit_converter_dict.keys()):
            ratio = np.array(condition[num][1]).reshape(-1, 1)
            z_score = self.composite_analyzer.data_scaler.scaling_dict[name].transform(ratio)[
                0][0]
            bit_data = self.composite_analyzer.auto_bit_converter.bit_converter_dict[name].float_to_bit(
                z_score)

            try_bit[self.composite_analyzer.weight_range_dict[name]] = bit_data

        return np.array(try_bit).reshape(1, -1)

    def predict(self, condition):
        """
        predict y from a specific condition

        Parameters
        ------------------
        condition: list
            e.g., ("C", "CC","CCC","CCCC","CCCCC","CCCCCC",  : smiles
                    1,   0.5,  0.3,   0.24,  0.1,    0.05 )  : weight ratio

        Returns
        ---------------------
        y: float
            predicted value (as z-score)
        """
        # log mode
        if self.log_file_path != "":
            if len(self.time_list) == 0:
                self.start_time = time.time()

        try_bit = self.prepare_bit(condition)
        try_bit = self.composite_analyzer.bit_compressor.transform(try_bit)
        try_bit = self.composite_analyzer.qubo_util.calc_interactions(try_bit)
        y = self.model.predict(try_bit)

        # log mode
        if self.log_file_path != "":
            t = time.time() - self.start_time
            self.time_list.append(t)
            self.y_list.append(y)
            self.x_list.append(condition)
            
        if self.verbose:
            print(condition,y)
        
        return y

    def save_log(self):
        joblib.dump((self.time_list, self.x_list,
                     self.y_list), self.log_file_path)

    def auto_loop(self, generation_func, N):
        """
        automatically search compounds from a specific condition_generation function

        Parameters
        -----------------
        generation_func: function
            please use get_best_condition, get_random_da_condition or get_random_condition of CompositeAnalyzer class
        N: int
            number of iterations

        Returns
        ------------------------
        res_list: list of float
            history of y
        best_hist: list of float
            history of best y
        x_list: list of array
            history of x
        """

        y_best = -999
        x_list = []

        res_list = []
        best_hist = []

        for i in tqdm(range(N)):
            condition = generation_func()
            y = self.predict(condition)[0]
            res_list.append((y))

            if y > y_best:
                y_best = y
                x_list.append(condition)
            best_hist.append(y_best)

        return res_list, best_hist, x_list
