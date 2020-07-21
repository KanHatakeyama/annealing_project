import numpy as np
import matplotlib.pyplot as plt
from Fingerprint import get_Tanimoto

class SimilarFPAnalyzer:
    def __init__(self, target_param_name, smiles_converter, similarity_function=get_Tanimoto):
        """
        utility class to explore similar smiles

        Parameters
        -----------
        target_param_name: string
            name of the target parameter
        smiles_converter: SMILESConverter class
            smiles_converter class
        similarity_function: func
            function to compare fingerprints
        """
        self.similarity_function = similarity_function
        self.target_param_name = target_param_name
        self.smiles_converter = smiles_converter

    def calc_similarity(self, best_fingerprint, x_db):
        """
        calculate fingerprint of the target bit with the database

        Parameters
        ------------
        best_fingerprint: bit array
            fingerprint found by e.g., DA
        x_db: np array
            use e.g., x_train

        Returns
        ----------------
        similarity_list: list of float
           similarity of the target bit with the data in the database 

        """
        similarity_list = [self.similarity_function(
            best_fingerprint, FP) for FP in self.smiles_converter.bit_compressor.inverse_transform(x_db)]
        return similarity_list

    #TODO: rather redundant
    def compare_with_data_base(self, best_fingerprint, x_all, y_all, plot=True, percent=5):
        """
        automatically analyze the relationships among best fingerprint and the database

        Parameters
        -----------------
        best_fingerprint: bit array
            fingerprint found by e.g., DA
        x_all: np array
            e.g., x_train
        y_all: np array
            e.g., y_train
        plot: bool
            if true, draw plot
        percent: float
            calculate results with the top "percent" of the database

        Returns
        -----------------------
        similarity_list: list of float
            list of similarity with the best fp
        y_all
            list of  y
        """

        similarity_list = self.calc_similarity(best_fingerprint, x_all)

        # remove array_of_one with similarity==1 for plot
        dataset = zip(similarity_list, y_all)
        dataset = [i for i in dataset if i[0] != 1]
        x, y = zip(*dataset)

        if plot:
            plt.scatter(x, y, c="b", s=3, alpha=0.5)
            plt.xlabel("similarity")
            plt.ylabel("performance")

        eff = (self.calc_extraction_efficiency(similarity_list, y_all))
        #print("{:.1f} %  extraction efficiency (percent = {})".format(eff, percent))

        sx, sy, sy2 = (self.get_extraction_scores(
            similarity_list, y_all, extract_ratio=percent/100))
        print("ave y: {:.2f}".format(sy))

        return similarity_list, y_all

    def calc_extraction_efficiency(self, similarity_list, y_all, percent=5):
        """
        kind of legacy core. higher is better
        """
        data_size = len(y_all)
        extract_size = int(data_size*percent/100)
        dataset = np.array(list(zip([i[0] for i in y_all], similarity_list)))

        # sort by value
        dataset = dataset[np.argsort(dataset[:, 0])][::-1]

        flag_list = np.zeros(data_size)
        flag_list[:extract_size] = 1
        flag_list = flag_list.reshape(-1, 1)

        dataset = np.hstack([dataset, flag_list])

        # sort by similarity
        dataset = dataset[np.argsort(dataset[:, 1])][::-1]
        hit = np.sum(dataset[:extract_size], axis=0)[2]

        efficiency = hit/extract_size*100
        return efficiency

    def get_extraction_scores(self, similarity_list, y_all, extract_ratio=0.05):
        """
        kind of legacy
        """
        dataset = list(zip(similarity_list, y_all))
        dataset.sort(reverse=True)
        selected_dataset = dataset[:int(len(dataset)*extract_ratio)]
        sx, sy = list(zip(*selected_dataset))

        return np.mean(sx), np.mean(sy), np.std(sy)

    def explore_actual_best_fingerprint(self, df, i,smiles_column="SMILES"):
        """
        extract "i"th best compound in the database

        Parameters
        -------------
        df: dataframe
            database
        i: int
            extract "i"th best compound
        
        Returns
        --------------
        actual_best_fingerprint: bit array
            best fingerprint in the database
        return2: bit array
            its compressed fingerprint
        """
        i = i-1
        SMILES = df.sort_values(self.target_param_name, ascending=False)[
            smiles_column][i:(i+1)]
        bit = self.smiles_converter.transform(SMILES)
        actual_best_fingerprint = self.smiles_converter.bit_compressor.inverse_transform(bit)[
            0]
        return actual_best_fingerprint, [self.smiles_converter.qubo_utility.get_uninteracted_bit(bit[0])]
