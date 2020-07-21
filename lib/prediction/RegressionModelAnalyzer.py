import copy
from tqdm import tqdm


class RegressionModelAnalyzer:
    """
    utility class of analyzing the model
    
    Parameters
    ---------------
    model: sklearn model
        trained sklearn model
    smiles_converter: SMILESConverter class
        -
    """
    def __init__(self, model, smiles_converter):
        self.model = model
        self.smiles_converter = smiles_converter
        self.best_Y = None
        self.ideal_fingerprint = None

    def predict_from_compound_bit(self, bit):
        """
        predict y from the compressed fingerprint

        Parameters
        ------------
        bit: bit array
            compressed fignerprint

        Returns
        --------------
        return: numpy arry
            predicted y
        """
        res = self.smiles_converter.qubo_utility.calc_interactions(bit)
        return self.model.predict(res)

    def set_ideal_bit(self, ideal_bit):
        self.best_Y = self.predict_from_compound_bit(ideal_bit)[0]
        ideal_fingerprint = self.smiles_converter.bit_compressor.inverse_transform([
                                                                                   list(ideal_bit)[0]])[0]
        self.ideal_fingerprint = ideal_fingerprint

    def check_init(self):
        if self.best_Y == None or self.ideal_fingerprint == None:
            raise ValueError("call set_ideal_bit before this function!")


    #legacy codes

    # calculate contribution of FP[i] for the ideal value
    def calc_diff_y_from_bit_i(self, i):
        """
        i: ideal_fingerprint[i] will be set to be zero
        """
        self.check_init()

        if self.smiles_converter.bit_compressor.filter_dict[i] == False:
            return 0

        tryFP = [copy.deepcopy(self.ideal_fingerprint)]
        tryFP[0][i] = 0

        bit = self.smiles_converter.bit_compressor.transform(tryFP)
        y = self.predict_from_compound_bit(bit)[0]

        return self.best_Y-y

    # automatic processing
    def calc_contributions_of_each_bit(self):
        self.check_init()

        cont_dict = {i: self.calc_diff_y_from_bit_i(
            i) for i in tqdm(range(len(self.ideal_fingerprint)))}
        cont_dict = sorted(cont_dict.items(), key=lambda x: x[1], reverse=True)
        cont_dict = [i for i in cont_dict if abs(i[1]) > 0]
        cont_dict = {i[0]: i[1] for i in cont_dict}
        return cont_dict

    # simply get the bit_ID
    def get_fingerprint_ids(self):
        return [i for i, j in enumerate(self.ideal_fingerprint) if j == 1]
