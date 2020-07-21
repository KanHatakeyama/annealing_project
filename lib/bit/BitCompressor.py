import pandas as pd
import numpy as np

# class to compress binary bits


class BitCompressor:
    def __init__(self, threshold=0.1):
        """
        Parameters
        --------------
        threshold: float
            threshold fo compress. if (1-x)*100% of x_i is the same in the database, x_i is replaced with its most mode.
        """
        self.threshold = threshold

    def calc_variety(self, n):
        """
        calculate variety of the bits

        Parameters
        -------------------
        n: int
            sum of x_i for each data

        Returns
        ------------------
        return: float (0-1)
            how much x_i has the variety. if (000000...0) or (111...11), return will be zero. if (01010101,...) return will be 1

        """
        n = abs(n-self.total_samples/2.0)
        return (1-(n)/self.total_samples*2)

    def filtering(self, n):
        if n > self.threshold:
            return True
        else:
            return False

    def transform(self, bit_list):
        """
        compress bit 
        Parameters
        ------------
        bit_list: list of (list of bit)
                [(101011..1), (100011..), ...]

        Returns
        ------------
        filtered_bit_list: list of (list of bit)
            compressed bit

        """

        bit_list_by_column = zip(*bit_list)
        filtered_bit_list_by_column = [col for col, filt in zip(
            bit_list_by_column, self.filter_dict.values()) if filt == True]
        filtered_bit_list = list(np.array(filtered_bit_list_by_column).T)
        filtered_bit_list = [list(i) for i in filtered_bit_list]

        return filtered_bit_list

    def fit_transform(self, bit_list):
        """
        compress bit
        Parameters
        ------------
        bit_list: list of (list of bit)
                [(101011..1), (100011..), ...]

        Returns
        ------------
        filtered_bit_list: list of (list of bit)
            compressed bit

        """
        self.total_samples = len(bit_list)

        # calculate modes and sums for each bit
        # TODO: using dataframe is slightly slow...
        df = pd.DataFrame(bit_list)
        self.mode_dict = df.mode().T.to_dict()[0]
        sum_dict = df.sum().to_dict()

        self.variety_dict = {k: self.calc_variety(
            v) for k, v in zip(sum_dict.keys(), sum_dict.values())}
        self.filter_dict = {k: self.filtering(v) for k, v in zip(
            self.variety_dict.keys(), self.variety_dict.values())}

        self.extracted_bits = sum(
            [1 for i in self.filter_dict.values() if i == True])
        print("extracted ", self.extracted_bits,
              "bits from", len(bit_list[0]), " bits")

        self.essential_column_list = [k for k, v in zip(
            self.filter_dict.keys(), self.filter_dict.values()) if v == True]

        return self.transform(bit_list)

    def inverse_bit(self, compound_bit):
        template_bit = [v for v in list(self.mode_dict.values())]

        for bit, ind in zip(compound_bit, self.essential_column_list):
            template_bit[ind] = bit
        return template_bit

    def inverse_transform(self, compound_bit_list):
        """
        inverse transforming of the compressed bit. Mode values will be used to recover bits

        Parameters
        -------------
        compound_bit_list: list of (list of bit)
            compressed bit

        Returns
        --------------
        return: list of (list of bit)
            decompressed bit
        """

        return [self.inverse_bit(i) for i in compound_bit_list]


def compare_bits(a, b):
    match_num = sum([1 if i == j else 0 for i, j in zip(a, b)])
    return match_num/len(a)
