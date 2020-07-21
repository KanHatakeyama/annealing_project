import numpy as np


class DecimalBitConverter:
    """
    a class convert bit to float (or float to bit). 
    this is so called unary method. simplest, but powerful. (robust against dynamic range)

    """

    def __init__(self, v_min, v_max, total_bit_len):
        """
        Parameters
        ------------------
        v_min: float
            minimum value
        v_max: float
            maximum value
        total_bit_len: int
            how many bits are used to express values

        """
        self.v_max = v_max
        self.v_min = v_min
        self.total_bit_len = total_bit_len
        self.val_list = np.linspace(v_min, v_max, total_bit_len+1)
        #print("valueList: ",self.val_list)

    def float_to_bit(self, v):
        """
        convert float to bit

        Parameters
        -----------------
        v: float
            value to be converted

        Returns
        ---------------
        bit: int array
            array of bits
        """
        # get nearest value in the val_list
        idx = np.abs(np.asarray(self.val_list) - v).argmin()
        bit = "0"*(self.total_bit_len-idx)+"1"*idx
        bit = [int(i) for i in bit]

        # shuffle bit
        # TODO: there are many ways to express a single value... ( [1,0,0] = [0,1,0] = [0,0,1])
        # random.shuffle(bit)
        return bit

    def bit_to_float(self, bit):
        """
        convert bit to float values

        Parameters
        -------------
        bit: int array
            array of bits

        Returns
        ------------
        val: float
            converted float

        """

        bit = [str(i) for i in bit]
        bit = "".join(bit)
        idx = bit.count("1")
        val = self.val_list[idx]

        return val
