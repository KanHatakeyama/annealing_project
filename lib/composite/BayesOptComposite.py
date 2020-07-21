import numpy as np
import GPyOpt


class BayesOptComposite:
    """
    utility class for bayes optimization of chemicals
    this is a wrapper class of GPyOpt
    """
    def __init__(self, smiles_list, screener):

        self.smiles_list = smiles_list
        self.screener = screener

        n_compounds = len(smiles_list)
        self.bounds = [
            {'name': 'comp1', 'type': 'discrete',
                'domain': list(range(0, n_compounds-1))},
            {'name': 'comp2', 'type': 'discrete',
             'domain': list(range(0, n_compounds-1))},
            {'name': 'comp3', 'type': 'discrete',
             'domain': list(range(0, n_compounds-1))},
            {'name': 'comp4', 'type': 'discrete',
             'domain': list(range(0, n_compounds-1))},
            {'name': 'comp5', 'type': 'discrete',
             'domain': list(range(0, n_compounds-1))},
            {'name': 'comp6', 'type': 'discrete',
             'domain': list(range(0, n_compounds-1))},

            {'name': 'ratio1', 'type': 'discrete', 'domain': (0, 0)},
            {'name': 'ratio2', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'ratio3', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'ratio4', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'ratio5', 'type': 'continuous', 'domain': (0, 1)},
            {'name': 'ratio6', 'type': 'continuous', 'domain': (0, 1)},
    
            
        ]
        """
        {'name': 'ratio1', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'ratio2', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'ratio3', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'ratio4', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'ratio5', 'type': 'discrete', 'domain': (0, 1)},
        {'name': 'ratio6', 'type': 'discrete', 'domain': (0, 1)},        
        """

    def lossFunc(self, x):
        """
        calculate loss (-y) from the specific condition

        Parameters
        ---------------
        x: list
            selected from self.bounds

        Returns
        ---------------
        -y: float
            - predicted value
        """
        try_condition = []

        for i in range(6):
            n = int(x[:, i][0])
            ratio = x[:, i+6][0]
            try_condition.append([self.smiles_list[n], ratio])
        y = self.screener.predict(try_condition)

        return -y

    def explore(self, max_iter=40):
        """
        explore conditions by bayes opt

        Parameters
        -----------------
        max_iter: int
            max iterations

        Returns
        -----------------
        y_list: list of float
            history of y
        y_best_hist: list of float
            history of best_y
        """
        self.opt = GPyOpt.methods.BayesianOptimization(
            f=self.lossFunc, domain=self.bounds)
        self.opt.run_optimization(max_iter=max_iter, verbosity=True)
        self.update()

        return self.y_list, self.y_best_hist

    def update(self):
        self.y_list = [i[0] for i in self.opt.Y]
        self.y_best_hist = calc_best_y_list(self.y_list)
        return self.y_list, self.y_best_hist


def calc_best_y_list(y_list):
    best_y_list = []
    for num, y in enumerate(y_list):
        best_y_list.append(np.max(y_list[:num+1]))

    return best_y_list
