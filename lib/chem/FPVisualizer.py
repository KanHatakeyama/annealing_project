from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem import rdMolDescriptors
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter

#please check the rdkit manual drawing chemical fragments
#https://www.rdkit.org/docs/GettingStartedInPython.html#drawing-molecules

class FPVisualizer:
    """
    Utility class to visualize and process chemical fragments 
    """
    def __init__(self, dataset, draw_all=False):
        """
        Parameters
        --------------------
        dataset: tuple of (list of string , list of bit)
            SMILES_list: list of smiles recorded in the database
            fingerprint_list: list of the corresponding FPs recorded in the database
        draw_all: bool
            if true, draw all fragments in a chemical
        """

        self.SMILES_list, self.fingerprint_list = zip(*dataset)
        self.draw_all = draw_all

    def get_mol_id_with_specific_bit_id(self, bit_ID):
        """
        extract chemicals whose fingerprint[bit_ID]==1
        
        Parameters
        --------------------
        bit_ID: int
            id of fignerprint

        Returns
        -------------------
        hit_ID: int
            ID of chemicals whose fingerpint's bit_ID ==1
        
        """

        hit = [True if fp[bit_ID] == 1 else False for fp in self.fingerprint_list]
        temp = list(range(len(hit)))
        hit_ID = [i for i, j in zip(temp, hit) if j == 1]
        return hit_ID

    def auto_draw_fragments(self, ID_list,draw=True):
        """
        draw chemical fragments with specific bit_ID

        Parameters
        ---------------
        ID_list: list of int
            list of bit_ID

        Returns
        -----------------
        self.draw_fragments(tup): image object
            chemical structures

        smiles_list: list of string
            corresponding smiles
        """
        tup, smiles_list = self.calc_draw_tuples(ID_list)
        
        #TODO: kekulization errors with some compounds
        if draw:
            img=self.draw_fragments(tup)
        else:
            img=None
        return img, smiles_list

    def calc_draw_tuples(self, ID_list):
        """
        internal function of auto_draw_fragments
        """
        draw_tuple = []
        smiles_list = []
        for bit_ID in ID_list:
            #get smiles indexes whose bit_ID ==1
            hit_ID = self.get_mol_id_with_specific_bit_id(bit_ID)

            #create mol object whose molecular weight is smallest
            match_SMILES_list = np.array(self.SMILES_list)[hit_ID]
            sm = sort_SMILES_list_by_MW(match_SMILES_list)[0]

            if sm == -1:
                continue

            smiles_list.append(sm)
            mol = Chem.MolFromSmiles(sm)

            bitI_rdkit = {}
            fp_rdkit = Chem.RDKFingerprint(mol, bitInfo=bitI_rdkit)
            draw_tuple.append((mol, bit_ID, bitI_rdkit))

        return draw_tuple, smiles_list

    def draw_fragments(self, draw_tuple):
        image_list = []
        for tup in draw_tuple:
            mol, bit_ID, fp = tup

            if self.draw_all:
                # one molecule can have multiple fragments
                for i in range(len(fp[bit_ID])):
                    img = Draw.DrawRDKitBit(mol, bit_ID, fp, whichExample=i)
                    image_list.append(img)
            else:
                img = Draw.DrawRDKitBit(mol, bit_ID, fp, whichExample=0)
                image_list.append(img)

        imgs = Image.fromarray(np.concatenate(image_list, axis=0))
        return imgs

    def calc_duplicate_array(self, bit_ID_list, threshold=0.5, plot=True):
        """
        this is an original function to drop similar fingerprints

        Parameters
        -----------------
        bit_ID_list: list of int
            list of bit_ID of fignerprints. If different bit_ID have simialr contributions, they will be merged.
        threshold: float
            threshold to drop similar bit_IDs
        plot: bool
            if true, plot similarity heatmap            
        
        """

        ID_types = len(bit_ID_list)
        subset_array = np.ones((ID_types, ID_types))

        # from the database, extract a compound whose bit_ID ==1
        for n1, i in enumerate(bit_ID_list):
            hit_ids1 = self.get_mol_id_with_specific_bit_id(i)
            for n2, j in enumerate(bit_ID_list):
                hit_ids2 = self.get_mol_id_with_specific_bit_id(j)

                # calcualte the difference of FP_i and FP_j
                subset_array[n1][n2] = len(
                    list((set(hit_ids1)-set(hit_ids2))))/len(hit_ids1)

        if plot:
            plt.imshow(subset_array, interpolation='nearest', cmap='jet')

        # delete similar bit_ids
        dup_score = np.mean(subset_array, axis=0)
        modif_bit_ID_list = [i for i, j in zip(
            bit_ID_list, dup_score) if j > threshold]

        return modif_bit_ID_list, subset_array


def calc_MW_from_SMILES_list(SMILES):
    mol = Chem.MolFromSmiles(SMILES)
    return rdMolDescriptors._CalcMolWt(mol)


def sort_SMILES_list_by_MW(SMILES_list):
    """
    sort smiles by molecular weight
    """
    if len(SMILES_list) == 0:
        return [-1]

    mw_list = [calc_MW_from_SMILES_list(i) for i in SMILES_list]
    dataset = (list(zip(mw_list, SMILES_list)))
    dataset.sort(key=lambda x: x[0])
    mw_list, SMILES_list = list(zip(*dataset))
    return SMILES_list
