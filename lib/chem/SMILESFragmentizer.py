
from rdkit.Chem import BRICS
from tqdm import tqdm
import itertools
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

# fragmentate chemicals
# see rdkit manual
# https://www.rdkit.org/docs/source/rdkit.Chem.BRICS.html


def fragmentate_chemicals(SMILES_list, return_only_fragments=True):
    """
    fragmentate chemicals by BRICs algorithm

    Parameters
    ---------------------
    SMILES_list: list of string
       list of smiles
    return_only_fragments: bool
       if true, return only fragment parts

    Returns
    ----------------------
    fragmentated_smiles: list of string
        list of fragmentated chemicals
    """

    mols = [Chem.MolFromSmiles(SMILES) for SMILES in SMILES_list]
    fragmentated_smiles = [BRICS.BRICSDecompose(mol) for mol in tqdm(mols)]

    # nested list to normal list
    fragmentated_smiles = (
        list(itertools.chain.from_iterable(fragmentated_smiles)))
    fragmentated_smiles = list(set(fragmentated_smiles))

    if return_only_fragments:
        fragmentated_smiles = [
            i for i in fragmentated_smiles if i.find("*") > 0]

    return fragmentated_smiles


def generate_chemicals_from_fragments(smiles_list, n=10):
    """
    reconstruct chemicals from fragments

    Paramters
    -----------------
    smiles_list: list of string
        list of smiles of fragments
    n: int
        number of chemicals to be generated

    Returns
    ---------------
    smiles_list: list of string
        list of newly generated smiles
    """

    # convert smiles to mol objects
    all_components = [Chem.MolFromSmiles(f) for f in smiles_list]
    builder = BRICS.BRICSBuild(all_components)

    generated_mol_list = []
    for i in (range(n)):
        m = next(builder)
        m.UpdatePropertyCache(strict=True)
        generated_mol_list.append(m)

    smiles_list = [Chem.MolToSmiles(m) for m in generated_mol_list]

    return smiles_list
