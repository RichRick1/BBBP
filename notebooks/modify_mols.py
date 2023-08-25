
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
import selfies as sf
import os
import pickle
from joblib import load
from padelpy import from_smiles
from rdkit import Chem
from multiprocessing.dummy import Pool
from GeneticClass import GeneticModel

def is_valid(smile):
    """
    Function to check if a SMILE is valid

    Parameters
    ----------
    smile : str
        SMILE string

    Returns
    -------
    bool
    """
    m = Chem.MolFromSmiles(smile, sanitize=False)
    try:
        problems = Chem.DetectChemistryProblems(m)
    except: return False
    return False if len(problems)>0 else True
    

def get_permeability_pred(flattened_one_hot, n_genes=235, return_smile=False):
    """
    Function to get permeability prediction from
      a one-hot vector representation of a molecule

    Parameters
    ----------
    flattened_one_hot : np.array
        One-hot vector of a molecule
    n_genes : int
        Number of genes in the one-hot vector
    return_smile : bool
        Whether to return the decoded SMILE string

    Returns
    -------
    predicted : float
    """
    one_hot = flattened_one_hot.reshape(n_genes, -1)
    alphabet_arr = np.array(alphabet)
    label = np.where(np.array(one_hot) == 1)[1]
    reconstracted_selfie = alphabet_arr[label]
    merged_selfie = ''.join(alphabet_arr[label][np.where(alphabet_arr[label] != '[nop]')].tolist())

    decoded_smile = sf.decoder(merged_selfie)
    
    if return_smile:
        return predict_permeability([decoded_smile]), decoded_smile
    
    predicted = predict_permeability([decoded_smile])
    valid = is_valid(decoded_smile)
    
    if valid:
        return predicted[0]
    else:
        return 0

def get_clf(clf_str, sampling_str):
    """Get b3clf fitted classifier
    """
    clf_list = ["dtree", "knn", "logreg", "xgb"]
    sampling_list = ["borderline_SMOTE", "classic_ADASYN",
                     "classic_RandUndersampling", "classic_SMOTE", "kmeans_SMOTE", "common"]

    # This could be moved to an initial check method for input parameters
    if clf_str not in clf_list:
        raise ValueError(
            "Input classifier is not supported; got {}".format(clf_str))
    elif sampling_str not in sampling_list:
        raise ValueError(
            "Input sampling method is not supported; got {}".format(sampling_str))

    dirname = '..'
    # Move data to new storage place for packaging
    clf_path = os.path.join(
        dirname, "pre_trained", "b3clf_{}_{}.joblib".format(clf_str, sampling_str))

    clf = load(clf_path)

    return clf

def predict_permeability(smiles):
    """Predict permeability from SMILES
    
    Parameters
    ----------
    smiles : list
        List of SMILES strings
        
    Returns
    -------
    list
        List of permeability predictions   
    """
    try:
        descriptors = pd.DataFrame(from_smiles(smiles, fingerprints=False, descriptors=True))
    except:
        return [0]

    # replace inf with nan and drop nan
    descriptors.replace([-np.inf, np.inf], np.nan, inplace=True)
    descriptors.dropna(axis=0, inplace=True)
    
    # remove columns with no values
    descriptors.replace([''], 0, inplace=True)
    descriptors = descriptors[features]
    try:
        descriptors_scaled = scaler.transform(descriptors)
    except:
        return [0]
    
    return clf.predict_proba(descriptors_scaled)[:, 1]







clf = get_clf('xgb', 'borderline_SMOTE')
scaler = load('../pre_trained/b3clf_scaler.joblib')

with open('../feature_list.txt', 'r') as f:
    features = f.read().splitlines()



test_smiles = ['O=C(O)c1cc(N=Nc2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O',
               'CCCC',
               'COC1(NC(=O)C(C(=O)O)c2ccc(O)cc2)C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)COC21']

bbb_fpath = "../B3DB/B3DB_classification_extended.tsv.gz"
df = pd.read_csv(bbb_fpath, sep="\t", compression="gzip")

# fixing the random seed
np.random.seed(42)
# random sampling
random_inds = np.random.choice(df.logBB[(df.logBB.notna()) & (df['BBB+/BBB-'] == 'BBB-')].index.values,
                               size=100)
test_smiles = df.SMILES[(df.logBB.notna()) & (df['BBB+/BBB-'] == 'BBB-')].iloc[random_inds].tolist()

# get set of all unique characters in the dataset
smiles = df.SMILES.to_list()
selfies = [sf.encoder(smile) for smile in smiles]
alphabet = sf.get_alphabet_from_selfies(selfies)
alphabet.add("[nop]")  # [nop] is a special padding symbol
alphabet = list(sorted(alphabet)) 


pad_to_len = max(sf.len_selfies(s) for s in selfies)  # 5
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
# encode the selfies
test_selfies = [sf.encoder(smile) for smile in test_smiles]

# get the one-hot vector representation of the selfies 
for k, test_selfie in zip(random_inds[13:], test_selfies[13:]):
    label, one_hot = sf.selfies_to_encoding(selfies=test_selfie,
                                            vocab_stoi=symbol_to_idx,
                                            pad_to_len=pad_to_len,
                                            enc_type="both")
    
    test_one_hot = np.array(one_hot)

    real_shape = test_one_hot.shape
    population = np.array([test_one_hot.flatten()])
    # initialize the model
    model = GeneticModel(0.5, 0.005, n_species=20, n_genes=235,func=get_permeability_pred, n_parents=4)

    # run the selection process
    res = model.run_selection(population, n_runs=15)
    # save the history
    with open(f'smiles_new_{k}.pickle', 'wb') as f:
        pickle.dump(model.history, f)

# save the history
with open(f'smiles_new_{k}.pickle', 'wb') as f:
    pickle.dump(model.history, f)
