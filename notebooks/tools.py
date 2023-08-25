import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn import preprocessing
import selfies as sf
import os
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

def predict_permeability(smiles, clf, scaler, features):
    """Predict permeability from SMILES
    
    Parameters
    ----------
    smiles : list
        List of SMILES strings
    clf : sklearn classifier
        Fitted classifier
    scaler : sklearn scaler
        Fitted scaler
    features : list
        List of features used for training the classifier

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

def mutate_one_hot(one_hot, mutation_rate):
    """
    Function to mutate a one-hot vector
    
    Parameters
    ----------
    one_hot : np.array
        One-hot vector
    mutation_rate : float
        Probability of mutation
        
    Returns
    -------
    mutated_one_hot : np.array
        Mutated one-hot vector
    """
    n_features = one_hot.shape[1]
    candidates = np.random.randint(0, n_features,
                                   size=one_hot.shape[0])
    
    mutated_one_hot = one_hot.copy()
    mutation_mask = np.random.rand(*candidates.shape)<mutation_rate
    
    mutated_inds = candidates[mutation_mask]
    for row_ind, gene_ind in zip(np.where(mutation_mask)[0], 
                                 mutated_inds):
        mutated_one_hot[row_ind] = np.zeros(one_hot.shape[1])
        mutated_one_hot[row_ind, gene_ind] = 1
    return mutated_one_hot



