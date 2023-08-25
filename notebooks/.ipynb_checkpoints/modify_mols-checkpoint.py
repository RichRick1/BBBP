
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

def mutate_one_hot(one_hot, mutation_rate):
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


# +
def is_valid(smile):
    m = Chem.MolFromSmiles(smile, sanitize=False)
    try:
        problems = Chem.DetectChemistryProblems(m)
    except: return False
    return False if len(problems)>0 else True
    

def get_permeability_pred(flattened_one_hot, n_genes=235, return_smile=False):
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


# -

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
    try:
        descriptors = pd.DataFrame(from_smiles(smiles, fingerprints=False, descriptors=True))
    except:
        return [0]
    descriptors.replace([-np.inf, np.inf], np.nan, inplace=True)
    descriptors.dropna(axis=0, inplace=True)
    
    descriptors.replace([''], 0, inplace=True)
    descriptors = descriptors[features]
    try:
        descriptors_scaled = scaler.transform(descriptors)
    except:
        return [0]
    
    return clf.predict_proba(descriptors_scaled)[:, 1]



class GeneticModel:
    def __init__(self, crossover_rate, mutation_rate,
                 n_mutations = None, 
                 n_species=500,
                 func=None,
                 n_parents=10,
                 n_genes = None):
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.n_mutations = n_mutations
        self.n_species = n_species
        self.population = None
        self.initial = None
        self.func = func
        self.n_parents=n_parents
        self.n_genes = n_genes
        self.history = None
        
    def mutate(self, population):
        mutation_genes = np.random.rand(*population.shape)
        mutation_genes = mutation_genes<self.mutation_rate
        mutated_population = population.copy()
        mutated_population[np.logical_and(mutation_genes, population==0)] = 1
        mutated_population[np.logical_and(mutation_genes, population==1)] = 0
        
        self.population = mutated_population
        return mutated_population
    
    def mutate_2d(self, population, n_genes):
        mutated_population = []
        
        for elem in population:
            mutated_elem = mutate_one_hot(elem.reshape(n_genes, -1), self.mutation_rate)
            mutated_population.append(mutated_elem.flatten())
            
        self.population = np.array(mutated_population)
        return np.array(mutated_population)
    
    def select(self, scores):
        best_args = np.argsort(scores)[-self.n_parents:]
        return self.population[best_args]
    
    def crossover(self, selected):
        offspings = []
        n_genes = selected.shape[1]
        for i in range(len(selected)//2):
            parent_1 = selected[i]
            parent_2 = selected[self.n_parents-1-i]
            offsping = np.hstack([parent_1[:n_genes//2], parent_2[n_genes//2:]])
            offspings.append(offsping)
        return offspings
    
    def populate(self, candidates, scores):
        probs = np.array(scores)/np.sum(scores)
        probs[np.isnan(probs)] = 1/len(probs)
        
        population_inds = np.random.choice(candidates.shape[0], self.n_species, p=probs)
        population = candidates[population_inds]
        self.population = population
        return population
    
    def run_selection(self, species, n_runs=2000):
        self.history = []
        func = self.func
        
        if self.initial is None:
            self.initial = species
            scores = [func(species)]
            
            self.populate(species, scores) 
            scores = [scores[0] for i in range(self.n_species)]
        else:
            scores = [func(elem) for elem in self.population]
        
        for i in range(n_runs):
            print(f"=======> {i+1}")
            print(self.population, i)
            scores = [func(elem) for elem in self.population]
            print(scores)
            best_species = self.select(scores)
            print(best_species)
            offsprings = self.crossover(best_species)
            
            pool = Pool()
            offspings_scores = [func(offsping) for offsping in offsprings]
            

            new_candidates = np.vstack([best_species[-self.n_parents//2: ],
                                        offsprings])
            new_score = sorted(scores)[-self.n_parents//2:]+offspings_scores
            
            self.history.append((new_candidates, new_score))
            
            new_population = self.populate(new_candidates, new_score)
            if i != n_runs-1:
                new_population = self.mutate_2d(new_population, self.n_genes)
            
            if not i%n_runs//5:
                print(best_species[-5:], sorted(scores)[-5:], scores[0])
            
        return new_candidates, new_score



clf = get_clf('xgb', 'borderline_SMOTE')
scaler = load('../pre_trained/b3clf_scaler.joblib')

with open('../feature_list.txt', 'r') as f:
    features = f.read().splitlines()



test_smiles = ['O=C(O)c1cc(N=Nc2ccc(S(=O)(=O)Nc3ccccn3)cc2)ccc1O',
               'CCCC',
               'COC1(NC(=O)C(C(=O)O)c2ccc(O)cc2)C(=O)N2C(C(=O)O)=C(CSc3nnnn3C)COC21']

bbb_fpath = "../B3DB/B3DB_classification_extended.tsv.gz"
df = pd.read_csv(bbb_fpath, sep="\t", compression="gzip")


# +
np.random.seed(42)

random_inds = np.random.choice(df.logBB[(df.logBB.notna()) & (df['BBB+/BBB-'] == 'BBB-')].index.values,
                               size=100)
test_smiles = df.SMILES[(df.logBB.notna()) & (df['BBB+/BBB-'] == 'BBB-')].iloc[random_inds].tolist()
# -


smiles = df.SMILES.to_list()
selfies = [sf.encoder(smile) for smile in smiles]
alphabet = sf.get_alphabet_from_selfies(selfies)
alphabet.add("[nop]")  # [nop] is a special padding symbol
alphabet = list(sorted(alphabet)) 

pad_to_len = max(sf.len_selfies(s) for s in selfies)  # 5
symbol_to_idx = {s: i for i, s in enumerate(alphabet)}

test_selfies = [sf.encoder(smile) for smile in test_smiles]

for k, test_selfie in zip(random_inds[13:], test_selfies[13:]):
    label, one_hot = sf.selfies_to_encoding(selfies=test_selfie,
                                            vocab_stoi=symbol_to_idx,
                                            pad_to_len=pad_to_len,
                                            enc_type="both")
    
    test_one_hot = np.array(one_hot)

    real_shape = test_one_hot.shape
    population = np.array([test_one_hot.flatten()])
    model = GeneticModel(0.5, 0.005, n_species=20, n_genes=235,func=get_permeability_pred, n_parents=4)

    res = model.run_selection(population, n_runs=15)
    with open(f'smiles_new_{k}.pickle', 'wb') as f:
        pickle.dump(model.history, f)

model.population

with open(f'smiles_new_{k}.pickle', 'wb') as f:
    pickle.dump(model.history, f)
