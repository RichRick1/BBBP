import numpy as np
from tools import mutate_one_hot
from functools import partial
from multiprocessing import Pool


class GeneticModel:
    """
    Class to run genetic algorithm for SMILES modification
    """
    def __init__(self, crossover_rate, mutation_rate,
                 n_mutations = None, 
                 n_species=500,
                 func=None,
                 n_parents=10,
                 n_genes = None):
        """
        Parameters
        ----------
        crossover_rate : float
            Probability of crossover
        mutation_rate : float
            Probability of mutation
        n_mutations : int
            Number of mutations
        n_species : int
            Number of species in the population
        func : function
            Function to evaluate the fitness of a species
        n_parents : int
            Number of parents to select
        n_genes : int
            Number of genes in the one-hot vector
        """
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
        """
        Function to mutate the single molecule
        
        Parameters
        ----------
        population : np.array
            Population of one-hot vectors
        
        Returns
        -------
        mutated_population : np.array
            Mutated population
        """
        mutation_genes = np.random.rand(*population.shape)
        mutation_genes = mutation_genes<self.mutation_rate
        mutated_population = population.copy()
        mutated_population[np.logical_and(mutation_genes, population==0)] = 1
        mutated_population[np.logical_and(mutation_genes, population==1)] = 0
        
        self.population = mutated_population
        return mutated_population
    
    def mutate_2d(self, population, n_genes):
        """
        Function to mutate the population
        
        Parameters
        ----------
        population : np.array
            Population of one-hot vectors
        n_genes : int
            Number of genes in the one-hot vector
            
        Returns
        -------
        mutated_population : np.array
            Mutated population
        """
        mutated_population = []
        
        for elem in population:
            mutated_elem = mutate_one_hot(elem.reshape(n_genes, -1), self.mutation_rate)
            mutated_population.append(mutated_elem.flatten())
            
        self.population = np.array(mutated_population)
        return np.array(mutated_population)
    
    def select(self, scores):
        """
        Function to select the best species
        
        Parameters
        ----------
        scores : list
            List of scores of the species
            
        Returns
        -------
        best_args : np.array
            Array of indices of the best species
        """
        best_args = np.argsort(scores)[-self.n_parents:]
        return self.population[best_args]
    
    def crossover(self, selected):
        """
        Function to perform crossover

        Parameters
        ----------
        selected : np.array
            Array of selected species

        Returns
        -------
        offspings : list
            List of offspings
        """
        offspings = []
        n_genes = selected.shape[1]
        for i in range(len(selected)//2):
            parent_1 = selected[i]
            parent_2 = selected[self.n_parents-1-i]
            offsping = np.hstack([parent_1[:n_genes//2], parent_2[n_genes//2:]])
            offspings.append(offsping)
        return offspings
    
    def populate(self, candidates, scores):
        """
        Function to populate the next generation

        Parameters
        ----------
        candidates : np.array
            Array of candidates
        scores : list
            List of scores of the species

        Returns
        -------
        population : np.array
        """
        probs = np.array(scores)/np.sum(scores)
        probs[np.isnan(probs)] = 1/len(probs)
        
        population_inds = np.random.choice(candidates.shape[0], self.n_species, p=probs)
        population = candidates[population_inds]
        self.population = population
        return population
    
    def run_selection(self, species, n_runs=2000):
        """
        Function to run the selection process
        
        Parameters
        ----------
        species : np.array
            Array of species
        n_runs : int
            Number of runs

        Returns
        -------
        new_candidates : np.array
            Array of new candidates
        new_score : list
            List of scores of the new candidates
        """
        self.history = []
        func = self.func
        # populate the initial population
        if self.initial is None:
            self.initial = species
            scores = [func(species)]
            
            self.populate(species, scores) 
            scores = [scores[0] for i in range(self.n_species)]
        else:
            scores = [func(elem) for elem in self.population]
        
        # run selection
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