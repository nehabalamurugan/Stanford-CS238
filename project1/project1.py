import pandas as pd
import numpy as np
from itertools import product
from math import lgamma
import sys
import random

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))

def bic_score(data, structure, alpha=1.0):
    """
    data: pandas dataframe
    structure: dictionary of parent: child relationships 
    alpha: parameter for BIC score (fixed at 1.0 for uniform dirichlet prior)
    """
    nodes = data.columns
    score = 0.0

    for node in nodes: 
        parents = structure.get(node, [])
        
        r_i = data[node].nunique() #number of possible values for node i
        q_i = 1 #no parent case

        #find all possible parent configurations
        if parents:
            parent_vals = [data[parent].unique() for parent in parents]
            q_i = np.prod([len(states) for states in parent_vals])
            parent_configs = list(product(*parent_vals))
            #print(f"node: {node}, parents: {parents}, parent_configs: {parent_configs}")
        else:
            parent_configs = [()]

        alpha_ij0 = 1

        for parent_config in parent_configs:
            if parents: 
                condition = np.ones(len(data), dtype=bool)
                for parent, value in zip(parents, parent_config):
                    condition &= data[parent] == value
                subset = data[condition]
                print(f"node: {node}, parents: {parents}, parent_config: {parent_config}")
            else: 
                subset = data

            m_ij0 = len(subset)
            alpha_ijk = 1

            m_ijk_counts = subset[node].value_counts().reindex(range(r_i), fill_value=0) #all observed counts of node i given parent configuration

            term_0 = lgamma(alpha_ij0) - lgamma(alpha_ij0 + m_ij0)                 #first term in BIC score
            term_k = np.sum([lgamma(alpha_ijk + m_ijk) - lgamma(alpha_ijk)
                            for m_ijk in m_ijk_counts]) 

            score += term_0 + term_k
            print(f"term_0: {term_0}, term_k: {term_k}")

    return score

def has_cycle(graph):
    from collections import defaultdict
    
    visited = set()
    rec_stack = set()
    
    def visit(node):
        if node in rec_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, []):
            if visit(neighbor):
                return True
        rec_stack.remove(node)
        return False
    
    for node in graph:
        if visit(node):
            return True
    return False

def generate_random_structure(nodes, edge_probability=0.2):
    random_structure = {node: [] for node in nodes}
    
    for child in nodes:
        for parent in nodes:
            if parent != child:
                # Add an edge with probability 'edge_probability'
                if random.random() < edge_probability:
                    random_structure[child].append(parent)
    
    # Ensure the structure is acyclic by checking for cycles and removing edges if necessary
    for child in random_structure:
        if has_cycle(random_structure):
            random_structure[child] = []  # Reset to remove cycles if detected
    
    return random_structure


def hill_climbing_bdeu(data, initial_structure=None, alpha=1.0, max_iter=1000, edge_probability=0.1):
    nodes = data.columns.tolist()
    # Generate a random initial structure if none is provided
    if initial_structure is None:
        current_structure = generate_random_structure(nodes, edge_probability=edge_probability)
        print("Initial Random Structure:", current_structure)
    else:
        current_structure = initial_structure.copy()
    
    best_score = bic_score(data, current_structure, alpha)
    improving = True
    iterations = 0
    
    while improving and iterations < max_iter:
        improving = False
        iterations += 1
        best_candidate = None
        best_candidate_score = best_score
        
        # Generate candidate structures by adding, removing, or reversing an edge
        for parent in nodes:
            for child in nodes:
                if parent == child:
                    continue
                candidate_structure = current_structure.copy()
                candidate_parents = candidate_structure[child][:]
                
                # Adding
                if len(candidate_parents) < 2 and parent not in candidate_parents:
                    candidate_parents.append(parent)
                    candidate_structure[child] = candidate_parents
                    # Check for cycles
                    if has_cycle(candidate_structure):
                        continue
                    score = bic_score(data, candidate_structure, alpha)
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate_structure.copy()
                    
                    candidate_parents.remove(parent)
                
                # Removing
                if parent in candidate_parents:
                    candidate_parents.remove(parent)
                    candidate_structure[child] = candidate_parents
                    score = bic_score(data, candidate_structure, alpha)
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate_structure.copy()
                    
                    candidate_parents.append(parent)
                
                # Reversing
                if parent in candidate_parents:
                    candidate_parents.remove(parent)
                    candidate_structure[child] = candidate_parents
                    candidate_structure[parent].append(child)
                    # Check for cycles
                    if has_cycle(candidate_structure):
                        candidate_structure[parent].remove(child)
                        continue
                    score = bic_score(data, candidate_structure, alpha)
                    if score > best_candidate_score:
                        best_candidate_score = score
                        best_candidate = candidate_structure.copy()
                    
                    candidate_structure[parent].remove(child)
                    candidate_parents.append(parent)
        
        if best_candidate is not None and best_candidate_score > best_score:
            current_structure = best_candidate
            best_score = best_candidate_score
            improving = True
    
    return current_structure, best_score


def write_structure_to_gph(structure, file_path):
    with open(file_path, 'w') as f:
        for child, parents in structure.items():
            for parent in parents:
                f.write(f"{parent},{child}\n")
    

def compute(infile, outfile):
    # Load the data from a CSV file
    data = pd.read_csv(infile)  # Replace 'sample_data.csv' with the path to your CSV file

    # Hill clibing algorithm
    learned_structure, learned_score = hill_climbing_bdeu(data, alpha=1.0)

    # Output the learned structure and its score
    print("Learned Structure:", learned_structure)
    print("Learned BDeu Score:", learned_score)

    # Write the learned structure to a .gph file
    write_structure_to_gph(learned_structure, outfile)



def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)



if __name__ == '__main__':
    main()
