import networkx as nx
import matplotlib.pyplot as plt


def read_gph(file_path):
    """
    Reads the .gph file and returns the structure as a dictionary.
    
    Parameters:
    - file_path: The path to the .gph file.
    
    Returns:
    - structure: A dictionary where keys are child nodes and values are lists of parent nodes.
    """
    structure = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Remove extra whitespace and split on '->'
            parent, child = line.strip().replace(" ", "").split('->')
            
            # Add the child and parent to the structure
            if child not in structure:
                structure[child] = []
            structure[child].append(parent)
            
            # Ensure the parent is also a node in the structure, even if it has no parents
            if parent not in structure:
                structure[parent] = []
    
    return structure

def visualize_bayesian_network(structure):
    """
    Visualizes the Bayesian network structure using networkx and matplotlib.
    
    Parameters:
    - structure: A dictionary where keys are child nodes and values are lists of parent nodes.
    """
    # Create a directed graph
    G = nx.DiGraph()

    # Add edges to the graph based on the structure
    for child, parents in structure.items():
        for parent in parents:
            G.add_edge(parent, child)

    # Draw the graph
    plt.figure(figsize=(8, 6))  # Set the figure size
    pos = nx.circular_layout(G)  # Position nodes in a circle
    nx.draw(G, pos, with_labels=True, node_size=200, node_color="lightblue", font_size=8, font_weight="bold", arrows=True, arrowstyle="->", arrowsize=15)
    
    
    # Show the plot
    plt.title("Bayesian Network Structure")
    plt.show()


if __name__ == "__main__":
    gph_file_path = 'large.gph'  # Path to your .gph file

    # Read the structure from the .gph file
    structure = read_gph(gph_file_path)

    # Visualize the Bayesian network structure
    visualize_bayesian_network(structure)