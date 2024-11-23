import networkx as nx
import numpy as np

# Function to plot the network graph in Matplotlib
def plot_network_graph(adjacency_matrix, ax):
    # Create a NetworkX graph from the adjacency matrix
    G = nx.from_numpy_array(adjacency_matrix)
    pos = nx.spring_layout(G)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=300, node_color='red')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='blue')
    
    ax.set_title("Network Graph Representation of Adjacency Matrix")
    ax.axis("off")
    
def plot_gp_fit(X, Y, X_new, mean, stddev, beta_sample, ax):
    ax.plot(X.numpy(), Y.numpy(), 'ro', label='Noisy Samples')
    ax.plot(X_new.numpy(), mean.numpy(), 'b-', label='Fitted Mean')
    ax.fill_between(X_new.numpy().flatten(),
                    (mean - 2 * stddev).numpy().flatten(),
                    (mean + 2 * stddev).numpy().flatten(),
                    color='lightblue', alpha=0.5, label='95% Confidence Interval')
    ax.set_title(f'Gaussian Process Fit (beta={beta_sample})')
    ax.set_xlabel('Node Number')
    ax.set_ylabel('Sampled Value')
    ax.grid()
    ax.legend()
    
def compute_fro(first_matrix, second_matrix, relative=True):
    "Calculates the Frobenius norm"
    diff_norm = np.linalg.norm(first_matrix - second_matrix)
    if not relative:
        return diff_norm * diff_norm
    else:
        return diff_norm / np.linalg.norm(first_matrix)