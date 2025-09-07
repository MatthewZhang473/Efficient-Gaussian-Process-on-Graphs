# Graph Bayes Optimization

In this folder we re-implement Bayes Optimization (BO) on graphs.

## High Level Design

### Data 
(For now, use the social network datasets): 
- Adjacency matrix A (sparse CSR format)
- Input Nodes: X
- Labels: y

### BO algorithms
- Greedy Search
- GRF-GPs
- Random Search

### Acquizition function:
- Thompson's Sampling

### Experiment Hyperparameters:
e.g.
- number of BO iterations
- number of repeats

### GRFs Hyperparameters:
- max walk length
- walks per node
- p_halt
- learning rate
- other GPytorch params

### Other Details:
- I/O (logs, step matrices)




