# Graph Bayes Optimization

In this folder we re-implement Bayes Optimization (BO) on graphs.



## Commands:

    python graph_bo/scripts/run_graph_bo.py --config graph_bo/configs/wind_magnitude.yaml 2>&1 | tee graph_bo/logs/wind_magnitude_log_$(date +%Y%m%d_%H%M%S).log

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




