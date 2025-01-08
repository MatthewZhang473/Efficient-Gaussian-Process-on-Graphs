# Efficient Gaussian Process on Graphs

This repository hosts an ongoing research project, developed as part of my fourth-year research for the Cambridge Engineering degree.

## Project Overview

The project aims to harness the General Graph Random Features (g-GRF) algorithm to enable efficient computation of covariance matrices on graphs. By doing so, it makes Gaussian Process (GP) inference computationally efficient and scalable for graph-structured data.

## Repository Structure

### **`experiments/`**
Contains Jupyter notebooks that demonstrate the use cases, experiments, and performance evaluations of the implemented methods.

- **`convergence_tests/`**: 
  - Demonstrates the convergence of different methods used to approximate covariance matrices compared to the ground truth covariance matrices.

  - We also showed that the error in GRF approximation does not increase as the graph size increases.

- **`gp_inference_tests/`**: 
  - Showcases Gaussian Process (GP) inference on synthetic data using various approximated covariance matrices.

- **`kernel_lazy_update/`**: 
  - Illustrates a lazy update algorithm designed for efficiently updating the graph covariance matrix when introducing a new node.

- **`error_analysis_in_gp_inference`**:
  - We illustrated the convergence of GP inference for the toy example, where the maximum likelihood inference of the hyperparameters converges to the ground truth hyperparameter (which we sued to sample the data) as we increase the number of node use in the graph.

- **`grf_performance_bounds`**:
  - Further experiments to show how the GRF method is getting converging and consistent result compared to exact kernel in the GP inference context.

### **`efficient_graph_gp/`**
Core module implementing the Graph Gaussian Process (GP) inference system, including the g-GRF algorithm.

### **`presentations/`**
Slides and other materials that explain:
- The project motivation.
- A detailed overview of the Gaussian Random Field (GRF) algorithm.

## Research Objectives

1. Develop scalable Gaussian Process inference methods tailored for graph data.
2. Implement and optimize the g-GRF algorithm to efficiently compute covariance matrices for graphs.
3. Demonstrate practical applications of GP inference on graph-structured datasets.

## Authors and Supervision

- **Author**: Matthew Zhang ([mz473@cam.ac.uk](mailto:mz473@cam.ac.uk))  
  - Fourth-year Engineering student at the University of Cambridge
  - Interested in bayesian machine learning & graph theory
- **Supervisors**:  
  - **Professor Rich Turner**
  - **Isaac Reid**

## Contributions

Contributions to the repository are welcome! Feel free to submit issues or pull requests to help improve the codebase, documentation, or explore additional applications of the g-GRF algorithm.  

For any questions or feedback, please contact the author at [mz473@cam.ac.uk](mailto:mz473@cam.ac.uk).

## Getting Started

To explore the project, clone the repository and navigate to the `experiments/` directory for examples and applications:

```bash
git clone https://github.com/MatthewZhang473/Efficient-Gaussian-Process-on-Graphs.git
cd Efficient-Gaussian-Process-on-Graphs
