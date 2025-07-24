# Cleaning up my thoughts for the IIB Project

We really need to built up a big picture of what we have been doing.

Always try to answer: **What is the Killer result**



## The Objective



## Theoretical Results (I think for the paper version, we need to add in more theories)

### 1. Previous Work: GRF as an unbiased estimator

### 2. Bounding GRF Variance (For the simplified scheme)
- We derived this for the simplified GRF schene
- This is very helpful in practice - we can figure out how much random walk we need to get a converging estimate of the kernel
- and of course we can plug in some assumptions of the graph to evaluate a even looser bound - I expect the spectral radius and the average degree to be very important factors.

### 3. Showing Graph GP is capable of learning the groundtruth model hyperparameters

- I am not quite sure for this - is this as important?



## Why do we use GRF-GP

GRF-GP has the following key advantages:
- efficiency: O(mN) computational complexity for Graph Gaussian Process Inference
- Rich parameterization to learn a board family of kernels from a graph (via hyperparameter optimization) - which exhibit superior model performance

## Replicating Experiment Results from Previous Papers

### 1. Convergence of GRF
- Linear log-log curve against number of random walkers
- Invariant to Graph Size 

### 2. Graph GP Inference 
- Graph GP Inference on Synthetic data
- Graph GP Regression on the Traffic Example

## Experiment of using GRF


### 1. Enlighting Example: GP Inference with GRF

- We show that how GRF works on graph GP inference
- Synthetic Data

### 2. GP Regression Taks with GRF (Traffic Example)
- Show that Arbitrary GRF GP exceed traditional GP kernel inference
- Show the scaling of performance with random walkers

## Downstream Applications

### 1. Bayes Opt
- What is the killer result??
- It is still doing Graph GP , but this time taking samples are expensive and we want to optimize our observations
- We illustrate how we can use GRF in this context
- Ideally we hope that using GRF we have faster inference
- And we also hope that using arbitrary kernel learning - we can have - on average, better performances than using exact diffusion kernel (I think this is possible)
- And compare with the random search - which should have worse performance.


## Endeavors for improving algorithm efficiency

### 1. Sparse Implementaion:

- Theoretical Result - the computation complexity for a graph with average degree $d$
- We could maybe do a simple sparse matrix experiment??  Let's think about this for the GRF.

### 2. Kernel Lazy Update:

- We actually do have a lot of exciting theoretical result with this, like importance sampling
- but at the end of the day, it is about how to estimate $W_{old}^n$ when you have a similar matrix $W_{old}^n$
- The fact that we are using PoFM / GRF enables this decomposition of the graph kernel - hence making this interesting

## New Contribution from my work
