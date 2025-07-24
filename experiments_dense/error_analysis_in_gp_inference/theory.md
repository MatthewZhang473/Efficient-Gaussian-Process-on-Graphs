# Error Analysis in Graph GP Inference

LLN - Law of Large Numbers

## The Question

Our toy model to validate our GP inference is to fit a noisy dataset sampled from a known GP. 

Previous we tried to compared the ML estimate of the fitted hyper parameters to the ground truth: they were reasonably close to the ground truth value, but:
1. we are not very clear about the convergence of this estimator (as we reduce noise level / increase number of data points)
2. we found it not really stable experimentally

It thus make sense to verify this carefully.


## Task Decomposition

There are two sources of error in our GRF-based Graph GP Inference:

1. In **GP Inference**:  are we learning the correct hyper-parameter?

    Our ML estimate for the hyper-parameter is $ \hat{\theta}_{ML} $, and we need to theoretically quantify the expeceted value and variance of this estimator. Experiments are also taken to prove this.

    Note that GRF is not yet introduced for this task, so we are working with exact kernel expression.

2. In **GRF Approximation**: is the kernel we approximated close to the exact evaluation?

    As implemented in *experiments/convergence_tests/test_convergence_fast_grf.ipynb*, we have proved that the frobenious norom of the GRF-approximated kernel decreases as we increase the number of walks per node. For small value of beta, we can reach a frobeninous norm of 10^-4 after 1000 walks.

    And here we want to integrate GRF into the GP inference to see how there error in approximation affects the GP inference to learn the hyperparameters.



## Exact GP Inference

### Convergence of $ \hat{\theta}_{ML} $: The Intuition

In the data generation part of our toy example, we are essentially sampling from a GP model, parameterised by the hyper-parameter $\theta_0$.

The idea is that, as we increase the data sampled, the posterior probability of the parameter $P(\hat{\theta} = \theta_)| {\{x_i\}}_{i=1}^{N})$ converges to 1 as N increases to infinity.


### An interesting observation

As we increase the degree of freedom in $\theta$, we can essentially estimate the entire covariance matrix freely.

When we observe the sampled data $x$, the ML estimator of the covariance matrix is just $x^Tx$, and we know that it is just an unbiased estimator of the true covariance matrix. ($E[x^Tx] = \Sigma $)


## A Better Way to Quantify the Estimator

Let's not look at the ML estimator (point estimator)!

Let's look at the posterior distribution!!


And we can use $P(\hat{\theta} \approx \theta_0)$ to quantify the performance. (It is visually better and more solid).


## KL Divergence


We have also mathematically derived the KL divergence of multivariate Gaussian, and it is a better metric than FRO.
## Trade Off

The higher the length scale parameter is, the closer the GP convergence is.

However, for close GRF approximation, a small length scale parameter is desired.




