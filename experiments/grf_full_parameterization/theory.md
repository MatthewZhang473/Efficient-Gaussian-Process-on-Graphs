# Full Parameterization of GRF


So far we have been pretty successful in using GRF in GP for the diffusion kernel, and it has a simple parameterization - the length scale parameter only!

To fully harvest the power of GRF in approximating arbitrary kernel, we should now introduce full parameterisation - that learns the modulator vector directly.

A good starting point, of course, is from the product-of-feature-matrices kernel. We should find some method to learn the modulator vector for that and after it works, the grf should really kick in naturally.

## Graph GP Inference with General PofM Kernel

### 1.Data Generation: 

random samples from a known diffusion kernel with known hyperparam $\beta_0$. 


### 2.GP Inference, for that we can:

a. Analytically calculates the ground truth PofM modulator with the ground truth beta $\theta_0 = f(\beta_0)$

b. Fit a diffusion kernel to find the best fit beta for the data $\beta_1$, then calculate the best fit PofM modulator $\theta_1 = f(\beta_1)$

b*. Fit a beta-paramterized PofM kernel (with one degree of freedom) for best fit beta $\beta_1^{\prime}$, then calculate the best fit PofM modulator $\theta_1^{\prime} = f(\beta_1)^{\prime}$ (This should be almost identical to case b).


c. Fit a PofM kernel to the data generated to find the experimental best-fit PofM modulator $\hat{\theta}$.

d*. Note it is very interesting to somehow see the posterior distribution of $\hat{\theta}$ - e.g. to see if it is gaussian. **A simple way to grid search it is to parameterise it by a scalar $\hat{\beta}$ such that $\hat{\theta} = f(\hat{\beta})$.**

### 3.Measure the Convergences

We should have nice guarantees that

a. $\theta_1$ will converge to $\theta_0$ as N approaches infinity (LLN, quite computationally expensive). **Note that $\theta_1$ is the paramter for data explanation.** 

b. $\theta_1^{\prime}$ will converge to $\theta_1$ tightly (upto a high order residue).

c. **The tricky part is to find if $\hat{\theta}$ converges to $\theta_1$.** This might not really be the case as $\hat{\theta}$ is having a much larger search space than $\theta_1$. We can measure that by calculating KLD of the distributions / FRO norm of the modulator vectors.

But I am guessing even they don't match nicely, the general PofM kernel will have a better marginal likelihood...
    


### 4. Break Down the Marginal Likelihood

Check the data fit, model complexity between different GP models - see what we get.

So far the PoFM parameterization is having better result that the exact kernel in both data fit and lower model complexity.

But does it guarantees generalization power? - need to check.

## Thoughts


### 1. Overfitting

We found that the PoFM model is giving much higher log marginal likelihood, a lot of time with better data fit term - but the fitted variance is typically very small - and this is a clear sign for overfitting. It might make sense to add a prior in the fitted noise variance to avoid that.


### 2. Inconsistent theta parameter (between the PoFM model and the Exact Diffusion model)

We have observed examples where the fitted theta param in the PoFM model is almost same as the exact diffusion model - but this is not common. For example, we would normally expect the 0th term of the modulator being one - but this is oftern not the case. Let's discuss with Isaac the physical meaning of the modulator vector and see what constraint / scaling we might need to add.

### 3. Convergence Issue

We all know the the maximum eigen value of the normalized laplacian matrix is 2 - which means the p-step walk matrix can diverge as you increase the maximum walk length. (For instance, you would expect the modulator coefficient to converge to zero quickly.) While this may be good for a convergence point of view, it is typically hard for a gradient descend algorithm to automatically learn a modulator vector that converges as the walk-length increase. 

Shall we manually add some converging condition? Alternatively, we can give the model 0.5 * laplacian, which has a maximum eigen value of 1 and (so it is kind of like adding a 0.5^k term into the modulator directly).

## Topics of Discussion

1. GRF performance v.s. Graph Size
2. GRF bounds - measuring the frobinious norm between GRF-based kernel and Exact Diffusion Kernel
3. Learn the full modulator
4. performances (log marginal likelihood)
5. Problems & Constraints


# TO DO

1. clamp the noise
2. test /train data
3. scale the graph - normalizing the 



# Thinking

What have we found so far:

The PoFM method is getting close log marginal likelihood compared to the true model - but it is not guaranteed to have better data fit but worse complexity.

We also observed that lots of time it is getting a kernel matrix that is having every different modulator compared to the exact diffusion kernel - but using FRO might be better.

Also we know that, as we increase the number of nodes - the true distribution - of exact diffusion kernel should be much more likely - so increasing graph size might have been a good idea.

So why don't we do a plot:

variables:
- graph size: 20, 40, 80, 160, 320 (avg degree 10)
- approximation degree of the PoFM (try out 5 different random seeds) [1, 2, 3, ... 10]
- plot FRO, data fit term, complexity term

But equally I am very interested in the graph scaling - only if we scale the normalized laplacian even further

# Visualization Experiments Conclusions

1. Similar Level of Log Likelihood is Produced  Between the Exact Diffusion Kernel and the PoFM
2. The larger the graph size, the smaller the FRO
3. FRO is smaller for smaller approximation degree
4. FRO is smaller for smaller beta
5. PoFM is terrible for degree > 8 - model becomes very complex

The next thing to do is to do **the masking**

