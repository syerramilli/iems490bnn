# Uncertainty quantification for neural networks

This repository contains code for my project for the course IEMS 490: Topics in Uncertainty Quantification. The objectives for the project are to compare different scalable uncertainty quantification mechanisms for neural networks. In particular, the methods being compared are

1. Ensemble neural networks, where each model in the ensemble corresponds to a different local optima.
2. Variational inference using independent Normals as the posterior distribution
3. Variational inference using a multivariate Normal distribution as the posterior distribution with the covariance matrix a sum of a diagonal and a low-rank matrix.
4. An adaptive variant of Stochastic gradient Hamiltonian Monte Carlo (SGHMC)
