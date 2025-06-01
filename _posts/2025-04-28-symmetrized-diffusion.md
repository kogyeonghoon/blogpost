---
layout: distill
title: Permutation Symmetrized Diffusion Models for 3D Molecular Generation
description: This article introduces the core concepts behind Permutation Symmetrized Diffusion Models for 3D Molecular Generation, a novel approach that addresses the inherent permutation symmetry of molecular data.
date: 2025-04-28 # You can change this date as needed
authors:
  - name: Author One # Placeholder, please update
    # url: "Link to Author One's page"
    # affiliations:
    #   name: "Affiliation One"
  - name: Author Two # Placeholder, please update
    # url: "Link to Author Two's page"
    # affiliations:
    #   name: "Affiliation Two"
bibliography: 2025-04-28-symmetrized-diffusion.bib # Assumed bibliography file name. This file would need to be created with the corresponding citation keys if using <d-cite> tags.
toc:
  - name: Original Diffusion Process Euclidean Space
    subsections:
    - name: Forward and Reverse Stochastic Differential Equations SDEs
    - name: Heat Kernel and Score Function Computation
  - name: Method Permutation Symmetrized Diffusion
    subsections:
    - name: Rethinking Permutation Symmetry in Diffusion Models
    - name: Heat Kernel on the Quotient Manifold
    - name: SDEs on the Quotient Manifold
    - name: Training Objective
  - name: Significance
---

The generation of novel 3D molecular structures is a significant task in computational chemistry, drug discovery, and materials science. Diffusion-based generative models have emerged as a powerful class of methods for such tasks. In this article, we propose a novel formulation of diffusion models that incorporates a stronger notion of permutation symmetry.

## Diffusion in Euclidean space


***
### Forward and Reverse Stochastic Differential Equations (SDEs)

In common, diffusion models operate in Euclidean space $$\mathcal{X} = \mathbb{R}^d$$. The forward noising often uses an Ornstein-Uhlenbeck (OU) process:

$$dx^{(t)} = -\frac{1}{2}x^{(t)}dt + dw^{(t)} , \quad t \in [0,T]$$

where $$w^{(t)}$$ is the standard d-dimensional Wiener process. This process has Gaussian transition kernels $$p_{t \mid s}(y \mid x)$$ and marginals $$p_t(x)$$ for $$0 \leq s < t \leq 0$$. The reverse denoising process follows the following SDE:

$$dx^{(t)} = \left[-\frac{1}{2}x^{(t)} - \nabla_{x^{(t)}} \log p_t(x^{(t)})\right]dt + d\overline{w}^{(t)} $$

where $$\overline{w}^{(t)}$$ is a Wiener process running in reverse time. The backward transition kernel parametrized by $$q_{\theta, s \mid t}$$, with a model parameter $$\theta$$, enabling sampling from the data distribution by solving the reverse SDE.

The score function $$\nabla_{x^{ (t) }} \log p_{t}(x^{ (t) })$$ is approximated by a neural network $$s_{ \theta }( x^{ (t) }, t)$$, trained using a denoising score loss. Given an initial sample $$x^{(0)}$$ and a noisy sample $$x^{ (t) }$$ obtained by perturbing $$x^{ (0) }$$ according to the forward process, the denoising score loss trains $$s_{\theta}( x^{ (t) }, t)$$ to estimate $$ \nabla_{ x^{ (t) } } \log p_{t \mid 0} { ( x^{ (t) }  \mid  x^{ (0) }) } $$. This objective provides an unbiased estimate for the score function $$ \nabla_{ x^{ (t) } } \log p_{t} ( x^{ (t) } )$$. Note that, the target quantity $$\nabla_{ x^{ (t) } } \log p_{t \mid 0}( x^{ (t) }  \mid  x^{ (0) } )$$ can be expressed in a closed-form equation since the transition kernel $$p_{t \mid 0}$$ is a closed form Gaussian.

***
### Heat Kernel and Score Function Computation

Beneath the formulation of diffusion models, there is an important fact that the **heat kernel** $$K(t,x,y)$$, solving $$\frac{\partial}{\partial t}K = \Delta_y K$$ with the initial condition $$\lim_{t \rightarrow 0} K = \delta(x-y)$$, is Gaussian:

$$K(t,x,y) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{ \mid x-y \mid ^2}{4t}\right).$$

Since the forward process distributions are Gaussian, their score functions (e.g., $$\nabla_{x^{(t)}} \log p_{t \mid 0}(x^{(t)} \mid x^{(0)})$$) have closed-form expressions, enabling training of a neural network $$s_{\theta}(x^{(t)}, t)$$ to approximate them.

***
## Method: Permutation Symmetrized Diffusion


***
### Rethinking Permutation Symmetry in Diffusion Models


For 3D molecular generation, data is a point cloud $$x = (x_1, \dots, x_N)$$ where $$x_i \in \mathbb{R}^d$$, residing in the space $$\mathcal{X} = \mathbb{R}^{d \times N}$$.
Molecules exhibit permutation symmetry under the group $$S_N$$, in a sense that for $$\sigma \in S_N$$, its group action $$\sigma(x) = (x_{\sigma^{-1}(1)}, \dots, x_{\sigma^{-1}(N)})$$ gives the same molecule.
A common way for respecting the permutation symmetry is using **permutation-equivariant transition kernels**:

$$p_{t \mid s}(x^{(t)} \mid x^{(s)}) = p_{t \mid s}(\sigma(x^{(t)}) \mid \sigma(x^{(s)})) \quad \text{(forward)},$$

$$q_{\theta,s \mid t}(x^{(s)} \mid x^{(t)}) = q_{\theta,s \mid t}(\sigma(x^{(s)}) \mid \sigma(x^{(t)})) \quad \text{(reverse)}.$$

The forward equivariance is achieved designing a permutation-equivariant forward SDE, and the reverse equivariance is achieved by using a permutation-equivaraint neural network. In this approach, *the process behaves identically* under permutation -- if the input is altered by a permutation, the otuput is altered accordingly by the same permutation.

On the other hand, we can think of a stronger notion of utilizing the symmetry. Instead of building an equivariant diffusion on $$\mathcal{X} = \mathbb{R}^{d\times N}$$, we model the diffusion process on the quotient manifold $$\tilde{\mathcal{X}} = \mathbb{R}^{d\times N} / S_N$$. Let $$\pi : {\mathcal{X}} = \mathbb{R}^{d\times N} \rightarrow \tilde{\mathcal{X}} = \mathbb{R}^{d\times N} / S_N $$ denote the quotient map, and let's write $$\tilde{x} = \pi(x)$$ for brevity. In this formulation, we have simply $$\tilde{\sigma(x)} = \tilde{x}$$, i.e. *the elements are identical* under permutation.

***
### Heat Kernel on the Quotient Manifold

Since the permutations are isometries of the Euclidean space, $$\tilde{\mathcal{X}}$$ inherits the Euclidean metric and has a well-defined heat kernel. The heat kernel $$K^{\tilde{\mathcal{X}}}(t, \tilde{x}, \tilde{y})$$ on $$\tilde{\mathcal{X}}$$ for $$\tilde{x}, \tilde{y} \in \tilde{\mathcal{X}}$$ is given by:

$$K^{\tilde{\mathcal{X}}}(t, \tilde{x}, \tilde{y}) = \frac{1}{(4\pi t)^{dN/2}} \sum_{\sigma \in S_N} \exp\left(-\frac{ \mid x - \sigma(y) \mid ^2}{4t}\right)$$

$$= \frac{1}{(4\pi t)^{dN/2}} \sum_{\sigma \in S_N} \prod_{i=1}^{N} \exp\left(-\frac{ \mid x_i - y_{\sigma^{-1}(i)} \mid ^2}{4t}\right) \quad$$

In other words, the heat kernel on $$\tilde{\mathcal{X}}$$ is simply the sum of the Euclidean heat kernel over all the permutations. Below we list some core arguments for the proof:

- The permutations are isometries, so the Riemannian metric of the quotient manifold is locally identical to the Euclidean metric. Hence if a function satisfy the diffusion equation locally in the Euclidean space, then it also satisfy the diffusion equation in the quotient space.
- The diffusion equation is linear, so the a sum of the solution for the diffusion equation is also a solution.
- The heat kernel on a Riemannian manifold is unique, so if we find one valid solution, then it's the only solution.
- The function $$K^{\tilde{\mathcal{X}}}$$ satisfies $$\lim_{t \rightarrow 0} K^{\tilde{\mathcal{X}}} = \sum_{\sigma \in S_N} \delta (x - \sigma(y))$$, which is the Dirac delta in the quotient space.

Identifying the heat kernel $$K^{\tilde{\mathcal{X}}}$$ reveals a key distinction between Euclidean diffusion models and our approach. This diffusion process introduces the possibility of *particles sharing their identity*. In a classic diffusion process, $$x_1$$ diffuses to $$y_1$$, and $$x_2$$ diffuses to $$y_2$$, and so on. However, in our model, $$x_1$$ can diffuse to any component $$y_i$$ (for $$i=1,\dots,N$$), and all possible configurations $$\sigma \in S_N$$, representing the diffusion of $$x$$ to $$\sigma(y)$$, are considered. This resembles the behavior of *bosons* in particle physics.


***
### SDEs on the Quotient Manifold

The Ornstein-Uhlenbeck process is pushed forward to $$\tilde{\mathcal{X}}$$. The **forward SDE** on $$\tilde{\mathcal{X}}$$ is:

$$d\tilde{x}^{(t)} = -\frac{1}{2}\tilde{x}^{(t)}dt + d\tilde{w}^{(t)}$$

and the **reverse SDE** is:

$$d\tilde{x}^{(t)} = \left[-\frac{1}{2}\tilde{x}^{(t)} - \nabla_{\tilde{x}^{(t)}} \log \tilde{p}_t(\tilde{x}^{(t)})\right]dt + d\overline{\tilde{w}}^{(t)}$$

where $$\tilde{w}^{(t)}$$ is the Brownian motion in the quotient manifold $$\hat{\mathcal{X}}$$ and $$\overline{\tilde{w}}^{(t)}$$ is its time-reversed version. Note that obtaining the push-forward of the SDEs did not require complicated considerations becauese the OU process permutation invariant. Since the heat kernel is a sum over permutations, the transition kernels $$\tilde{p}_{t \mid s}$$ and marginals $$\tilde{p}_t$$ on the quotient manifold are similarly obtained by summing their Euclidean counterparts over all permutations.

***
### Training Objective

The target score function for the diffusion loss is $$\nabla \log \tilde{p}_t$$. Although we identified the transition kernel $$\tilde{p}_t$$, direct computation of $$\tilde{p}_t$$ is impossible becuase it involves summation over the permutation group that has a size $$N!$$. Instead, we look at the score function $$\nabla_{\tilde{y}}\log\tilde{p}_{t}(\tilde{y} \mid \tilde{x})$$ directly. Letting $$I(\sigma) = -\frac{ \mid x-\sigma(y) \mid ^2}{4t}$$, the score function is:

$$\nabla_{\tilde{y}}\log\tilde{p}_{t}(\tilde{y} \mid \tilde{x}) = \sum_{\sigma\in S_{N}}\frac{\exp(I(\sigma))}{\sum_{\sigma^{\prime}\in S_{N}}\exp(I(\sigma^{\prime}))}\nabla_{y}I(\sigma).$$

If we consider a distribution $$\mathcal{S}$$ on $$S_N$$ with the probability mass function $$q(\sigma) \propto \exp(I(\sigma))$$, the score function becomes the expectation $$\mathbb{E}_{\mathcal{S}}[\nabla_{y}I(\sigma)]$$.

Sampling permutations from $$\mathcal{S}$$ can be done by Markov chain Monte Carlo (MCMC) method. Define the cost matrix $$C = (C_{ij})$$ with $$C_{ij} = -\frac{(x_i - y_j)^2}{4t}$$. We use an MCMC starting from $$\sigma_0 = \text{id} \in S_N$$ and the proposals yielded by swapping entries of $$i,j$$, for $$i \in \{1,\cdots,N\}$$ sampled uniformly at random and $$j$$ sampled from distribution proportional to $$\exp(C_{ij})$$. Once the permutations $$\sigma_1,\cdots,\sigma_K \sim \mathcal{S}$$ are sampled, penalizing the model $$s_\theta$$ towards $$\mathbb{E}_{\mathcal{S}} [\nabla_{y} I(\sigma)] \approx \sum_{k=1}^K \nabla_{y} I(\sigma_k)$$ gives an unbiased estimate of the gradient:

$$ \nabla_\theta \big  \|s_{\theta} - \mathbb{E}_{\mathcal{S}} [\nabla_{y} I(\sigma)] \big \| ^2 \\
= \mathbb{E}_{\sigma_1,\cdots,\sigma_K \sim \mathcal{S}} \bigg[\nabla_\theta \big\| s_{\theta} - \sum_{k=1}^K \nabla_{y} I(\sigma) \big\| ^2 \bigg]
$$xw


## Significance

The proposed method of defining diffusion on the quotient manifold $$\tilde{\mathcal{X}}$$ intrinsically embeds permutation symmetry into the foundational space of the generative process. This contrasts with approaches that rely solely on equivariant architectures to handle such symmetries. This framework offers a principled way to model permutation-invariant data, with potential applications in 3D molecular generation and other domains.

For further details, please refer to the full research paper [Placeholder for Paper Link].