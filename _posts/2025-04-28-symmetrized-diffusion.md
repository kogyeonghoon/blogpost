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
where $$w^{(t)}$$ is the standard d-dimensional Wiener process. This process has Gaussian transition kernels $$p_{t|s}(y|x)$$ and marginals $$p_t(x)$$ for $$0 \leq s < t \leq 0$$. The reverse denoising process follows the following SDE:
$$dx^{(t)} = \left[-\frac{1}{2}x^{(t)} - \nabla_{x^{(t)}} \log p_t(x^{(t)})\right]dt + d\overline{w}^{(t)} $$
where $$\overline{w}^{(t)}$$ is a Wiener process running in reverse time. The backward transition kernel parametrized by $$q_{\theta, s|t}$$, with a model parameter $$\theta$$, enabling sampling from the data distribution by solving the reverse SDE.

The score function $$\nabla_{x^{ (t) }} \log p_{t}(x^{ (t) })$$ is approximated by a neural network $$s_{ \theta }( x^{ (t) }, t)$$, trained using a denoising score loss. Given an initial sample $$x^{(0)}$$ and a noisy sample $$x^{ (t) }$$ obtained by perturbing $$x^{ (0) }$$ according to the forward process, the denoising score loss trains $$s_{\theta}( x^{ (t) }, t)$$ to estimate $$\nabla_{ x^{ (t) } } \log p_{t|0}{ ( x^{ (t) } | x^{ (0) }) }$$. This objective provides an unbiased estimate for the score function $$\nabla_{ x^{ (t) } } \log p_{t}( x^{ (t) } )$$. Note that, the target quantity $$\nabla_{ x^{ (t) } } \log p_{t|0}( x^{ (t) } | x^{ (0) } )$$ can be expressed in a closed-form equation since the transition kernel $$p_{t|0}$$ is a closed form Gaussian.

***
### Heat Kernel and Score Function Computation

Beneath the formulation of diffusion models, there is an important fact that the **heat kernel** $$K(t,x,y)$$, solving $$\frac{\partial}{\partial t}K = \Delta_y K$$ with the initial condition $$\lim_{t \rightarrow 0} K = \delta(x-y)$$, is Gaussian:
$$K(t,x,y) = \frac{1}{(4\pi t)^{d/2}} \exp\left(-\frac{\|x-y\|^2}{4t}\right).$$
Since the forward process distributions are Gaussian, their score functions (e.g., $$\nabla_{x^{(t)}} \log p_{t|0}(x^{(t)}|x^{(0)})$$) have closed-form expressions, enabling training of a neural network $$s_{\theta}(x^{(t)}, t)$$ to approximate them.

***
## Method: Permutation Symmetrized Diffusion


***
### Rethinking Permutation Symmetry in Diffusion Models


For 3D molecular generation, data is a point cloud $$x = (x_1, \dots, x_N)$$ where $$x_i \in \mathbb{R}^d$$, residing in the space $$\mathcal{X} = \mathbb{R}^{d \times N}$$.
Molecules exhibit permutation symmetry under the group $$S_N$$, in a sense that for $$\sigma \in S_N$$, its group action $$\sigma(x) = (x_{\sigma^{-1}(1)}, \dots, x_{\sigma^{-1}(N)})$$ gives the same molecule.
A common way for respecting the permutation symmetry is using **permutation-equivariant transition kernels**:
$$p_{t|s}(x^{(t)}|x^{(s)}) = p_{t|s}(\sigma(x^{(t)})|\sigma(x^{(s)})) \quad \text{(forward)},$$
$$q_{\theta,s|t}(x^{(s)}|x^{(t)}) = q_{\theta,s|t}(\sigma(x^{(s)})|\sigma(x^{(t)})) \quad \text{(reverse)}.$$
The forward equivariance is achieved designing a permutation-equivariant forward SDE, and the reverse equivariance is achieved by using a permutation-equivaraint neural network. In this approach, *the process behaves identically* under permutation -- if the input is altered by a permutation, the otuput is altered accordingly by the same permutation.

On the other hand, we can think of a stronger notion of utilizing the symmetry. Instead of building an equivariant diffusion on $$\mathcal{X} = \mathbb{R}^{d\times N}$$, we model the diffusion process on the quotient manifold $$\tilde{\mathcal{X}} = \mathbb{R}^{d\times N} / S_N$$. Let $$\pi : {\mathcal{X}} = \mathbb{R}^{d\times N} \rightarrow \tilde{\mathcal{X}} = \mathbb{R}^{d\times N} / S_N $$ denote the quotient map, and let's write $$\tilde{x} = \pi(x)$$ for brevity. In this formulation, we have simply $$\tilde{\sigma(x)} = \tilde{x}$$, i.e. *the elements are identical* under permutation.

***
### Heat Kernel on the Quotient Manifold

Since the permutations are isometries of the Euclidean space, $$\tilde{\mathcal{X}}$$ inherits the Euclidean metric and has a well-defined heat kernel. The heat kernel $$K^{\tilde{\mathcal{X}}}(t, \tilde{x}, \tilde{y})$$ on $$\tilde{\mathcal{X}}$$ for $$\tilde{x}, \tilde{y} \in \tilde{\mathcal{X}}$$ is given by:
$$K^{\tilde{\mathcal{X}}}(t, \tilde{x}, \tilde{y}) = \frac{1}{(4\pi t)^{dN/2}} \sum_{\sigma \in S_N} \exp\left(-\frac{\|x - \sigma(y)\|^2}{4t}\right)$$
$$= \frac{1}{(4\pi t)^{dN/2}} \sum_{\sigma \in S_N} \prod_{i=1}^{N} \exp\left(-\frac{\|x_i - y_{\sigma^{-1}(i)}\|^2}{4t}\right) \quad$$
This kernel is the sum of Euclidean heat kernels over all permutations, arising because permutations are Euclidean isometries. 

***
### SDEs on the Quotient Manifold

The Ornstein-Uhlenbeck process is pushed forward to $$\tilde{\mathcal{X}}$$. The **forward SDE** on $$\tilde{\mathcal{X}}$$ is:
$$d\tilde{x}^{(t)} = -\frac{1}{2}\tilde{x}^{(t)}dt + d\tilde{w}^{(t)}$$
and the **reverse SDE** is:$$d\tilde{x}^{(t)} = \left[-\frac{1}{2}\tilde{x}^{(t)} - \nabla_{\tilde{x}^{(t)}} \log \tilde{p}_t(\tilde{x}^{(t)})\right]dt + d\tilde{w}^{(t)}$$
Since the heat kernel is a sum over permutations, the transition kernels $$\tilde{p}_{t|s}$$ and marginals $$\tilde{p}_t$$ on the quotient manifold are similarly obtained by summing their Euclidean counterparts over all permutations.

***
### Training Objective

The target score function for the diffusion loss is $$\nabla \log \tilde{p}_t$$. Direct computation of $$\tilde{p}_t$$ by summing over all $$N!$$ permutations is intractable for non-trivial $$N$$. Letting $$I(\sigma) = -\frac{\|x-\sigma(y)\|^2}{4t}$$, the score function is:
$$\nabla_{\tilde{y}}\log\tilde{p}_{t}(\tilde{y}|\tilde{x}) = \sum_{\sigma\in S_{N}}\frac{exp(I(\sigma))}{\sum_{\sigma^{\prime}\in S_{N}}exp(I(\sigma^{\prime}))}\nabla_{y}I(\sigma) \quad$$
This is an expectation $$\mathbb{E}_{\mathcal{S}}[\nabla_{y}I(\sigma)]$$, where $$\mathcal{S}$$ is a distribution on $$S_N$$ with $$q(\sigma) \propto exp(I(\sigma))$$. Training involves sampling permutations to estimate this expected score.
Training involves sampling permutations $$\sigma_1, \dots, \sigma_K \sim \mathcal{S}$$ (e.g., using MCMC methods) and penalizing the model $$s_{\theta}$$ towards an unbiased estimate of this expected score.

## Significance

The proposed method of defining diffusion on the quotient manifold $$\tilde{\mathcal{X}}$$ intrinsically embeds permutation symmetry into the foundational space of the generative process. This contrasts with approaches that rely solely on equivariant architectures to handle such symmetries. This framework offers a principled way to model permutation-invariant data, with potential applications in 3D molecular generation and other domains.

For further details, please refer to the full research paper [Placeholder for Paper Link].