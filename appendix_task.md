# TASK: Write a 1-to-2-Page LaTeX Appendix on Bayesian Inference & Hamiltonian Monte Carlo (HMC)

## 1. Context & Objective
You are an expert technical writer and mathematical statistician. Your goal is to write a standalone, publication-quality LaTeX document that serves as an **Appendix** for a larger report on **Multilevel / Partial-Pooling Models**. 

The purpose of this appendix is to explain **how HMC works**, **why it is necessary for Bayesian inference**, and **the formal mathematical foundations backing it**.

---

## 2. Tone & Exposition Style
* **Intuition-First, Theorem-Backed:** Lead with physical and geometric intuition, then immediately anchor that intuition with formal mathematical definitions/theorems.
* **Level of Mathematical Rigor:** State formal theorems, definitions, and equations clearly (e.g., Target Distributions, Hamilton’s Equations, Leapfrog updates, Detailed Balance, Ergodic Theorem), but **DO NOT include proofs**. Provide the exact formal mathematical statements that justify *why* the intuition works.
* **Length Constraint:** Strictly **1 to 2 pages** when compiled in standard LaTeX (`article` class, 10pt/11pt font, standard margins). Keep explanations concise, dense, and structured.

---

## 3. Detailed Structure & Required Content

### Title & Header
* **Appendix Title:** *Appendix: Fundamentals of Hamiltonian Monte Carlo for Bayesian Posterior Sampling*

---

### Section 1: From Markov Chains to Metropolis-Hastings (1D Sampling)
1. **Markov Chain Monte Carlo (MCMC) Concept:**
   * Define a Markov chain with transition kernel $T(x \to x')$.
   * Explain the goal of MCMC: constructing a chain whose **stationary distribution** $\pi(x)$ matches an arbitrary target distribution $f(x) / Z$.
2. **The Metropolis-Hastings (MH) Algorithm:**
   * Describe the 1D/low-dimensional algorithm: Proposal step $x^* \sim q(x^* \mid x)$ followed by the acceptance check $\alpha = \min\left(1, \frac{f(x^*) q(x \mid x^*)}{f(x) q(x^* \mid x)}\right)$.
3. **Formal Backbone (Detailed Balance):**
   * State the **Detailed Balance Condition**: $\pi(x) T(x \to x') = \pi(x') T(x' \to x)$.
   * Briefly note why detailed balance implies invariance ($\int \pi(x) T(x \to x') dx = \pi(x')$).
4. **The Curse of Dimensionality:**
   * Explain intuitively why standard MH fails in high dimensions (random walk behavior, volume concentration in hyperspheres, exponential decay of acceptance rates).

---

### Section 2: Hamiltonian Monte Carlo (HMC) as High-Dimensional Extension
1. **The Phase Space Extension:**
   * Explain extending $D$-dimensional position space $\boldsymbol{\theta} \in \mathbb{R}^D$ to $2D$-dimensional phase space $(\boldsymbol{\theta}, \mathbf{p}) \in \mathbb{R}^{2D}$ by adding auxiliary momentum $\mathbf{p} \sim \mathcal{N}(0, \mathbf{M})$.
   * Define total energy via the Hamiltonian: $H(\boldsymbol{\theta}, \mathbf{p}) = U(\boldsymbol{\theta}) + K(\mathbf{p})$, where potential energy $U(\boldsymbol{\theta}) = -\ln f(\boldsymbol{\theta})$ and kinetic energy $K(\mathbf{p}) = \frac{1}{2}\mathbf{p}^T \mathbf{M}^{-1}\mathbf{p}$.
2. **Physical Intuition:**
   * Describe the physical analogy: a frictionless puck sliding on a multi-dimensional terrain defined by potential energy $U(\boldsymbol{\theta})$.
3. **Continuous Dynamics & Hamilton’s Equations:**
   * Present Hamilton's differential equations:
     $$\frac{d\boldsymbol{\theta}}{dt} = \frac{\partial H}{\partial \mathbf{p}} = \mathbf{M}^{-1}\mathbf{p}, \quad \frac{d\mathbf{p}}{dt} = -\frac{\partial H}{\partial \boldsymbol{\theta}} = -\nabla U(\boldsymbol{\theta})$$
4. **Formal Backing Theorems (No Proofs):**
   * **Conservation of Energy:** $\frac{dH}{dt} = 0 \implies H(\boldsymbol{\theta}(t), \mathbf{p}(t)) = H(\boldsymbol{\theta}(0), \mathbf{p}(0))$.
   * **Volume Preservation (Liouville's Theorem):** Divergence of velocity field is zero $\implies \det J_{\Phi_t} = 1$.
   * State why these properties guarantee continuous trajectories have an acceptance probability $\alpha = 1$.

---

### Section 3: Bayesian Inference & The Complete HMC Algorithm
1. **The Bayesian Posterior Challenge:**
   * Formally state Bayes' Theorem for parameters $\boldsymbol{\theta}$ and data $\mathcal{D}$:
     $$p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta})}{\int p(\mathcal{D} \mid \boldsymbol{\theta}) p(\boldsymbol{\theta}) d\boldsymbol{\theta}} = \frac{f(\boldsymbol{\theta})}{Z}$$
   * Identify $p(\mathcal{D})$ as intractable in high dimensions, making grid integration or conjugate updates impossible.
   * Define the potential energy specifically for Bayesian models: $U(\boldsymbol{\theta}) = -\ln p(\mathcal{D} \mid \boldsymbol{\theta}) - \ln p(\boldsymbol{\theta})$.

2. **Discrete Integration (Leapfrog Integrator):**
   * Explain why numerical integration is required in discrete time $\epsilon$.
   * State the 3-step **Leapfrog (Störmer-Verlet) Integrator**:
     1. $\mathbf{p}\left(t + \frac{\epsilon}{2}\right) = \mathbf{p}(t) - \frac{\epsilon}{2} \nabla U(\boldsymbol{\theta}(t))$
     2. $\boldsymbol{\theta}(t + \epsilon) = \boldsymbol{\theta}(t) + \epsilon \mathbf{M}^{-1} \mathbf{p}\left(t + \frac{\epsilon}{2}\right)$
     3. $\mathbf{p}(t + \epsilon) = \mathbf{p}\left(t + \frac{\epsilon}{2}\right) - \frac{\epsilon}{2} \nabla U(\boldsymbol{\theta}(t + \epsilon))$
   * **Formal Property (Symplecticity & Shadow Hamiltonian):** State that Leapfrog preserves phase volume exactly ($\det J = 1$), is time-reversible, and tracks a *Shadow Hamiltonian* $\tilde{H} = H + \mathcal{O}(\epsilon^2)$, preventing long-term energy drift. Contrast this with standard Forward Euler (which diverges).

3. **Complete Step-by-Step HMC Algorithm Box:**
   Provide an explicit pseudo-code algorithm block:
   * **Input:** Initial state $\boldsymbol{\theta}^{(0)}$, trajectory length $L$, step size $\epsilon$, mass matrix $\mathbf{M}$.
   * **Iteration $s = 1, \dots, S$:**
     1. Sample fresh momentum: $\mathbf{p}^{(s)} \sim \mathcal{N}(0, \mathbf{M})$.
     2. Set $(\boldsymbol{\theta}_0, \mathbf{p}_0) = (\boldsymbol{\theta}^{(s-1)}, \mathbf{p}^{(s)})$.
     3. Simulate $L$ Leapfrog steps using $\nabla U(\boldsymbol{\theta})$ to obtain $(\boldsymbol{\theta}^*, \mathbf{p}^*)$.
     4. Compute MH acceptance probability: $\alpha = \min\left(1, \exp\left(-H(\boldsymbol{\theta}^*, \mathbf{p}^*) + H(\boldsymbol{\theta}_0, \mathbf{p}_0)\right)\right)$.
     5. Set $\boldsymbol{\theta}^{(s)} = \boldsymbol{\theta}^*$ with probability $\alpha$; else $\boldsymbol{\theta}^{(s)} = \boldsymbol{\theta}^{(s-1)}$.

4. **Connecting Draws to Model Summaries (The Ergodic Theorem):**
   * State the **MCMC Ergodic Theorem**:
     $$\frac{1}{S} \sum_{s=1}^S g(\boldsymbol{\theta}^{(s)}) \xrightarrow[S \to \infty]{\text{a.s.}} \mathbb{E}_{p(\boldsymbol{\theta} \mid \mathcal{D})}[g(\boldsymbol{\theta})]$$
   * Explain briefly how this allows estimating posterior means, credible intervals, and predictions for hierarchical/partial pooling parameters (e.g., group intercepts $\alpha_j$ and hyperparameters $\bar{\alpha}, \sigma$) directly from column averages of the output sample matrix.

---