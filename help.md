# TASK SPECIFICATION: Bayesian Multilevel Rating Models with Partial Pooling & Shrinkage (Statistical Rethinking Ch. 13 Extension)

## 1. PROJECT OBJECTIVE & BACKGROUND
We are implementing a practical extension to Chapter 13 of Richard McElreath's *Statistical Rethinking* (2nd Edition) focusing on Multilevel Models, Varying Intercepts, Partial Pooling, and Bayesian Shrinkage.

Our primary goal is to produce **shrunken, un-overfitted baseline ratings** for Airbnb listings (`id`) clustered by host (`host_id`), explicitly incorporating sample-size uncertainty (`number_of_reviews`).

### The Core Problem:
- Raw sample average ratings for listings with low review counts ($N_i = 1$ or $2$) are highly noisy and prone to extreme values ($5.0$ or $1.0$).
- Raw unpooled averages give undue credibility to low-sample listings.
- Multilevel partial pooling pulls estimates of small-sample listings toward the global population mean ($\bar{\alpha}$), while allowing large-sample listings ($N_i > 100$) to retain their empirical average.

---

## 2. DATASET SPECIFICATION & PREPROCESSING
Assume a CSV file named `listings.csv` with the following columns:
- `id`: Unique identifier for each listing/place.
- `host_id`: Identifier for the host (can have multiple listings per host).
- `review_scores_rating`: Observed overall rating (typically 0–100 or 1–5 scale).
- `number_of_reviews`: Total review count ($n_i$).
- `price`: Price string (e.g., "$150.00").

### Preprocessing Requirements in R:
1. Filter `!is.na(review_scores_rating)`, `number_of_reviews > 0`, and clean `price` using `readr::parse_number()`.
2. Convert scale: Normalize `review_scores_rating` into standard scale $y_i \in [1, 5]$ or standard z-score $z_i = \frac{y_i - \bar{y}}{s_y}$ depending on the likelihood chosen.
3. Create clean contiguous integer index variables starting at 1 for `ulam`:
   - `df$listing_idx <- as.integer(as.factor(df$id))`
   - `df$host_idx    <- as.integer(as.factor(df$host_id))`
4. Standardize continuous predictors: `log_rev = log(number_of_reviews) - mean(log(number_of_reviews))`.

---

## 3. THREE MODELING APPROACHES TO IMPLEMENT

Implement all three distinct models in R using the `rethinking` package (`ulam()` engine with Stan HMC).

---

### MODEL 1: Aggregated Gaussian with Sample-Size Weighted Variance (Approach A)
Continuous Gaussian likelihood where the observational standard deviation is scaled inversely by $\sqrt{n_i}$, shrinking listings with fewer reviews.

**Mathematical Formula:**
$$y_i \sim \text{Normal}\left(\mu_i, \frac{\sigma}{\sqrt{n_i}}\right)$$
$$\mu_i = \alpha_{\text{listing}[i]} + \beta_{\text{rev}} \cdot \text{log\_rev}_i$$
$$\alpha_j \sim \text{Normal}(\bar{\alpha}, \sigma_{\text{listing}})$$
$$\bar{\alpha} \sim \text{Normal}(4.0, 1.0)$$
$$\beta_{\text{rev}} \sim \text{Normal}(0, 0.5)$$
$$\sigma_{\text{listing}} \sim \text{Exponential}(1)$$
$$\sigma \sim \text{Exponential}(1)$$

---

### MODEL 2: Binomial Success-Count Representation (Approach B)
Model total rating points obtained out of maximum possible points as a Binomial process (analogous to the Reed Frog survival model in Chapter 13).

**Transformation:**
- Let $S_i = \text{round}\left( \frac{y_i - 1}{4} \times n_i \right)$ (number of "success units" scaled to $[0, 1]$ out of $n_i$ trials).

**Mathematical Formula:**
$$S_i \sim \text{Binomial}\left(n_i, p_i\right)$$
$$\text{logit}(p_i) = \alpha_{\text{listing}[i]} + \beta_{\text{rev}} \cdot \text{log\_rev}_i$$
$$\alpha_j \sim \text{Normal}(\bar{\alpha}, \sigma_{\text{listing}})$$
$$\bar{\alpha} \sim \text{Normal}(0, 1)$$
$$\beta_{\text{rev}} \sim \text{Normal}(0, 0.5)$$
$$\sigma_{\text{listing}} \sim \text{Exponential}(1)$$

---

### MODEL 3: Hierarchical Host-Listing Cross-Classified / Nested Intercept Model (Approach C)
Incorporate host-level variation (`host_idx`) alongside listing-level variation (`listing_idx`) to exploit shared host quality across multiple listings.

**Mathematical Formula:**
$$y_i \sim \text{Normal}\left(\mu_i, \frac{\sigma}{\sqrt{n_i}}\right)$$
$$\mu_i = \bar{\alpha} + a_{\text{host}[i]} + a_{\text{listing}[i]} + \beta_{\text{rev}} \cdot \text{log\_rev}_i$$
$$a_{\text{host}} \sim \text{Normal}(0, \sigma_{\text{host}})$$
$$a_{\text{listing}} \sim \text{Normal}(0, \sigma_{\text{listing}})$$
$$\bar{\alpha} \sim \text{Normal}(4.0, 1.0)$$
$$\beta_{\text{rev}} \sim \text{Normal}(0, 0.5)$$
$$\sigma_{\text{host}} \sim \text{Exponential}(1)$$
$$\sigma_{\text{listing}} \sim \text{Exponential}(1)$$
$$\sigma \sim \text{Exponential}(1)$$

---

## 4. BENCHMARK & COMPARISON REQUIREMENTS

Alongside the three Bayesian multilevel models, implement a **No-Pooling Benchmark** (independent sample averages per listing) and an **Unpooled Fixed-Effects Model**.

### Analysis & Deliverables Required from Script:

1. **Model Comparison via WAIC & LOO:**
   - Use `rethinking::compare(m1, m2, m3)` (or equivalent WAIC/LOO calculation) to compute WAIC, $p_{\text{WAIC}}$ (effective number of parameters), $d\text{WAIC}$, and Akaike weights.

2. **Shrinkage Visualization Script:**
   - Plot raw empirical mean rating vs. posterior mean estimate $\hat{\mu}_i$ for each listing.
   - Color-code or size points by $n_i$ (number of reviews) to visually illustrate that listings with $n_i < 5$ undergo high shrinkage toward $\bar{\alpha}$, whereas $n_i > 50$ undergo near-zero shrinkage.

3. **Covariate Effect Analysis:**
   - Report posterior distribution for $\beta_{\text{rev}}$ (slopes showing how review counts correlate with baseline quality).

4. **Diagnostics & Robustness:**
   - Check $\hat{R}$ (Rhat < 1.01) and effective sample sizes ($n_{\text{eff}} > 500$) across all parameters using `precis(fit, depth=2)`.
   - Ensure non-centered parameterization is available if divergent transitions occur during HMC sampling.

---

## 5. OUTPUT DESIRED
Please generate a single, clean, end-to-end executable R script (`rethinking_ratings_extension.R`) containing:
1. Complete library loading (`tidyverse`, `rethinking`, `cmdstanr`).
2. Synthetic data generator (if `listings.csv` is not present) matching real Airbnb columns so the script runs out-of-the-box.
3. Clean data transformation and indexing.
4. Model definitions using `alist()` and compilation via `ulam()`.
5. Summary output generation (`precis()`, `compare()`).
6. `ggplot2` visualization code for shrinkage plots and posterior predictive checks.