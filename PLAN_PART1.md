# Part 1 — Multilevel modelling of Airbnb ratings from review counts alone

A plan for implementing, illustrating and extending Chapter 13 of *Statistical
Rethinking* (2nd ed.) on Airbnb listing data.

**The idea.** An overall rating `y ∈ [1,5]` is read as the fraction of customers
who had a positive experience, `p = (y−1)/4`. With `N` reviews that gives a
success count `P = round(p·N)` out of `N` trials — which is exactly the tadpole
survival data of Chapter 13, with **listings in place of tanks** and **reviews in
place of tadpoles**.

---

## Contents

- [1. Problems in the original proposal](#1-problems-in-the-original-proposal)
- [2. File structure](#2-file-structure)
- [3. §0 — Data prep](#3-0--data-prep)
- [4. §1 — Subsample and binomial outcome](#4-1--subsample-and-binomial-outcome)
- [5. §2 — The two models](#5-2--the-two-models)
- [6. §3 — The σ grid](#6-3--the-σ-grid)
- [7. §4 — Robustness check on the σ curve](#7-4--robustness-check-on-the-σ-curve)
- [8. §5 — Figure 1: σ agreement](#8-5--figure-1-σ-agreement)
- [9. §6 — Figure 2: shrinkage](#9-6--figure-2-shrinkage)
- [10. Outputs](#10-outputs)
- [11. Deviations from the original spec](#11-deviations-from-the-original-spec)

---

## 1. Problems in the original proposal

Six issues were found in the initial specification. Two of them change what gets
written; the rest change how it is described.

### 1.1 `P` is data, not a parameter with a prior

The proposal said "a quantity P (positive reviews) whose prior is set to be
Binomial with N, p". `P_i` is the **outcome**; `Binomial(N_i, p_i)` is the
**likelihood**. The prior is on `α_listing[i]`.

This matters for the writeup, not the code — the code is identical either way —
but the vocabulary must be right or the essay reads as confused. This is exactly
`S_i ~ Binomial(N_i, p_i)` in the Reed-frog model (R code 13.1).

### 1.2 `dnorm(0,1)` for the no-pooling intercepts sabotages the comparison

Airbnb ratings cluster around 4.7–4.8. So `p_i ≈ (4.75−1)/4 ≈ 0.94`, i.e.
`logit(p_i) ≈ 2.7`. A `Normal(0,1)` prior puts the **typical listing** at 2.7
prior SDs from its mean. That prior isn't weak — it's a hard pull toward
`p = 0.5`, which is itself a form of pooling (toward a fixed, wrong point). Two
consequences:

- The "no pooling" model isn't really no-pooling, so the intended contrast is
  contaminated.
- Worse, heavy fixed regularization **lowers** pWAIC. The no-pooling model could
  easily show the *smaller* pWAIC, and the headline claim collapses.

**Fix.** Use the book's `dnorm(0, 1.5)` as the headline (fidelity to m13.1), and
add one sensitivity fit with `dnorm(0, 5)` — a genuinely flat no-pooling prior —
to prove the pWAIC gap is not an artifact of prior width. Same for `α_bar`: the
book uses `dnorm(0, 1.5)`.

Note also that the pWAIC ordering is an **empirical** result, not a theorem. It
holds when the learned `σ` is tighter than the fixed prior. Expected `σ ≈
0.5–1.0`, comfortably tighter than 1.5, so it should work — but the writeup
should explain the **mechanism** ("the adaptive prior is tighter than the fixed
one, so each intercept is worth less than a full parameter"), and then the result
is interesting whichever way it lands.

### 1.3 "the σ maximizing PSIS is *exactly* the σ maximizing the posterior" is not exactly true

The near-miss is the interesting part. The marginal posterior of `σ` in the
`dexp(1)` model is

$$p(\sigma \mid y) \propto p(y \mid \sigma)\cdot \text{Exp}(\sigma \mid 1)$$

so its mode maximizes `log p(y|σ) − σ`. Meanwhile the fixed-σ grid model's LOO
score is `Σᵢ log p(yᵢ | y₋ᵢ, σ)`. Those two objectives are asymptotically
equivalent but **not identical**:

- LOO conditions every point on `n−1` others; the marginal likelihood conditions
  on `0, 1, …, n−1` others (the prequential decomposition).
- The `Exp(1)` prior pulls the posterior mode slightly **down** relative to the
  CV optimum.

**The honest claim:** the CV-optimal σ lands at (or very near) the posterior
mode, well inside the 89% interval — *the hierarchical model learns by
regularization the same thing cross-validation finds by brute force.* That is a
stronger and more defensible point than "exactly equal", and it is the actual
Chapter 13 message (McElreath: the varying-intercept model is a regularizing
prior whose strength is learned from the data).

Do **not** claim exact equality. It is false and a knowledgeable reader will
catch it.

### 1.4 `PSIS` in the `rethinking` package is lower-is-better

It is reported on the deviance scale (`−2·elpd`). The proposal said "maximizes
the PSIS value". Either plot `elpd = −PSIS/2` and maximize, or plot `PSIS` and
minimize. Pick one and be consistent — mixing them up in a figure caption is an
easy own-goal.

### 1.5 PSIS will be shaky here, for a structural reason

Every listing has exactly **one** binomial observation. So dropping observation
`i` removes *all* data about `α_i`. The importance weights are therefore
extremely heavy-tailed, and Pareto-k warnings should be expected on nearly every
point.

- **Conceptually this is excellent** for the demonstration: `p(yᵢ | y₋ᵢ)` becomes
  literally "predict a brand-new listing from the population", which is precisely
  what `σ` governs. The σ-grid experiment measures exactly the right thing.
- **Numerically it is a risk.** Hence the direct leave-one-cluster-out
  computation in [§4](#7-4--robustness-check-on-the-σ-curve), plus WAIC plotted
  on the same grid as a third curve.

### 1.6 Data volume

~70k listings = 70k intercepts per fit, and the σ grid needs ~12 fits. Not
viable. Subsample to **200 listings**, stratified across review counts — about 4×
the book's 48 tanks, and each fit takes seconds.

### 1.7 Two minor data notes

- `number_of_reviews` counts all reviews; `review_scores_rating` is computed only
  from reviews that carried a rating — so `N_i` slightly overstates the binomial
  trials.
- `review_scores_rating` is a **mean of 1–5 stars**, not a proportion of
  positives; `(y−1)/4` reinterprets it as one.

Both are fine as stated modelling assumptions — they just must be stated, not
left silent.

---

## 2. File structure

```
prep_data.R              # one-time: 222 MB csv -> data/ratings_small.csv
part1_multilevel.R       # everything below
outputs/part1/           # figures + text tables
```

`listings.csv` is 222 MB; reading it on every run wastes minutes. `prep_data.R`
runs once, selects 4 columns, filters, and writes a small csv.
`part1_multilevel.R` reads only that.

---

## 3. §0 — Data prep

```r
library(tidyverse)
d <- read_csv("listings.csv") %>%
  select(id, host_id, review_scores_rating, number_of_reviews) %>%
  filter(!is.na(review_scores_rating), number_of_reviews > 0,
         review_scores_rating >= 1, review_scores_rating <= 5)
write_csv(d, "data/ratings_small.csv")
```

Also emit the motivating table here — SD of rating by review-count bucket. It
costs nothing and is the best one-paragraph motivation for the whole chapter.

---

## 4. §1 — Subsample and binomial outcome

```r
d <- read_csv("data/ratings_small.csv")

# stratify so the shrinkage plot spans the full range of evidence
set.seed(13)
d <- d %>%
  mutate(bin = cut(number_of_reviews, c(0,2,5,10,50,Inf))) %>%
  group_by(bin) %>% slice_sample(n = 40) %>% ungroup()   # 200 listings

d$P <- round((d$review_scores_rating - 1) / 4 * d$number_of_reviews)
d$listing <- 1:nrow(d)

dat <- list(
  P       = d$P,
  N       = d$number_of_reviews,
  listing = d$listing
)
```

Five bins × 40 = 200. The stratification is deliberate: a random sample would be
dominated by the mass of the distribution and the shrinkage figure would have no
low-`N` listings to show shrinking.

---

## 5. §2 — The two models

Mirrors R code 13.2 / 13.3.

```r
# no pooling
m13.1 <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(0, 1.5)
  ), data = dat, chains = 4, log_lik = TRUE)

# multilevel / partial pooling
m13.2 <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(a_bar, sigma),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1)
  ), data = dat, chains = 4, log_lik = TRUE)

compare(m13.1, m13.2)
```

Plus the sensitivity fit `m13.1b` with `dnorm(0, 5)` per [§1.2](#12-dnorm01-for-the-no-pooling-intercepts-sabotages-the-comparison).
Report `pWAIC` for all three in one table.

**Watch for:** divergences from the funnel. If they appear, the non-centered form
(R code 13.22 style) is the fix — but try centered first. With 200 clusters and
informative `N_i` it will probably be fine, and centered code is what the book
shows at this point.

---

## 6. §3 — The σ grid

Fit the same multilevel model with `σ` pinned. Cleanest way to loop without
metaprogramming: **pass σ in as data.**

```r
sigma_grid <- exp(seq(log(0.15), log(3), length.out = 12))

fits <- list()
for (i in seq_along(sigma_grid)) {
  dat_s <- c(dat, list(sigma_fixed = sigma_grid[i]))
  fits[[i]] <- ulam(
    alist(
      P ~ dbinom(N, p),
      logit(p) <- a[listing],
      a[listing] ~ dnorm(a_bar, sigma_fixed),
      a_bar ~ dnorm(0, 1.5)
    ), data = dat_s, chains = 4, log_lik = TRUE)
}

psis_curve <- sapply(fits, function(m) PSIS(m)$PSIS)
waic_curve <- sapply(fits, function(m) WAIC(m)$WAIC)
```

Choose the grid **after** fitting `m13.2`, so it brackets the posterior: roughly
`[0.25·σ̂, 3·σ̂]` on a log scale. That is not cheating, it just avoids wasting 12
fits on the tails — but say so in the writeup.

`a_bar` stays free. Only `σ` is fixed — otherwise two things change at once.

---

## 7. §4 — Robustness check on the σ curve

Because of [§1.5](#15-psis-will-be-shaky-here-for-a-structural-reason), add a
direct computation that does not rely on importance sampling. For a fixed `σ`,
the leave-one-out predictive for listing `i` is available by Monte Carlo, since
dropping `i` removes all information about `α_i`:

$$p(P_i \mid y_{-i}, \sigma) \;=\; \int \text{Binomial}(P_i \mid N_i, \text{logistic}(a))\,\text{Normal}(a \mid \bar\alpha, \sigma)\,da$$

averaged over the posterior draws of `ᾱ`. In R: draw
`a_sim ~ rnorm(n_draws, post$a_bar, sigma)`, evaluate
`dbinom(P[i], N[i], inv_logit(a_sim))`, average, take the log, sum over `i`. Six
lines. Plot it alongside the PSIS curve.

If the two curves peak at the same σ, the PSIS result is trustworthy. If they
diverge, the direct one is right — and there is then an honest paragraph about
why PSIS struggles with one observation per cluster, which is itself a good
extension result.

---

## 8. §5 — Figure 1: σ agreement

Two stacked panels sharing the x-axis (σ, log scale):

- **Top:** `elpd_PSIS(σ)` across the grid, points + line. Mark the argmax with a
  vertical dashed line. Optionally overlay the direct-LOO curve from §4 and WAIC.
- **Bottom:** the posterior density of `σ` from `m13.2` (`dens(post$sigma)`),
  with its mode marked and its 89% interval shaded.

The payoff is visual: the dashed line from the top panel falls on the peak of the
bottom panel. **Two completely different criteria — one predictive, one Bayesian
— pick the same regularization strength.**

Two stacked panels, *not* a dual axis. Dual axes would invite the reader to
compare heights that are not comparable.

---

## 9. §6 — Figure 2: shrinkage

Adapted from R code 13.4.

```r
post <- extract.samples(m13.2)
d$p_est <- apply(inv_logit(post$a), 2, mean)
d$p_emp <- (d$review_scores_rating - 1) / 4
```

- x-axis: listing index, **sorted ascending by `number_of_reviews`**
- y-axis: proportion positive (0–1), or rescaled to 1–5 via `1 + 4p` — pick one
  and label it; the 1–5 scale is what a reader recognizes
- open points: empirical `p_emp`; filled points: posterior mean `p_est`; a thin
  segment joining each pair so the **direction and size** of the move is visible
- horizontal dashed line at `inv_logit(mean(post$a_bar))`
- vertical dividers at the review-count bin boundaries, labelled ("1–2 reviews",
  "3–5", …)

What it will show: on the left (few reviews) the segments are long and all point
toward the dashed line; on the right (many reviews) they shrink to nothing. That
is Figure 13.1 of the book, on Airbnb data.

**One adaptation:** McElreath's 48 tanks fit on one axis; 200 listings will be
crowded. Either drop to ~60 listings for this figure only (12 per bin), or keep
200 with small semi-transparent points. Preference: 60 for the figure, noting it
is a subset of the fitted 200.

---

## 10. Outputs

| file | contents |
|---|---|
| `00_noise_by_reviews.txt` | SD of rating by review-count bucket (motivation) |
| `10_compare.txt` | `compare(m13.1, m13.1b, m13.2)` — the pWAIC table |
| `11_precis_m13.2.txt` | `a_bar`, `sigma`, diagnostics |
| `20_sigma_grid.txt` | σ, PSIS, WAIC, direct-LOO per grid point |
| `fig1_sigma_agreement.png` | §5 |
| `fig2_shrinkage.png` | §6 |

---

## 11. Deviations from the original spec

1. `dnorm(0,1)` → `dnorm(0,1.5)` on the no-pooling intercepts, plus a
   `dnorm(0,5)` sensitivity fit.
2. The σ claim softened from "exactly" to "at the posterior mode / inside the 89%
   interval", with the direct-LOO cross-check added.
3. 200 stratified listings, not the full 70k.

Everything else in the original spec stands. The structure — two models →
`compare()` → σ grid → shrinkage plot — is exactly the Chapter 13 arc, and the
σ-grid experiment is a genuinely good extension: it is not in the book, and it
makes the "regularization learned from data" claim concrete rather than asserted.
