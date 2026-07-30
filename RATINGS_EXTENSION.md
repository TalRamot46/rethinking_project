# `rethinking_ratings_extension.R` — a detailed walkthrough

A Bayesian multilevel treatment of Airbnb listing ratings, extending Chapter 13
of *Statistical Rethinking* (2nd ed.) — varying intercepts, partial pooling, and
shrinkage.

**The problem.** A listing with one review rated 5.0 is not better than a listing
with 400 reviews averaging 4.83. The raw average treats them as equals. This
script produces baseline ratings that discount low-evidence listings toward the
population mean, in proportion to how little evidence they actually have.

**Contents**

- [1. Quick start](#1-quick-start)
- [2. Why the raw average fails](#2-why-the-raw-average-fails)
- [3. The data structure — and why it is the Reed-Frog model](#3-the-data-structure--and-why-it-is-the-reed-frog-model)
- [4. The five models](#4-the-five-models)
- [5. Three problems found while building this](#5-three-problems-found-while-building-this)
- [6. Script anatomy, section by section](#6-script-anatomy-section-by-section)
- [7. Results](#7-results)
- [8. Output files](#8-output-files)
- [9. Reading the figures](#9-reading-the-figures)
- [10. Known limitations](#10-known-limitations)
- [11. Extending the script](#11-extending-the-script)

---

## 1. Quick start

```bash
Rscript rethinking_ratings_extension.R           # full run, ~7 min
RETHINK_SMOKE=1 Rscript rethinking_ratings_extension.R   # ~2 min sanity check
```

**Requires:** `tidyverse`, `rethinking`, `loo`, and ideally `cmdstanr` +
CmdStan (falls back to `rstan` automatically) and `data.table` (falls back to
`readr`). Verified on R 4.6.1, `rethinking` 2.42, CmdStan 2.39.0, ggplot2 4.0.3.

**Input:** `listings.csv` in the working directory, with columns `id`,
`host_id`, `review_scores_rating`, `number_of_reviews`, `price`. **If the file
is absent the script generates synthetic data with the same column names and
scales and runs anyway** — so it works out of the box, and the synthetic path
additionally unlocks a ground-truth RMSE check (§7.4).

Everything lands in `outputs/`. Nothing else is written.

### Configuration

All knobs live in one `CFG` list at the top:

| field | default | meaning |
|---|---|---|
| `csv_path` | `"listings.csv"` | absent ⇒ synthetic mode |
| `out_dir` | `"outputs"` | all tables and figures |
| `seed` | `1913` | fixed; runs are reproducible |
| `n_listings` | `1200` | subsample budget (see below) |
| `chains` / `cores` / `iter` | `4` / `4` / `2000` | HMC settings; warmup = `iter/2` |
| `adapt_delta` | `0.95` | raise toward `0.99` if divergences appear |
| `noncentered` | `TRUE` | non-centered parameterization |
| `use_cmdstan` | `TRUE` | auto-downgrades to `rstan` if CmdStan is missing |

**Why `n_listings = 1200` and not all 70,711.** Every listing gets its own
intercept, so the full dataset means ~70,000 parameters per model — hours of
sampling and gigabytes of posterior. 1200 listings is enough to show the
shrinkage structure cleanly while each model fits in 1–3 minutes. Raise it if
you want; nothing else needs to change.

---

## 2. Why the raw average fails

The script's first output table (`00_noise_by_review_count.txt`) measures the
problem on the real data before fitting anything:

| reviews | listings | mean rating | **SD of rating** |
|---|---|---|---|
| 1–2 | 13,998 | 4.567 | **0.868** |
| 3–5 | 10,615 | 4.655 | **0.496** |
| 6–10 | 9,959 | 4.699 | **0.341** |
| 11–50 | 24,332 | 4.741 | **0.239** |
| 50+ | 11,807 | 4.771 | **0.179** |

The spread collapses by a factor of ~4.8 as review counts grow, while the mean
barely moves. That is the signature of **sampling noise, not real quality
differences**: low-*n* listings scatter because their average is computed from
one or two reviews, not because they are genuinely more variable.

Concretely, of the 8,426 listings with exactly one review, the rating is an
integer — 6,308 are exactly 5.0 and 342 are exactly 1.0. Those are not the best
and worst listings in the city. They are listings with one review.

A multilevel model fixes this by treating each observed average as a noisy
measurement whose precision is known (it scales with `√n`), and pulling
imprecise measurements toward the population mean.

---

## 3. The data structure — and why it is the Reed-Frog model

**`id` is unique in `listings.csv`.** One row per listing, so every listing
"cluster" contains exactly one observation. This looks alarming for a multilevel
model but is exactly the structure of McElreath's Reed-Frog tadpole example in
Ch. 13, which also has one row per tank.

Partial pooling still works, because the outcome is **already an average**:

- `review_scores_rating` is the mean of `number_of_reviews` individual reviews.
- So its standard error is `σ / √n_i`, with `σ` the per-review SD.

That is the entire mechanism. The likelihood's scale term tells the model how
much to trust each row, and the adaptive prior does the pulling. A listing with
`n=1` has a wide likelihood and gets dragged to the mean; a listing with `n=400`
has a razor-thin likelihood and stays where it is.

### Preprocessing decisions

**Whole-host subsampling.** Rows are *not* sampled at random. Hosts are shuffled
and kept intact until the listing budget fills:

```r
keep_hosts <- host_sizes %>% slice_sample(prop = 1) %>%
  mutate(cum = cumsum(n_list)) %>% filter(cum <= CFG$n_listings) %>% pull(host_id)
```

Random *row* sampling would scatter hosts and leave almost no host with more
than one listing, which makes `sigma_host` in Model C unidentifiable. Keeping
clusters intact preserves the nesting: the 1200-listing sample retains 154
multi-listing hosts.

**`price` is parsed but never filtered on.** The spec asked for
`parse_number()`, which is applied — but 22,041 of 70,711 otherwise-usable rows
(31%) have no price, and **no model uses price**. Filtering on it would discard
a third of the rating data for nothing. It is kept as a descriptive column only.
This is a deliberate deviation.

**Index variables** are contiguous 1..N integers, as `ulam` requires:

```r
listing_idx = as.integer(as.factor(id))
host_idx    = as.integer(as.factor(host_id))
log_rev     = log(number_of_reviews) - mean(log(number_of_reviews))  # centered
```

**Binomial transform** (Model B only): `S_i = round((y_i − 1)/4 × n_i)`, mapping
a rating on [1,5] to "success units" out of `n_i` trials. Three of 70,711 real
rows have ratings below 1.0; they are clamped to 1.0 rather than dropped
silently, and the script reports the clamp count.

---

## 4. The five models

Let `y_i` be the observed rating, `n_i` the review count, and
`x_i = log_rev_i` the centered log review count.

### Benchmark 0 — no pooling at all

The raw observed rating itself. Since there is one row per listing, the
"independent sample average per listing" *is* `review_scores_rating`. The script
states this plainly rather than pretending to compute a group mean.

### `m0` — Unpooled fixed effects

$$y_i \sim \text{Normal}(\alpha_{\text{listing}[i]} + \beta_{\text{rev}}x_i,\ \texttt{obs\_sd}_i)$$
$$\alpha_j \sim \text{Normal}(4.0,\ 1.0) \quad\text{(fixed, non-adaptive)}$$

No `σ_listing`. Each listing is estimated in isolation — the overfitting
baseline. **`obs_sd` is supplied as data, not estimated**, for a reason
explained in §5.2.

### `m1` — Approach A: Gaussian with sample-size-weighted variance

$$y_i \sim \text{Normal}\!\left(\mu_i,\ \frac{\sigma}{\sqrt{n_i}}\right), \qquad \mu_i = \alpha_{\text{listing}[i]} + \beta_{\text{rev}} x_i$$
$$\alpha_j \sim \text{Normal}(\bar\alpha, \sigma_{\text{listing}}), \quad \bar\alpha \sim \text{Normal}(4.0, 1.0)$$
$$\beta_{\text{rev}} \sim \text{Normal}(0, 0.5), \quad \sigma_{\text{listing}} \sim \text{Exponential}(1), \quad \sigma \sim \text{Exponential}(1)$$

The core model. `σ` is the per-review SD; dividing by `√n_i` turns it into the
standard error of the observed average. `α_j` is the shrunken baseline quality.

### `m2` — Approach B: Binomial success counts

$$S_i \sim \text{Binomial}(n_i,\ p_i), \qquad \text{logit}(p_i) = \alpha_{\text{listing}[i]} + \beta_{\text{rev}} x_i$$
$$\alpha_j \sim \text{Normal}(\bar\alpha, \sigma_{\text{listing}}), \quad \bar\alpha \sim \text{Normal}(0, 1)$$

The direct Reed-Frog analogue. Its structural advantage: **it cannot predict an
impossible rating.** Shrinkage arises from the binomial variance itself — no
`σ` term is needed, because `Binomial(n, p)` already knows that small `n` means
weak evidence. Estimates are mapped back with `1 + 4p`.

### `m3` — Approach C: Cross-classified host + listing intercepts

$$y_i \sim \text{Normal}\!\left(\mu_i, \frac{\sigma}{\sqrt{n_i}}\right), \qquad \mu_i = \bar\alpha + a_{\text{host}[i]} + a_{\text{listing}[i]} + \beta_{\text{rev}} x_i$$
$$a_{\text{host}} \sim \text{Normal}(0, \sigma_{\text{host}}), \quad a_{\text{listing}} \sim \text{Normal}(0, \sigma_{\text{listing}})$$

Adds host-level variation so that a host's other listings inform a new listing's
estimate. Both offsets are centered at zero with `ā` carrying the global level.
See §10 for a serious identifiability caveat.

### Centered vs. non-centered

Every pooled model is written **twice**, and `CFG$noncentered` picks between
them. The non-centered form reparameterizes `a = ā + z·σ` with `z ~ Normal(0,1)`:

```r
transpars > vector[N_listing]:a_listing <<- a_bar + z_listing * sigma_listing,
vector[N_listing]:z_listing ~ dnorm(0, 1),
```

This removes the funnel geometry that causes divergent transitions when `σ` is
small. `transpars` keeps the reconstructed intercepts in the posterior, so all
downstream code is identical either way. Non-centered is the default, and with
it **all pooled models sample with zero divergent transitions.**

---

## 5. Three problems found while building this

These are not stylistic notes. Each would have silently corrupted results.

### 5.1 `ulam`'s generated `log_lik` is wrong for Models A and C

When a likelihood's scale contains a data vector, `ulam` generates:

```stan
for ( i in 1:N ) log_lik[i] = normal_lpdf( y[i] | mu[i] , sigma * inv_sqrt_n );
//                                                        ^^^^^^^^^^^^^^^^^^ not indexed by [i]
```

The vector is **unindexed**. Stan broadcasts, so each point's log-likelihood
becomes the sum of that point's density evaluated at *all N* scale values.

Verified on a 60-row test case:

| | `ulam`'s `log_lik` | correct |
|---|---|---|
| constant `σ` (control) | 27.773 | 27.773 ✅ max diff 9.2e-07 |
| `σ * inv_sqrt_n` | **177366.3** | **42.188** ❌ |

The mechanism was confirmed exactly: `log_lik[1] = −15.457 =`
`sum_j lpdf(y₁ | μ₁, σ·inv_sqrt_n[j])`.

**Consequence:** `rethinking::WAIC()`, `PSIS()`, and `compare()` cannot be used
on `m1` or `m3`. The script builds pointwise log-likelihood matrices in R and
scores them with `loo`:

```r
ll_gauss_scaled <- function(fit) {
  post <- extract.samples(fit)
  mu   <- mu_gauss(post)
  sd_i <- outer(as.vector(post$sigma), invs)   # sigma_s * inv_sqrt_n_i
  dnorm(Y(nrow(mu)), mu, sd_i, log = TRUE)
}
```

The control row above is the calibration: on a model where `ulam` *is* correct,
this same code reproduces `ulam`'s own values to 9.2e-07. The model fits
themselves were always fine — only the generated-quantities block was affected.

### 5.2 The unpooled benchmark is non-identified if it estimates its own `σ`

With one row and one free intercept per listing, the likelihood is **saturated**:
`a_listing[i]` can sit exactly on `y_i − β·x_i` for every `i`, so the density is
unbounded as `σ → 0` and the sampler chases it down a funnel.

Measured on a 150-listing trial: 18/600 divergent transitions, E-BFMI < 0.3 on
both chains, `σ` Rhat 2.14 and n_eff 2.7, and a **WAIC of −182 that "beat" every
pooled model.** That number is an artifact of the collapse, not evidence.

This degeneracy is itself the lesson — nothing stops no-pooling from fitting the
noise exactly; the adaptive prior in `m1`/`m3` is what holds `σ` up. The fix
pins `σ` at `m1`'s posterior mean:

```r
sigma_hat    <- mean(extract.samples(m1)$sigma)
obs_sd_fixed <- sigma_hat * dat_gauss$inv_sqrt_n
```

`m0` now has an observation model **identical** to `m1`'s, so the comparison
isolates pooling and nothing else. This is why `m0` is fitted *last*.

### 5.3 Model B is not WAIC-comparable to A and C

`m2`'s outcome is a count `S_i` out of `n_i` trials; A's and C's is the
continuous rating `y_i`. Information criteria are only comparable across models
of the **same outcome**. Putting all three in one `compare()` table — as a naive
reading of the spec would — produces a meaningless ranking.

The script therefore runs `compare()` on the Gaussian family only
(`m0`, `m1`, `m3`) and scores `m2` separately, comparing it to the others on the
rating scale via the shrinkage tables and figures instead.

---

## 6. Script anatomy, section by section

| § | What it does |
|---|---|
| **0** | `CFG` config, `RETHINK_SMOKE` override, CmdStan probe, palette, `theme_rethink()`, `save_fig()`, `tee()` |
| **1** | Load `listings.csv` via `fread`, or `simulate_listings()` if absent |
| **2** | Coerce types, parse price, filter, noise-by-*n* table, whole-host subsample, indices, binomial transform |
| **3** | Build `dat_gauss` and `dat_binom` lists for `ulam` |
| **4** | All model formulas as `alist()` — centered and non-centered variants |
| **5** | Fit `m1`, `m2`, `m3`, then `m0` (needs `m1`'s `σ`) |
| **6** | `precis(depth=2)` per model; Rhat / n_eff / divergence summary table |
| **7** | Correct pointwise log-likelihoods, built in R |
| **8** | WAIC + PSIS-LOO comparison; `loo_compare()`; `m2` scored separately |
| **9** | Posterior estimates on the rating scale; `30_shrunken_ratings.csv`; shrinkage-by-*n* table; RMSE vs. truth (synthetic only) |
| **10** | `β_rev` posterior summary across all four models |
| **11** | Five figures, plus the impossible-ratings statistic |
| **12** | Run summary and file listing |

Two helpers worth knowing:

- **`tee(x, file)`** prints a table to console *and* writes it to `outputs/`, so
  the terminal transcript and the saved artifacts never diverge.
- **`mu_gauss(post)`** reconstructs the linear predictor for whichever Gaussian
  model it is handed, dispatching on which parameters exist in the posterior
  (`a_host` present ⇒ `m3`; no `sigma` ⇒ `m0`). One function serves all three.

### A Stan typing detail

`inv_sqrt_n = 1/√n` is precomputed in R so the likelihood reads
`sigma * inv_sqrt_n` — a `real * vector` product, which is valid Stan. Writing
`sigma / sqrt(n_reviews)` with an integer array would not compile reliably, and
keeping `sqrt()` out of the model block avoids recomputing it every leapfrog
step.

---

## 7. Results

All figures below are from the 1200-listing / 644-host run with `seed = 1913`.

### 7.1 Sampling diagnostics

| model | params | max Rhat | # Rhat > 1.01 | min n_eff | **divergences** |
|---|---|---|---|---|---|
| `m0` unpooled | 1201 | 1.037 | 51 | 157 | **0** |
| `m1` Approach A | 2404 | **1.007** | **0** | **1066** | **0** |
| `m2` Approach B | 2403 | **1.008** | **0** | **1280** | **0** |
| `m3` Approach C | 3693 | 1.011 | 1 | 492 | **0** |

`m1` and `m2` clear all three targets (Rhat < 1.01, n_eff > 500, zero
divergences). `m3` has one parameter marginally over on each. **`m0` fails by
design** — see §5.2; its poor mixing is the no-pooling pathology, not a
configuration problem.

### 7.2 Model comparison (Gaussian family — same outcome)

| model | WAIC | dWAIC | **p_WAIC** | actual params |
|---|---|---|---|---|
| `m3` host + listing | **786.7** | 0.0 | **145.8** | 3693 |
| `m1` listing pooling | 1283.0 | 496.2 | **215.1** | 2404 |
| `m0` unpooled | 1639.7 | 853.0 | **507.9** | 1201 |

PSIS-LOO agrees decisively: elpd_diff −265.8 ± 20.7 (`m1`) and −646.1 ± 36.7
(`m0`) relative to `m3`.

**The `p_WAIC` column is the whole Chapter 13 lesson in one line.** `m0` has the
*fewest* actual parameters (1201) but the *largest* effective number (507.9);
`m3` has three times the parameters and a third of the effective count. Pooling
buys flexibility without paying for it in overfitting.

Model B, scored separately (**not comparable to the above**):
WAIC 3057.9, p_WAIC 265.4, LOOIC 3168.0.

### 7.3 Shrinkage by review count

Mean absolute shrinkage — how far pooling moved each listing:

| reviews | 1–2 | 3–5 | 6–10 | 11–50 | 50+ |
|---|---|---|---|---|---|
| A Gaussian | 0.729 | 0.448 | 0.255 | 0.151 | **0.059** |
| B Binomial | 0.705 | 0.393 | 0.211 | 0.121 | **0.045** |
| C Host+listing | 0.661 | 0.367 | 0.196 | 0.121 | **0.041** |

Monotone decay in all three models, spanning more than an order of magnitude —
exactly the specified behavior: heavy shrinkage below 5 reviews, near-zero above
50. All three approaches agree closely despite very different likelihoods, which
is reassuring.

### 7.4 The `β_rev` covariate effect

| model | scale | mean | 89% interval | P(β>0) |
|---|---|---|---|---|
| `m0` unpooled | rating pts | 0.179 | [0.140, 0.218] | 1.00 |
| `m1` Approach A | rating pts | **0.061** | [0.049, 0.074] | 1.00 |
| `m2` Approach B | logit | 0.214 | [0.167, 0.260] | 1.00 |
| `m3` Approach C | rating pts | **0.046** | [0.035, 0.057] | 1.00 |

More-reviewed listings do rate genuinely higher, and the sign is certain in every
model. **But the magnitude falls by a factor of ~4 once pooling is applied**
(0.179 → 0.046). Most of the raw association is a compositional artifact of noisy
low-*n* listings, not a real effect — which is itself a finding, and one the
unpooled model gets badly wrong.

*Caveat:* with one observation and one free intercept per listing, `β_rev` is
identified only through the adaptive prior, so it partly trades off against the
spread of `a_listing`.

### 7.5 The posterior predictive check finds a real defect

**19.4% (A) and 18.8% (C) of posterior predictive draws fall outside [1,5].**

The Gaussian likelihood has unbounded support; the rating scale stops at 5. Both
Gaussian models therefore put nearly a fifth of their predictive mass on ratings
that cannot exist. Approach B cannot do this by construction.

This is the strongest argument for Approach B — **and WAIC cannot see it**, since
WAIC only ranks A against C. It is a direct caveat on §7.2's ranking: the
best-scoring model in that table is not the best-specified model.

### 7.6 Ground truth (synthetic mode only)

When `listings.csv` is absent, `simulate_listings()` records the true latent
quality, and §9 scores every estimator against it by RMSE — the classic
demonstration that partial pooling beats both no-pooling and complete pooling.
To trigger it on demand, point `CFG$csv_path` at a nonexistent filename.

---

## 8. Output files

| file | contents |
|---|---|
| `00_noise_by_review_count.txt` | the motivating noise table (§2) |
| `10_precis_m{0,1,2,3}.txt` | full `precis(depth=2)` — every parameter |
| `11_diagnostics.txt` | Rhat / n_eff / divergence summary |
| `20_compare_gaussian.txt` | WAIC + LOO, Gaussian family |
| `21_loo_compare_gaussian.txt` | pairwise elpd differences with SEs |
| `22_score_binomial.txt` | Model B, scored separately |
| `30_shrunken_ratings.csv` | **the deliverable** — per-listing estimates |
| `31_shrinkage_by_n.txt` | shrinkage magnitude by review bucket |
| `40_beta_rev.txt` | `β_rev` posterior summaries |
| `fig1`–`fig5` `.png` | figures (§9) |

`30_shrunken_ratings.csv` columns: `id`, `host_id`, `number_of_reviews`,
`raw_rating`, `m0_unpooled`, `m1_approachA`, `m2_approachB`, `m3_approachC`
(plus `quality_true` in synthetic mode). **For production use, prefer
`m1_approachA` or `m2_approachB`** — see §10 on why `m3`'s decomposition is
weakly identified.

---

## 9. Reading the figures

**`fig1_shrinkage_scatter.png`** — raw rating (x) vs. posterior estimate (y),
one panel per model, colored by review count on a log sequential ramp. The
dashed 45° line is "no shrinkage"; the grey horizontal line is each model's own
`ā` (Model B's mapped back from logit). **Pale points (few reviews) collapse
onto the grey line regardless of their raw rating; dark points (many reviews)
sit on the dashed line.** The single-review listing rated 1.0 is revised to
≈4.4.

**`fig2_shrinkage_vs_n.png`** — shrinkage vs. review count (log x), with a
10–90% envelope. A clean funnel collapsing to zero. Color is *not* mapped to
shrinkage here: the y-axis already encodes it, and one +3.4 outlier flattened
the ramp to near-white. The color budget went to the envelope instead, which
adds information the points don't carry.

**`fig3_posterior_predictive.png`** — observed density (black) against 60
predictive replicates (blue), with the hard ceiling at 5 marked in red. The
replicates visibly cross it. This is §7.5 made visual.

**`fig4_beta_rev_posterior.png`** — `β_rev` posteriors, **faceted by scale**.
Model B's slope is in logit units and A's/C's in rating points; plotting them on
one shared axis would be meaningless, so they get separate panels. Densities are
direct-labeled A/B/C at their modes as well as color-coded.

**`fig5_variance_components.png`** — `σ_host` vs. `σ_listing` in Model C. **Read
the subtitle before drawing conclusions** — see §10.

Figures use a colorblind-safe palette: a single-hue blue sequential ramp for
magnitude and a three-slot categorical set validated for all-pairs separation.
No dual axes anywhere.

---

## 10. Known limitations

**Model C's variance split is weakly identified.** 490 of 644 hosts own exactly
one listing. For those hosts `a_host` and `a_listing` are **perfectly
confounded** — only their sum is identified, and the split between them is
decided by the priors, not the data. So `σ_listing ≈ 0.016` versus
`σ_host ≈ 0.169` is **not** evidence that listings don't vary. Model C's WAIC
win is real (it predicts better); the "variation lives at the host level"
interpretation is not supported. Only the 154 multi-listing hosts carry real
information here.

**Both Gaussian models violate the support of the data** (§7.5). If bounded
predictions matter for your use, use Approach B.

**High p_WAIC and Pareto-k warnings are structural, not bugs.** With one
observation and one free intercept per listing, every point has a nearly
dedicated parameter and is influential when dropped. `bad_k` counts are large by
construction. Treat the WAIC/LOO gaps as suggestive of ordering, not as precise
effect sizes — the gaps here are large enough (>10 SE) that the ordering is safe
even so.

**1200 of 70,711 listings.** The subsample is random over hosts and unbiased,
but it is a subsample. Raise `CFG$n_listings` to check stability.

**`σ` is assumed constant across listings.** The model allows each listing its
own mean but shares one per-review noise scale. A listing with genuinely polarized
reviews is not distinguished from a consistent one.

**`m0`'s `σ` is borrowed from `m1`** (§5.2). This is the right call for isolating
the pooling effect, but it does mean `m0` is not a fully independent model — it
is handed one number by the model it is being compared against.

---

## 11. Extending the script

**Add a covariate** (e.g. price): add it to `dat_gauss`, then add a slope term
to `mu` and a prior. `log_price` is already parsed as `price_clean` — note the
31% missingness discussed in §3.

**Non-centered off**: set `CFG$noncentered <- FALSE` to see the centered
parameterization diverge, which is instructive.

**Full dataset**: raise `CFG$n_listings`. Expect long runs and large posteriors;
consider dropping `outputs/10_precis_*.txt` writes, which become enormous.

**A better Model B**: the current version discretizes to integer success counts.
A Beta-Binomial or an ordered-logit over the five star levels would respect the
bound *and* handle over-dispersion.

**If you add a model**, remember §5.1 — do not trust `ulam`'s `log_lik` if your
likelihood's scale contains a data vector. Write the log-likelihood in R and
calibrate against a constant-scale control, as `ll_gauss_scaled()` does.
