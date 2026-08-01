# Findings

What was not obvious in advance, and what the numbers turned out to be. The
argument itself is in [report.tex](report.tex); this file is the working notes
behind it.

---

## 1. The comparison works, and the sensitivity check earns its place

| model | parameters | WAIC | dWAIC | dSE | **pWAIC** |
|---|---|---|---|---|---|
| `partial` | **202** | **485.1** | 0.0 | — | **37.5** |
| `no_pooling_flat` — `Normal(0,5)` | 200 | 508.3 | 23.2 | 14.7 | 73.3 |
| `complete_pooling` | 1 | 547.5 | 62.4 | 22.5 | 2.8 |
| `no_pooling` — `Normal(0,1.5)` | 200 | 690.2 | 205.1 | 18.7 | 112.1 |

The multilevel model has the most actual parameters and nearly the fewest
effective ones, which is the chapter's central claim. Complete pooling is the
only model whose effective count (2.8) is close to its actual one (1) — with a
single parameter there is nothing to overfit — and it still loses by 62 WAIC,
because `sigma = 0.62` is small but not zero.

**The flat prior comes out backwards from the naive expectation.**
`no_pooling_flat` has the *wider* prior and the *lower* pWAIC of the two
no-pooling models. The reason is that `Normal(0,1.5)` is not merely tight, it is
tight *in the wrong place*: it is centred on `p = 0.5` while the data sit at
`p = 0.947`. Prior and likelihood fight over every listing, and that conflict
inflates the variance of the log-likelihood — which is what pWAIC measures.

This is why the sensitivity fit is in the repository rather than being dropped
once it agreed. Without it, someone could reasonably object that the multilevel
model only wins because the fixed prior was badly chosen. It wins against both.

## 2. The centered parameterization fails quietly

| | divergent | min E-BFMI | max R̂ | min n_eff |
|---|---|---|---|---|
| centered | 0 | **0.216** | 1.023 | **150** |
| non-centered | 0 | **0.736** | 1.001 | **714** |

**Neither version produced a single divergent transition.** The warning most
often associated with the funnel never fired, so anyone checking only for
divergences would have concluded the centered model was fine.

It is not fine. The energy diagnostic E-BFMI came in at 0.216 against a 0.3
threshold, and the centered model extracted 150 effectively independent draws of
`sigma` where the non-centered model got 714 from the same 2,000 samples. Since
the whole extension in §9 of the report rests on the `sigma` posterior, a
fivefold difference in its precision is not cosmetic.

Both are kept in `src/03-models.R` and both are fitted, because the failure is
the lesson.

## 3. `saveRDS` on a cmdstan-backed `ulam` fit silently loses the draws

Under cmdstan the posterior lives in temporary CSV files that are deleted when
the R session ends. A saved `ulam` object therefore reloads without its samples,
and `extract.samples()` on it fails with a missing-file error that names a path
in `Temp/` and explains nothing.

The pipeline saves `extract.samples(fit)` instead of the fit object. See the
comment in `src/04-fit.R`.

## 4. PSIS and WAIC give the wrong answer for the σ grid

This is the most interesting thing in the project.

The extension pins `sigma` at twelve fixed values and asks which one predicts
best out of sample. Three criteria disagree:

| criterion | peak σ |
|---|---|
| WAIC | 1.161 |
| PSIS | 0.696 |
| **direct leave-one-out** | **0.539** |
| posterior mode of the `dexp(1)` model | **0.623** |
| 89% interval | [0.456, 0.786] |

The direct computation agrees with the posterior; PSIS and WAIC both drift
toward *larger* σ, i.e. they recommend *less* pooling than is good for
prediction.

**Why.** Every listing contributes exactly one binomial observation, so dropping
listing `i` removes *all* information about `α_i`. PSIS estimates the effect of
dropping a point by reweighting a posterior that was fitted *with* that point;
when the point is the only thing holding a parameter in place, that reweighting
is being asked to do far too much. The Pareto-k column in
`tables/06-sigma-grid.txt` shows it directly — the count of listings with
`k > 0.7` climbs from 2 at σ = 0.15 to 97 at σ = 2.5, and is already 27 at the σ
PSIS itself nominates.

**The fix.** The same fact that breaks PSIS makes the honest answer easy. With
`α_i` unconstrained by the remaining data, the leave-one-out prediction is just
the prediction for a brand new listing:

```
p(P_i | y_-i, σ) = ∫ Binomial(P_i | N_i, logistic(a)) Normal(a | ᾱ, σ) da
```

averaged over the posterior of `ᾱ`. No importance sampling anywhere. It handles
`α_i` exactly; its one approximation is reusing the full-data posterior of `ᾱ`
rather than refitting without listing `i`, which moves `ᾱ` by a fraction of its
own posterior SD when one listing in two hundred is removed — a far milder
approximation, and not one that biases σ.

Recomputing with independent draws and a tenth of the sample gives the same peak
(`loco_check` in the same table), so this is not Monte Carlo noise.

**What to claim.** Not that the two numbers are identical. The peak is flat —
elpd differs by 0.23 between σ = 0.539 and σ = 0.696 — and the posterior mode
0.623 falls between those two adjacent grid points. The defensible statement is
that the cross-validation optimum and the posterior mode land in the same place,
comfortably inside the 89% interval, while both ends of the grid are decisively
worse.

## 5. Mean shrinkage distance is not monotone, and that is not a bug

| reviews | mean shift | spread before | spread after | shrink factor |
|---|---|---|---|---|
| 1–2 | 0.327 | 0.314 | 0.032 | **0.899** |
| 3–5 | 0.396 | 0.445 | 0.059 | 0.869 |
| 6–10 | 0.214 | 0.257 | 0.045 | 0.826 |
| 11–50 | 0.111 | 0.168 | 0.063 | 0.625 |
| 50+ | 0.047 | 0.116 | 0.070 | **0.395** |

The 1–2 bin moves *less* (0.327) than the 3–5 bin (0.396), which looks wrong. It
isn't. A one-review listing is nearly always rated exactly 5.0, and 5.0 is only
about 0.2 rating points above the population mean of 4.79 — so there is almost
no distance to travel even though roughly 90% of it is taken away.

A first attempt measured this as the mean per-listing fraction of the gap
closed. That was unstable and produced values above 1, because the denominator
`p_emp − p̄` is near zero for exactly those low-`N` listings. `shrink_factor`
compares aggregate spread before and after — a ratio of means rather than a mean
of ratios — and decays cleanly from 0.90 to 0.40.

## 6. The simulation says the ranking depends on σ, and WAIC cannot see that

Every comparison above ranks models without ever seeing the truth. `src/10-simulation.R`
simulates 60 listings from a *known* `a_bar` and `sigma`, so error can be
measured directly. Mean absolute error against the true probability:

| scenario | reviews | complete | no pooling | partial |
|---|---|---|---|---|
| fitted, σ = 0.62 | 2 | **0.0296** | 0.1122 | 0.0320 |
| | 5 | 0.0310 | 0.0718 | **0.0289** |
| | 15 | **0.0166** | 0.0447 | 0.0167 |
| | 50 | 0.0269 | 0.0318 | **0.0198** |
| | **overall** | 0.0260 | 0.0651 | **0.0243** |
| book, σ = 1.50 | 2 | 0.1913 | 0.2097 | **0.1179** |
| | 5 | 0.2021 | 0.1234 | **0.1081** |
| | 15 | 0.1117 | 0.0725 | **0.0592** |
| | 50 | 0.1758 | 0.0526 | **0.0525** |
| | **overall** | 0.1702 | 0.1145 | **0.0844** |

**Two scenarios, because one would have misled.** Running only the fitted
parameters makes complete pooling look nearly as good as partial pooling
(0.0260 against 0.0243), and it actually *wins* at `N = 2`. That is not an
error — with `sigma = 0.62` the listings genuinely are nearly identical, so
throwing away all between-listing information costs little.

Switch to the book's `sigma = 1.50` and complete pooling becomes the *worst*
estimator (0.1702), while partial pooling wins every single group. The ordering
of the two extremes flips entirely on a parameter the real data cannot hand you
for free.

So the honest claim is not "partial pooling always wins by a lot". It is that
each extreme is badly wrong in one of the two regimes, and partial pooling is
never much worse than whichever extreme happens to be right. That is the
underfitting/overfitting trade-off of section 13.2, and it is invisible to
WAIC on a single data set.

---

## Modelling assumptions worth stating

- `review_scores_rating` is a mean of 1–5 star ratings, not a proportion of
  positives. Reading it as one is a choice.
- `number_of_reviews` counts all reviews, while the rating is computed only from
  reviews that carried a score, so `N` slightly overstates the trials.
- The sample is 200 listings, 40 per review-count band. Stratified, not random:
  a random sample is swamped by the middle of the distribution and leaves too
  few one-review listings, which are the ones the model exists to handle.
