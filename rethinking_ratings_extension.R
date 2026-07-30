# =============================================================================
#  rethinking_ratings_extension.R
#
#  Bayesian multilevel rating models with partial pooling & shrinkage.
#  An extension of Statistical Rethinking (2nd ed.), Chapter 13.
#
#  Goal: shrunken, un-overfitted baseline ratings for Airbnb listings, where a
#  listing's credibility scales with how many reviews it actually has.
#
#  Models
#    m0  Unpooled fixed effects   : a[listing] ~ Normal(4, 1)      (no pooling)
#    m1  Approach A               : Gaussian, sd = sigma / sqrt(n_i)
#    m2  Approach B               : Binomial success-count (Reed-Frog analogue)
#    m3  Approach C               : Cross-classified host + listing intercepts
#  Plus a no-pooling benchmark (the raw observed rating itself).
#
#  Run:  Rscript rethinking_ratings_extension.R
#  Fast smoke test:  RETHINK_SMOKE=1 Rscript rethinking_ratings_extension.R
#
# -----------------------------------------------------------------------------
#  FIVE THINGS TO KNOW BEFORE READING THE RESULTS
#
#  (1) ONE ROW PER LISTING. In listings.csv, `id` is unique, so every listing
#      cluster holds exactly ONE observation. This is not a defect -- it is
#      precisely the Reed-Frog tadpole structure of Ch. 13 (one row per tank).
#      Partial pooling still works, because the n_i-weighted likelihood tells
#      the model how *precise* each single observation is. `review_scores_rating`
#      is already an average over n_i reviews; number_of_reviews is its n.
#
#  (2) ulam's AUTO-GENERATED log_lik IS WRONG FOR MODELS 1 AND 3.  <-- important
#      When the likelihood's scale contains a data vector, ulam emits
#          log_lik[i] = normal_lpdf( y[i] | mu[i] , sigma * inv_sqrt_n );
#      leaving the vector UNINDEXED, so each point's log-lik is silently summed
#      over all N scale values. Measured on a 60-row test case: ulam reported
#      WAIC 177366.3 where the correct value is 42.188. (A control model with a
#      constant sigma matched a manual computation to 9e-7, confirming the fault
#      is the unindexed vector and not the manual method.)
#      => rethinking::WAIC/PSIS/compare() CANNOT be used on m1/m3 here. This
#      script builds pointwise log-lik matrices in R and uses loo:: instead.
#      See ll_gauss_scaled() / ll_binom() and section 8.
#
#  (3) MODEL 2 IS NOT WAIC-COMPARABLE TO MODELS 1/3. Its outcome is a count
#      S_i out of n_i trials; theirs is the continuous rating y_i. Information
#      criteria are only comparable across models of the SAME outcome, so m2 is
#      scored on its own and compared to the others on the rating scale instead.
#
#  (4) p_waic / Pareto-k WARNINGS ARE EXPECTED, NOT A BUG. With one observation
#      per listing and one free intercept per listing, each point has its own
#      near-dedicated parameter, so it is always influential when dropped. Read
#      the WAIC/LOO gaps as suggestive, and lean on the shrinkage plots and (on
#      synthetic data) the RMSE-vs-truth table in section 9 for the real verdict.
#
#  (5) APPROACHES A AND C PREDICT RATINGS ABOVE 5. Their Gaussian likelihood has
#      unbounded support while the real scale stops at 5, so the posterior
#      predictive puts mass on impossible values: on the 1200-listing run,
#      19.4% (A) and 18.8% (C) of predictive draws fell outside [1,5]. See
#      section 11 and fig3. Approach B cannot do this -- a Binomial on n_i
#      trials respects the bound by construction. That is the strongest reason
#      to keep B even though its WAIC is not comparable to A's and C's, and it
#      is a caveat on the WAIC ranking, which only scores A and C against each
#      other and cannot see that both violate the support of the data.
# =============================================================================

suppressPackageStartupMessages({
  library(tidyverse)
  library(rethinking)
  library(loo)
})

# ---- 0. Configuration -------------------------------------------------------
CFG <- list(
  csv_path    = "listings.csv",
  out_dir     = "outputs",
  seed        = 1913,
  n_listings  = 1200,   # subsample budget: 70k listings => 70k intercepts, too slow
  chains      = 4,
  cores       = 4,
  iter        = 2000,   # rethinking uses warmup = iter/2
  adapt_delta = 0.95,
  noncentered = TRUE,   # TRUE => non-centered parameterisation (see section 6)
  use_cmdstan = TRUE
)

if (nzchar(Sys.getenv("RETHINK_SMOKE"))) {
  message("--- SMOKE TEST MODE: tiny subsample, short chains ---")
  CFG$n_listings <- 150; CFG$chains <- 2; CFG$cores <- 2; CFG$iter <- 600
}

dir.create(CFG$out_dir, showWarnings = FALSE, recursive = TRUE)
set.seed(CFG$seed)

if (CFG$use_cmdstan) {
  ok <- tryCatch({ cmdstanr::cmdstan_version(); TRUE }, error = function(e) FALSE)
  if (!ok) { message("cmdstan not found; falling back to rstan."); CFG$use_cmdstan <- FALSE }
}

# Palette: pre-validated 3-slot categorical subset + single-hue sequential ramp.
PAL <- list(
  surface = "#fcfcfb", ink = "#0b0b0b", ink2 = "#52514e",
  muted   = "#898781", grid = "#e1e0d9", axis = "#c3c2b7",
  s1 = "#2a78d6", s2 = "#eb6834", s3 = "#1baf7a",              # categorical
  seq = c("#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"), # sequential blue
  div_lo = "#2a78d6", div_mid = "#f0efec", div_hi = "#e34948"     # diverging
)

theme_rethink <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.background  = element_rect(fill = PAL$surface, colour = NA),
      panel.background = element_rect(fill = PAL$surface, colour = NA),
      panel.grid.major = element_line(colour = PAL$grid, linewidth = 0.3),
      panel.grid.minor = element_blank(),
      panel.spacing    = unit(1.1, "lines"),
      axis.line   = element_line(colour = PAL$axis, linewidth = 0.4),
      axis.ticks  = element_line(colour = PAL$axis, linewidth = 0.3),
      axis.text   = element_text(colour = PAL$muted),
      axis.title  = element_text(colour = PAL$ink2),
      strip.text  = element_text(colour = PAL$ink, face = "bold", hjust = 0),
      plot.title    = element_text(colour = PAL$ink,  face = "bold", size = rel(1.2)),
      plot.subtitle = element_text(colour = PAL$ink2, size = rel(0.92)),
      plot.caption  = element_text(colour = PAL$muted, size = rel(0.78), hjust = 0),
      legend.title  = element_text(colour = PAL$ink2),
      legend.text   = element_text(colour = PAL$ink2)
    )
}

save_fig <- function(p, file, w = 9, h = 5.4) {
  ggsave(file.path(CFG$out_dir, file), p, width = w, height = h,
         dpi = 200, bg = PAL$surface)
  message("  wrote ", file.path(CFG$out_dir, file))
}

tee <- function(x, file) {                     # print to console AND to a file
  cat("\n"); print(x)
  capture.output(print(x), file = file.path(CFG$out_dir, file))
  invisible(x)
}

# ---- 1. Data: real listings.csv, else a synthetic stand-in ------------------
# The synthetic generator mirrors the real Airbnb column names/scales so the
# script runs out of the box, and it records the TRUE latent quality so
# section 9 can score pooling against no-pooling on known ground truth.
IS_SYNTHETIC <- !file.exists(CFG$csv_path)

read_listings <- function(path) {
  cols <- c("id", "host_id", "review_scores_rating", "number_of_reviews", "price")
  if (requireNamespace("data.table", quietly = TRUE)) {
    # 222 MB file -- fread is far faster than readr here
    as_tibble(data.table::fread(path, select = cols, showProgress = FALSE))
  } else {
    read_csv(path, col_select = all_of(cols), show_col_types = FALSE,
             progress = FALSE)
  }
}

simulate_listings <- function(n_hosts = 900, seed = 1913) {
  set.seed(seed)
  sigma_host_true    <- 0.22
  sigma_listing_true <- 0.30
  a_bar_true         <- 4.72
  b_rev_true         <- 0.06
  sigma_true         <- 0.90   # per-REVIEW sd; a mean of n reviews has sd/sqrt(n)

  n_per_host <- 1 + rbinom(n_hosts, 6, 0.16)              # mostly 1, some many
  host_id    <- rep(seq_len(n_hosts) + 100000L, n_per_host)
  n_list     <- length(host_id)
  a_host     <- rnorm(n_hosts, 0, sigma_host_true)[rep(seq_len(n_hosts), n_per_host)]
  a_listing  <- rnorm(n_list, 0, sigma_listing_true)

  # heavy skew toward tiny review counts, exactly as in the real data
  n_reviews  <- 1L + rnbinom(n_list, mu = 22, size = 0.45)
  log_rev_c  <- log(n_reviews) - mean(log(n_reviews))

  quality_true <- a_bar_true + a_host + a_listing + b_rev_true * log_rev_c
  # observed rating = mean of n_reviews noisy reviews, then clipped to [0, 5]
  rating <- rnorm(n_list, quality_true, sigma_true / sqrt(n_reviews))
  rating <- round(pmin(pmax(rating, 0), 5), 2)

  tibble(
    id                   = seq_len(n_list) + 500000L,
    host_id              = host_id,
    review_scores_rating = rating,
    number_of_reviews    = n_reviews,
    price                = paste0("$", format(round(exp(rnorm(n_list, 5.1, 0.6)), 2),
                                              nsmall = 2, trim = TRUE)),
    quality_true         = quality_true
  )
}

if (IS_SYNTHETIC) {
  message("No ", CFG$csv_path, " found -- generating synthetic Airbnb-shaped data.")
  raw <- simulate_listings()
} else {
  message("Reading ", CFG$csv_path, " ...")
  raw <- read_listings(CFG$csv_path)
}

# ---- 2. Preprocessing -------------------------------------------------------
# NOTE ON price: we parse it as specified, but we deliberately do NOT filter on
# it. In the real file 22,041 of 70,711 otherwise-usable rows (31%) have a
# missing price, and price appears in none of the three models -- dropping those
# rows would throw away a third of the rating data to no purpose. price is
# retained purely as a descriptive column.
df <- raw %>%
  mutate(
    review_scores_rating = suppressWarnings(as.numeric(review_scores_rating)),
    number_of_reviews    = suppressWarnings(as.integer(number_of_reviews)),
    price_clean          = readr::parse_number(as.character(price))
  ) %>%
  filter(!is.na(review_scores_rating), !is.na(number_of_reviews),
         number_of_reviews > 0)

cat("\n==== Raw data ====\n")
cat("usable listings:", nrow(df), " hosts:", n_distinct(df$host_id), "\n")
cat("rating range:", paste(range(df$review_scores_rating), collapse = " to "), "\n")

# The motivating fact: small-n ratings are wildly noisier than large-n ratings.
noise_by_n <- df %>%
  mutate(bucket = cut(number_of_reviews, c(0, 2, 5, 10, 50, Inf),
                      labels = c("1-2", "3-5", "6-10", "11-50", "50+"))) %>%
  summarise(listings = n(), mean_rating = mean(review_scores_rating),
            sd_rating = sd(review_scores_rating), .by = bucket) %>%
  arrange(bucket)
tee(as.data.frame(noise_by_n), "00_noise_by_review_count.txt")

# Subsample WHOLE HOSTS, not random rows. Sampling rows would scatter hosts and
# leave almost no host with >1 listing, making sigma_host in Model 3
# unidentifiable. Keeping intact host clusters preserves the nesting.
host_sizes <- df %>% count(host_id, name = "n_list")
keep_hosts <- host_sizes %>%
  slice_sample(prop = 1) %>%                       # random host order
  mutate(cum = cumsum(n_list)) %>%
  filter(cum <= CFG$n_listings) %>%
  pull(host_id)
df <- df %>% filter(host_id %in% keep_hosts)

# Contiguous 1..N integer indices for ulam
df <- df %>%
  mutate(listing_idx = as.integer(as.factor(id)),
         host_idx    = as.integer(as.factor(host_id)),
         log_rev     = log(number_of_reviews) - mean(log(number_of_reviews)))

N_listing <- max(df$listing_idx)
N_host    <- max(df$host_idx)
cat("\n==== Analysis subsample ====\n")
cat("listings:", nrow(df), " hosts:", N_host,
    " hosts with >1 listing:", sum(table(df$host_idx) > 1), "\n")
cat("listings per host:\n"); print(table(pmin(as.vector(table(df$host_idx)), 5)))
cat("review-count quantiles:\n")
print(quantile(df$number_of_reviews, c(0, .25, .5, .75, .95, 1)))

# Approach B transform: rating on [1,5] -> success units out of n_reviews trials.
# 3 of 70,711 real rows sit below 1.0; they are clamped (the transform assumes
# a [1,5] support). Clamping 3 rows is preferable to dropping them silently.
df <- df %>%
  mutate(
    rating_clamped = pmin(pmax(review_scores_rating, 1), 5),
    p_raw          = (rating_clamped - 1) / 4,
    successes      = as.integer(round(p_raw * number_of_reviews))
  )
cat("\nrows clamped for the binomial transform:",
    sum(df$review_scores_rating < 1 | df$review_scores_rating > 5), "\n")

# ---- 3. Data lists for ulam -------------------------------------------------
# inv_sqrt_n is precomputed so the likelihood scale is `sigma * inv_sqrt_n`,
# a real * vector product -- valid Stan, and it keeps sqrt() out of the model.
dat_gauss <- list(
  rating      = df$review_scores_rating,
  inv_sqrt_n  = 1 / sqrt(as.numeric(df$number_of_reviews)),
  log_rev     = df$log_rev,
  listing_idx = df$listing_idx,
  host_idx    = df$host_idx,
  N_listing   = N_listing,
  N_host      = N_host
)
dat_binom <- list(
  successes   = df$successes,
  n_reviews   = df$number_of_reviews,
  log_rev     = df$log_rev,
  listing_idx = df$listing_idx,
  N_listing   = N_listing
)

# ---- 4. Model formulas ------------------------------------------------------
# Centred and non-centred variants of each. The non-centred forms reparameterise
# a = a_bar + z * sigma with z ~ Normal(0,1), which removes the funnel geometry
# that produces divergent transitions when sigma_listing is small. `transpars`
# keeps the reconstructed intercepts in the posterior so downstream code is
# identical either way.

# --- m0: UNPOOLED fixed effects. Fixed prior, no adaptive sigma_listing:
#         each listing is estimated in isolation. The no-pooling benchmark.
#
#   WHY obs_sd IS SUPPLIED AS DATA RATHER THAN ESTIMATED. Letting this model
#   estimate its own sigma makes it NON-IDENTIFIED. With one row and one free
#   intercept per listing the likelihood is saturated -- a_listing[i] can sit
#   exactly on y_i - b_rev*log_rev_i for every i -- so the density is unbounded
#   as sigma -> 0 and the sampler chases it down a funnel. Measured on a
#   150-listing / 2-chain / 600-iter trial of this script: 18/600 divergent
#   transitions, E-BFMI < 0.3 on both chains, sigma Rhat 2.14 and n_eff 2.7,
#   and a WAIC of -182 that "beat" every pooled model. That number is an
#   artifact of the collapse, not evidence.
#
#   This degeneracy is itself the lesson: nothing stops no-pooling from fitting
#   the noise exactly. The adaptive prior in m1/m3 is what holds sigma up. So
#   we pin sigma at m1's posterior mean, giving m0 an observation model
#   identical to m1's -- the comparison then isolates pooling and nothing else.
f_m0 <- alist(
  rating ~ dnorm(mu, obs_sd),
  mu <- a_listing[listing_idx] + b_rev * log_rev,
  a_listing[listing_idx] ~ dnorm(4.0, 1.0),
  b_rev ~ dnorm(0, 0.5)
)

# --- m1: APPROACH A. Gaussian, observation sd scaled by 1/sqrt(n_i).
f_m1_centered <- alist(
  rating ~ dnorm(mu, sigma * inv_sqrt_n),
  mu <- a_listing[listing_idx] + b_rev * log_rev,
  a_listing[listing_idx] ~ dnorm(a_bar, sigma_listing),
  a_bar ~ dnorm(4.0, 1.0),
  b_rev ~ dnorm(0, 0.5),
  sigma_listing ~ dexp(1),
  sigma ~ dexp(1)
)
f_m1_noncentered <- alist(
  rating ~ dnorm(mu, sigma * inv_sqrt_n),
  mu <- a_listing[listing_idx] + b_rev * log_rev,
  transpars > vector[N_listing]:a_listing <<- a_bar + z_listing * sigma_listing,
  vector[N_listing]:z_listing ~ dnorm(0, 1),
  a_bar ~ dnorm(4.0, 1.0),
  b_rev ~ dnorm(0, 0.5),
  sigma_listing ~ dexp(1),
  sigma ~ dexp(1)
)

# --- m2: APPROACH B. Binomial success counts (the Reed-Frog survival analogue).
f_m2_centered <- alist(
  successes ~ dbinom(n_reviews, p),
  logit(p) <- a_listing[listing_idx] + b_rev * log_rev,
  a_listing[listing_idx] ~ dnorm(a_bar, sigma_listing),
  a_bar ~ dnorm(0, 1),
  b_rev ~ dnorm(0, 0.5),
  sigma_listing ~ dexp(1)
)
f_m2_noncentered <- alist(
  successes ~ dbinom(n_reviews, p),
  logit(p) <- a_listing[listing_idx] + b_rev * log_rev,
  transpars > vector[N_listing]:a_listing <<- a_bar + z_listing * sigma_listing,
  vector[N_listing]:z_listing ~ dnorm(0, 1),
  a_bar ~ dnorm(0, 1),
  b_rev ~ dnorm(0, 0.5),
  sigma_listing ~ dexp(1)
)

# --- m3: APPROACH C. Cross-classified host + listing intercepts.
#     Both offsets are centred on zero with a_bar carrying the global level;
#     a_host and a_listing are separated only through their two variance
#     components, so the non-centred form is strongly preferred here.
f_m3_centered <- alist(
  rating ~ dnorm(mu, sigma * inv_sqrt_n),
  mu <- a_bar + a_host[host_idx] + a_listing[listing_idx] + b_rev * log_rev,
  a_host[host_idx] ~ dnorm(0, sigma_host),
  a_listing[listing_idx] ~ dnorm(0, sigma_listing),
  a_bar ~ dnorm(4.0, 1.0),
  b_rev ~ dnorm(0, 0.5),
  sigma_host ~ dexp(1),
  sigma_listing ~ dexp(1),
  sigma ~ dexp(1)
)
f_m3_noncentered <- alist(
  rating ~ dnorm(mu, sigma * inv_sqrt_n),
  mu <- a_bar + a_host[host_idx] + a_listing[listing_idx] + b_rev * log_rev,
  transpars > vector[N_host]:a_host <<- z_host * sigma_host,
  transpars > vector[N_listing]:a_listing <<- z_listing * sigma_listing,
  vector[N_host]:z_host ~ dnorm(0, 1),
  vector[N_listing]:z_listing ~ dnorm(0, 1),
  a_bar ~ dnorm(4.0, 1.0),
  b_rev ~ dnorm(0, 0.5),
  sigma_host ~ dexp(1),
  sigma_listing ~ dexp(1),
  sigma ~ dexp(1)
)

pick <- function(centered, noncentered) if (CFG$noncentered) noncentered else centered

# ---- 5. Fit -----------------------------------------------------------------
fit_model <- function(flist, dat, label) {
  message("\n>>> fitting ", label, " ...")
  t0 <- proc.time()[["elapsed"]]
  f <- ulam(flist, data = dat,
            chains = CFG$chains, cores = CFG$cores, iter = CFG$iter,
            cmdstan = CFG$use_cmdstan, log_lik = FALSE, refresh = 0,
            control = list(adapt_delta = CFG$adapt_delta, max_treedepth = 12))
  message("    ", label, " done in ", round(proc.time()[["elapsed"]] - t0, 1), "s")
  f
}
# log_lik = FALSE on purpose: ulam's generated log_lik is wrong for the
# vector-scale models (see header note 2). We build it correctly in section 7.

m1 <- fit_model(pick(f_m1_centered, f_m1_noncentered), dat_gauss, "m1  Approach A")
m2 <- fit_model(pick(f_m2_centered, f_m2_noncentered), dat_binom, "m2  Approach B")
m3 <- fit_model(pick(f_m3_centered, f_m3_noncentered), dat_gauss, "m3  Approach C")

# m0 is fitted last because it borrows m1's sigma (see the note on f_m0).
sigma_hat    <- mean(extract.samples(m1)$sigma)
obs_sd_fixed <- sigma_hat * dat_gauss$inv_sqrt_n
cat("\nsigma held fixed in the unpooled benchmark at m1's posterior mean:",
    round(sigma_hat, 4), "\n")
dat_unpooled <- c(dat_gauss, list(obs_sd = obs_sd_fixed))
m0 <- fit_model(f_m0, dat_unpooled, "m0  unpooled fixed effects")

models <- list(m0 = m0, m1 = m1, m2 = m2, m3 = m3)

# ---- 6. Diagnostics: Rhat, n_eff, divergences -------------------------------
diag_one <- function(fit, label) {
  pr <- precis(fit, depth = 2)
  d  <- as.data.frame(pr)
  capture.output(print(pr), file = file.path(CFG$out_dir,
                  paste0("10_precis_", label, ".txt")))
  # precis() names these differently across versions/backends: Rhat4 / rhat,
  # n_eff / ess_bulk. Match case-insensitively rather than assuming.
  rhat_col <- grep("rhat", names(d), value = TRUE, ignore.case = TRUE)[1]
  ess_col  <- grep("n_eff|ess", names(d), value = TRUE, ignore.case = TRUE)[1]
  rh  <- if (!is.na(rhat_col)) d[[rhat_col]] else NA_real_
  ess <- if (!is.na(ess_col))  d[[ess_col]]  else NA_real_
  # divergent transitions, from whichever backend produced the fit
  ndiv <- tryCatch({
    cs <- attr(fit, "cstanfit")
    if (!is.null(cs)) sum(cs$diagnostic_summary(quiet = TRUE)$num_divergent)
    else sum(rstan::get_divergent_iterations(attr(fit, "stanfit")))
  }, error = function(e) NA_integer_)
  tibble(model = label, n_par = nrow(d),
         max_Rhat = max(rh, na.rm = TRUE),
         n_Rhat_gt_1.01 = sum(rh > 1.01, na.rm = TRUE),
         min_n_eff = min(ess, na.rm = TRUE),
         n_ess_lt_500 = sum(ess < 500, na.rm = TRUE),
         divergences = ndiv)
}

cat("\n\n############ DIAGNOSTICS ############\n")
diags <- imap_dfr(models, ~ diag_one(.x, .y))
tee(as.data.frame(diags), "11_diagnostics.txt")
cat("\nTargets: max_Rhat < 1.01, min_n_eff > 500, divergences == 0.\n")
if (any(diags$divergences > 0, na.rm = TRUE) && !CFG$noncentered)
  cat("!! Divergences with the centred parameterisation.",
      "Set CFG$noncentered <- TRUE and refit.\n")
if (any(diags$divergences > 0, na.rm = TRUE) && CFG$noncentered)
  cat("!! Divergences persist under non-centring;",
      "raise CFG$adapt_delta toward 0.99.\n")

# Top-level parameters only -- the readable summary.
cat("\n---- population-level parameters ----\n")
for (nm in names(models)) {
  cat("\n##", nm, "\n")
  print(precis(models[[nm]], depth = 1,
               pars = intersect(c("a_bar", "b_rev", "sigma_listing",
                                  "sigma_host", "sigma"),
                                names(extract.samples(models[[nm]])))))
}

cat("\n---- two diagnostics that are expected, not failures ----\n")
cat("* m0's b_rev mixes poorly (low n_eff, Rhat above target). With a fixed,\n",
    "  non-adaptive prior, the free per-listing intercepts soak up any slope,\n",
    "  so b_rev is barely identified. This is a property of no-pooling, and it\n",
    "  is part of what the comparison is meant to expose.\n", sep = "")
cat("* In m3, sigma_listing often piles up near zero while sigma_host does not.\n",
    "  ", sum(table(df$host_idx) == 1), " of ", N_host, " hosts own exactly one ",
    "listing; for those, a_host and\n  a_listing are perfectly confounded and ",
    "only their sum is identified. Do\n  not read this as 'listings do not vary'",
    " -- see fig5.\n", sep = "")

# ---- 7. Correct pointwise log-likelihoods -----------------------------------
# Built by hand because ulam's generated log_lik is broken for m0/m1/m3
# (header note 2). Verified against a constant-sigma control model, where this
# exact code reproduced ulam's own log_lik to within 9.2e-07.
y   <- df$review_scores_rating
Y   <- function(nr) matrix(y, nr, length(y), byrow = TRUE)
invs <- dat_gauss$inv_sqrt_n

mu_gauss <- function(post) {                     # [samples x N] linear predictor
  if (!is.null(post$a_bar) && !is.null(post$a_host)) {         # m3
    as.vector(post$a_bar) + post$a_host[, df$host_idx] +
      post$a_listing[, df$listing_idx] +
      outer(as.vector(post$b_rev), df$log_rev)
  } else {                                                     # m0, m1
    post$a_listing[, df$listing_idx] + outer(as.vector(post$b_rev), df$log_rev)
  }
}

ll_gauss_scaled <- function(fit) {
  post <- extract.samples(fit)
  mu   <- mu_gauss(post)
  sd_i <- if (is.null(post$sigma)) {
    matrix(obs_sd_fixed, nrow(mu), ncol(mu), byrow = TRUE)  # m0: sigma is data
  } else {
    outer(as.vector(post$sigma), invs)                      # sigma_s * inv_sqrt_n_i
  }
  dnorm(Y(nrow(mu)), mu, sd_i, log = TRUE)
}

ll_binom <- function(fit) {
  post   <- extract.samples(fit)
  eta    <- post$a_listing[, df$listing_idx] +
              outer(as.vector(post$b_rev), df$log_rev)
  p      <- 1 / (1 + exp(-eta))
  ns     <- nrow(p)
  dbinom(matrix(df$successes, ns, nrow(df), byrow = TRUE),
         matrix(df$number_of_reviews, ns, nrow(df), byrow = TRUE),
         p, log = TRUE)
}

LL <- list(m0 = ll_gauss_scaled(m0), m1 = ll_gauss_scaled(m1),
           m2 = ll_binom(m2),        m3 = ll_gauss_scaled(m3))

# ---- 8. Model comparison: WAIC & PSIS-LOO -----------------------------------
score <- function(ll, label) {
  w <- suppressWarnings(loo::waic(ll))
  l <- suppressWarnings(loo::loo(ll))
  tibble(model = label,
         WAIC    = w$estimates["waic", "Estimate"],
         SE_WAIC = w$estimates["waic", "SE"],
         pWAIC   = w$estimates["p_waic", "Estimate"],
         LOOIC   = l$estimates["looic", "Estimate"],
         p_loo   = l$estimates["p_loo", "Estimate"],
         bad_k   = sum(l$diagnostics$pareto_k > 0.7))
}

cat("\n\n############ MODEL COMPARISON ############\n")
cat("Gaussian family only -- same outcome y_i, so these are comparable.\n")
gauss_tbl <- bind_rows(
  score(LL$m0, "m0 unpooled (no pooling)"),
  score(LL$m1, "m1 Approach A (listing pooling)"),
  score(LL$m3, "m3 Approach C (host + listing)")
) %>%
  arrange(WAIC) %>%
  mutate(dWAIC  = WAIC - min(WAIC),
         weight = exp(-0.5 * dWAIC) / sum(exp(-0.5 * dWAIC))) %>%
  relocate(dWAIC, weight, .after = SE_WAIC)
tee(as.data.frame(gauss_tbl), "20_compare_gaussian.txt")

# Pairwise elpd differences with SEs -- more honest than raw WAIC gaps.
cat("\n---- pairwise PSIS-LOO differences (elpd_diff +/- SE) ----\n")
loos <- list(`m0 unpooled` = suppressWarnings(loo::loo(LL$m0)),
             `m1 ApproachA` = suppressWarnings(loo::loo(LL$m1)),
             `m3 ApproachC` = suppressWarnings(loo::loo(LL$m3)))
tee(loo::loo_compare(loos), "21_loo_compare_gaussian.txt")

cat("\n---- m2 Approach B, scored separately ----\n")
cat("Outcome is S_i ~ Binomial(n_i, p_i), NOT y_i. Its WAIC lives on a\n",
    "different scale and must NOT be placed in the table above.\n", sep = "")
tee(as.data.frame(score(LL$m2, "m2 Approach B (binomial)")),
    "22_score_binomial.txt")

cat("\nNOTE: high p_waic / Pareto-k here is structural, not a bug -- one\n",
    "observation and one free intercept per listing. See header note 4.\n", sep = "")

# ---- 9. Posterior estimates on the rating scale -----------------------------
post_mu_mean <- function(fit) colMeans(mu_gauss(extract.samples(fit)))

post_rating_binom <- function(fit) {             # logit p -> rating in [1,5]
  post <- extract.samples(fit)
  eta  <- post$a_listing[, df$listing_idx] +
            outer(as.vector(post$b_rev), df$log_rev)
  colMeans(1 + 4 / (1 + exp(-eta)))
}

# Each model has its own population mean, and m2's lives on the logit scale, so
# it must be mapped back to [1,5] before it can be drawn on a rating axis. One
# shared reference line would be wrong in two of the three panels.
a_bar_m1 <- mean(extract.samples(m1)$a_bar)
a_bar_m2 <- mean(1 + 4 / (1 + exp(-extract.samples(m2)$a_bar)))
a_bar_m3 <- mean(extract.samples(m3)$a_bar)

est <- df %>%
  transmute(id, host_id, number_of_reviews,
            raw_rating   = review_scores_rating,     # no-pooling benchmark
            m0_unpooled  = post_mu_mean(m0),
            m1_approachA = post_mu_mean(m1),
            m2_approachB = post_rating_binom(m2),
            m3_approachC = post_mu_mean(m3))
if (IS_SYNTHETIC) est$quality_true <- df$quality_true

write_csv(est, file.path(CFG$out_dir, "30_shrunken_ratings.csv"))
message("  wrote ", file.path(CFG$out_dir, "30_shrunken_ratings.csv"))

long <- est %>%
  pivot_longer(c(m1_approachA, m2_approachB, m3_approachC),
               names_to = "model", values_to = "estimate") %>%
  mutate(model = recode(model,
           m1_approachA = "A  Gaussian, sd = sigma/sqrt(n)",
           m2_approachB = "B  Binomial success counts",
           m3_approachC = "C  Host + listing intercepts"),
         shrinkage = estimate - raw_rating)

# How much shrinkage, by review count? This is the headline table.
cat("\n\n############ SHRINKAGE BY REVIEW COUNT ############\n")
shrink_tbl <- long %>%
  mutate(bucket = cut(number_of_reviews, c(0, 2, 5, 10, 50, Inf),
                      labels = c("1-2", "3-5", "6-10", "11-50", "50+"))) %>%
  summarise(listings = n(),
            mean_raw = mean(raw_rating),
            mean_est = mean(estimate),
            mean_abs_shrinkage = mean(abs(shrinkage)),
            .by = c(model, bucket)) %>%
  arrange(model, bucket)
tee(as.data.frame(shrink_tbl), "31_shrinkage_by_n.txt")
cat("\nExpect mean_abs_shrinkage to fall monotonically as n grows:\n",
    "small-n listings are pulled hard toward a_bar, large-n listings are left\n",
    "essentially at their empirical average.\n", sep = "")

# On synthetic data the latent truth is known, so pooling vs no-pooling can be
# scored directly -- the classic Chapter 13 demonstration.
if (IS_SYNTHETIC) {
  cat("\n############ RMSE vs KNOWN TRUTH (synthetic data only) ############\n")
  rmse <- function(a, b) sqrt(mean((a - b)^2))
  rmse_tbl <- tibble(
    estimator = c("raw average (no pooling)", "m0 unpooled fixed effects",
                  "m1 Approach A", "m2 Approach B", "m3 Approach C"),
    RMSE = c(rmse(est$raw_rating,   est$quality_true),
             rmse(est$m0_unpooled,  est$quality_true),
             rmse(est$m1_approachA, est$quality_true),
             rmse(est$m2_approachB, est$quality_true),
             rmse(est$m3_approachC, est$quality_true))
  ) %>% arrange(RMSE)
  tee(as.data.frame(rmse_tbl), "32_rmse_vs_truth.txt")
  small <- est %>% filter(number_of_reviews <= 3)
  cat("\nRestricted to listings with n <= 3 (", nrow(small), " listings):\n", sep = "")
  cat("  raw average   RMSE:", round(rmse(small$raw_rating,   small$quality_true), 4), "\n")
  cat("  m1 Approach A RMSE:", round(rmse(small$m1_approachA, small$quality_true), 4), "\n")
  cat("  m3 Approach C RMSE:", round(rmse(small$m3_approachC, small$quality_true), 4), "\n")
}

# ---- 10. Covariate effect: beta_rev ----------------------------------------
cat("\n\n############ POSTERIOR FOR beta_rev ############\n")
beta_tbl <- imap_dfr(models, function(fit, nm) {
  b <- as.vector(extract.samples(fit)$b_rev)
  tibble(model = nm,
         scale = if (nm == "m2") "logit" else "rating points",
         mean = mean(b), sd = sd(b),
         q5.5 = quantile(b, .055), q94.5 = quantile(b, .945),
         `P(b>0)` = mean(b > 0))
})
tee(as.data.frame(beta_tbl), "40_beta_rev.txt")
cat("\nb_rev is the association between log review count and baseline quality,\n",
    "*after* pooling. Caution: with one observation per listing and one free\n",
    "intercept per listing, b_rev is identified only through the adaptive\n",
    "prior, so it partly trades off against the spread of a_listing.\n", sep = "")

# ---- 11. Figures ------------------------------------------------------------
message("\n>>> figures")
model_cols <- c("A  Gaussian, sd = sigma/sqrt(n)" = PAL$s1,
                "B  Binomial success counts"      = PAL$s2,
                "C  Host + listing intercepts"    = PAL$s3)

# FIG 1 -- the shrinkage plot. Raw vs posterior, coloured by n (sequential).
abar_lines <- tibble(
  model = c("A  Gaussian, sd = sigma/sqrt(n)", "B  Binomial success counts",
            "C  Host + listing intercepts"),
  a_bar = c(a_bar_m1, a_bar_m2, a_bar_m3)
)

p1 <- ggplot(long, aes(raw_rating, estimate)) +
  geom_abline(slope = 1, intercept = 0, colour = PAL$axis,
              linetype = "22", linewidth = 0.5) +
  geom_hline(data = abar_lines, aes(yintercept = a_bar), colour = PAL$muted,
             linetype = "solid", linewidth = 0.4) +
  geom_point(aes(colour = number_of_reviews), size = 1.5, alpha = 0.75,
             stroke = 0) +
  scale_colour_gradientn(colours = PAL$seq, transform = "log10",
                         name = "reviews (n)",
                         breaks = c(1, 10, 100, 1000)) +
  facet_wrap(~ model) +
  labs(
    title = "Partial pooling shrinks noisy low-review listings toward the mean",
    subtitle = paste0("Dashed line = no shrinkage (estimate == raw). ",
                      "Grey line = that model's own population mean a_bar",
                      " (B's mapped back from logit).\n",
                      "Pale points (few reviews) collapse toward the grey line; ",
                      "dark points (many reviews) stay on the dashed line."),
    x = "Raw observed rating (no-pooling benchmark)",
    y = "Posterior mean estimate",
    caption = paste0("n = ", nrow(df), " listings, ", N_host, " hosts. ",
                     "Full per-listing values in outputs/30_shrunken_ratings.csv")
  ) +
  theme_rethink()
save_fig(p1, "fig1_shrinkage_scatter.png", w = 11, h = 4.6)

# FIG 2 -- shrinkage magnitude vs n, with a 10-90% envelope.
# The y position already encodes the size AND sign of the pull, so colouring by
# shrinkage too would double-encode one variable on two channels (and a single
# +3.4 outlier flattens the whole ramp to near-white). Instead the colour budget
# goes to a quantile envelope, which adds information the points do not carry:
# the rate at which the correction collapses.
env <- long %>%
  mutate(nbin = cut(number_of_reviews,
                    breaks = c(0, 1, 2, 3, 5, 8, 13, 21, 34, 55, 90, 150, 250, Inf))) %>%
  summarise(n_mid = exp(mean(log(number_of_reviews))),
            lo = quantile(shrinkage, 0.10), hi = quantile(shrinkage, 0.90),
            .by = c(model, nbin)) %>%
  filter(!is.na(nbin))

p2 <- ggplot(long, aes(number_of_reviews, shrinkage)) +
  geom_ribbon(data = env, inherit.aes = FALSE,
              aes(x = n_mid, ymin = lo, ymax = hi),
              fill = PAL$seq[1], alpha = 0.75) +
  geom_hline(yintercept = 0, colour = PAL$axis, linewidth = 0.4) +
  geom_point(colour = PAL$seq[4], size = 1.4, alpha = 0.65) +
  scale_x_log10(breaks = c(1, 3, 10, 30, 100, 300, 1000)) +
  facet_wrap(~ model) +
  labs(
    title = "Shrinkage vanishes as the review count grows",
    subtitle = paste0("Distance from zero is how far pooling moved the listing; ",
                      "above zero = revised up, below = revised down. ",
                      "Band = 10-90% of listings."),
    x = "Number of reviews (log scale)",
    y = "Posterior estimate minus raw rating",
    caption = "Per-bucket magnitudes in outputs/31_shrinkage_by_n.txt"
  ) +
  theme_rethink()
save_fig(p2, "fig2_shrinkage_vs_n.png", w = 11, h = 4.6)

# FIG 3 -- posterior predictive check, Gaussian models (same outcome scale).
set.seed(CFG$seed)
ppc_draws <- function(fit, label, n_rep = 60) {
  post <- extract.samples(fit)
  mu   <- mu_gauss(post)
  idx  <- sample(nrow(mu), min(n_rep, nrow(mu)))
  map_dfr(seq_along(idx), function(k) {
    s <- idx[k]
    tibble(rep = k, model = label,
           value = rnorm(ncol(mu), mu[s, ], post$sigma[s] * invs))
  })
}
ppc <- bind_rows(ppc_draws(m1, "A  Gaussian, sd = sigma/sqrt(n)"),
                 ppc_draws(m3, "C  Host + listing intercepts"))

# The PPC's real finding: a Gaussian likelihood has unbounded support, so
# Approaches A and C put probability mass on ratings above 5 -- values that
# cannot exist. Quantify it rather than leaving it to the eye.
pct_impossible <- ppc %>%
  summarise(pct = 100 * mean(value > 5 | value < 1), .by = model)
cat("\n---- posterior predictive mass on impossible ratings ----\n")
print(as.data.frame(pct_impossible))
cat("Observed ratings are hard-capped at 5, but the Gaussian likelihood in A\n",
    "and C is unbounded, so both predict ratings that cannot occur. This is\n",
    "the principled argument for Approach B: a Binomial on n_i trials respects\n",
    "the [1,5] bound by construction. It is also why B is worth keeping even\n",
    "though its WAIC cannot be compared with A's and C's.\n", sep = "")

p3 <- ggplot() +
  geom_vline(xintercept = 5, colour = PAL$div_hi, linewidth = 0.5,
             linetype = "22") +
  geom_density(data = ppc, aes(value, group = rep),
               colour = scales::alpha(PAL$s1, 0.22), linewidth = 0.3) +
  geom_density(data = crossing(model = unique(ppc$model), value = y),
               aes(value), colour = PAL$ink, linewidth = 1.1) +
  facet_wrap(~ model) +
  coord_cartesian(xlim = c(2.5, 5.6)) +
  labs(
    title = "Posterior predictive check: both Gaussian models predict impossible ratings",
    subtitle = paste0("Black = observed. Blue = 60 posterior predictive ",
                      "replicates. Red dashed = the hard ceiling at 5, which\n",
                      "the replicates cross because a Gaussian has unbounded ",
                      "support."),
    x = "Rating", y = "Density",
    caption = paste0("Approach B is omitted: its outcome is a count out of n_i ",
                     "trials, a different scale. It cannot breach the bound.")
  ) +
  theme_rethink()
save_fig(p3, "fig3_posterior_predictive.png", w = 10, h = 5)

# FIG 4 -- beta_rev posteriors. Faceted by SCALE, never on one shared axis:
# m2's slope is in logit units and m1/m3's in rating points.
beta_draws <- imap_dfr(models[c("m1", "m2", "m3")], function(fit, nm) {
  tibble(model = recode(nm,
           m1 = "A  Gaussian, sd = sigma/sqrt(n)",
           m2 = "B  Binomial success counts",
           m3 = "C  Host + listing intercepts"),
         scale = if (nm == "m2") "logit scale" else "rating-point scale",
         b_rev = as.vector(extract.samples(fit)$b_rev))
})
# Direct labels sit at each density's PEAK, not at y = 0. Anchoring them to the
# mean put A and C on top of each other in the shared rating-point panel and
# rendered both unreadable; short A/B/C tags at the mode cannot collide, and the
# legend carries the full names.
beta_lab <- beta_draws %>%
  reframe({ d <- density(b_rev)
            tibble(x = d$x[which.max(d$y)], y = max(d$y)) },
          .by = c(model, scale)) %>%
  mutate(tag = substr(model, 1, 1))

p4 <- ggplot(beta_draws, aes(b_rev, fill = model, colour = model)) +
  geom_vline(xintercept = 0, colour = PAL$axis, linewidth = 0.4) +
  geom_density(alpha = 0.35, linewidth = 0.7) +
  geom_text(data = beta_lab, aes(x = x, y = y, label = tag),
            inherit.aes = FALSE, vjust = -0.5, size = 4, fontface = "bold",
            colour = PAL$ink) +
  scale_fill_manual(values = model_cols, name = NULL) +
  scale_colour_manual(values = model_cols, name = NULL) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.12))) +
  facet_wrap(~ scale, scales = "free") +
  labs(
    title = "Posterior for beta_rev: does review count predict baseline quality?",
    subtitle = paste0("Panels are separate scales and must not share an axis: ",
                      "B's slope is in logit units, A's and C's in rating points. ",
                      "Mass away from zero is the effect."),
    x = "beta_rev", y = "Density",
    caption = "Numeric intervals in outputs/40_beta_rev.txt"
  ) +
  theme_rethink() + theme(legend.position = "bottom")
save_fig(p4, "fig4_beta_rev_posterior.png", w = 10, h = 5)

# FIG 5 -- Model C's variance decomposition: host-level vs listing-level.
# READ THIS ONE WITH CARE. Do not conclude "listings barely vary". Most hosts
# own exactly ONE listing, and for those hosts a_host and a_listing are
# perfectly confounded -- only their sum is identified, and the split between
# them is decided by the priors, not by the data. Only multi-listing hosts carry
# real information about which level the variation belongs to.
pct_single <- mean(table(df$host_idx) == 1)
vc <- extract.samples(m3)
p5 <- tibble(
    component = rep(c("sigma_host (between hosts)",
                      "sigma_listing (between listings)"),
                    each = length(vc$sigma_host)),
    value = c(as.vector(vc$sigma_host), as.vector(vc$sigma_listing))
  ) %>%
  ggplot(aes(value, fill = component, colour = component)) +
  geom_density(alpha = 0.35, linewidth = 0.7) +
  scale_fill_manual(values = c(PAL$s1, PAL$s3), name = NULL) +
  scale_colour_manual(values = c(PAL$s1, PAL$s3), name = NULL) +
  labs(
    title = "Approach C: how the variance splits between host and listing",
    subtitle = paste0(
      round(100 * pct_single), "% of hosts here own exactly one listing, and for ",
      "those the two effects are\nconfounded -- only their sum is identified. ",
      "The split below leans on the ",
      sum(table(df$host_idx) > 1), " multi-listing hosts."),
    x = "Standard deviation (rating points)", y = "Density",
    caption = paste0("Not evidence that listings do not vary. ",
                     "Full summary in outputs/10_precis_m3.txt")
  ) +
  theme_rethink() + theme(legend.position = "bottom")
save_fig(p5, "fig5_variance_components.png", w = 8.5, h = 5)

# ---- 12. Wrap-up ------------------------------------------------------------
cat("\n\n############ DONE ############\n")
cat("Data source:", if (IS_SYNTHETIC) "SYNTHETIC" else CFG$csv_path, "\n")
cat("Parameterisation:", if (CFG$noncentered) "non-centred" else "centred", "\n")
cat("Listings:", nrow(df), " Hosts:", N_host, "\n")
cat("\nAll tables and figures written to ", CFG$out_dir, "/\n", sep = "")
print(sort(basename(list.files(CFG$out_dir, full.names = TRUE))))
