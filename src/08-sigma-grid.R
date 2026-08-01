# The extension: is sigma really a regularization strength?
#
# Refit the multilevel model with sigma PINNED at a grid of values, leaving
# a_bar free, and score each fit by cross-validation. If the chapter's claim
# holds, the score should peak where the posterior of sigma peaked in the model
# that estimated it.

source("src/02-sample.R")

post       <- readRDS(file.path(CACHE, "post-partial.rds"))
sigma_grid <- exp(seq(log(0.15), log(2.5), length.out = 12))
cache_file <- file.path(CACHE, "sigma-grid.rds")

# --- fit one model per grid value --------------------------------------------
# sigma is passed in as data, which keeps the formula literal: only sigma is
# fixed, a_bar stays free. Twelve fits take a few minutes, so they are cached;
# delete .cache/sigma-grid.rds to force a refit.

fixed_sigma <- alist(
  P ~ dbinom(N, p),
  logit(p) <- a[listing],
  a[listing] ~ dnorm(a_bar, sigma_fixed),
  a_bar ~ dnorm(0, 1.5)
)

cached <- if (file.exists(cache_file)) readRDS(cache_file) else NULL

if (!is.null(cached) && isTRUE(all.equal(cached$sigma_grid, sigma_grid))) {
  cat("reusing cached grid fits\n")
  grid <- cached
} else {
  grid <- list(sigma_grid = sigma_grid,
               psis = numeric(), waic = numeric(),
               bad_k = integer(), abar = list())

  for (s in sigma_grid) {
    cat("\n--- fitting sigma =", round(s, 3), "---\n")
    m <- ulam(fixed_sigma, data = c(dat, list(sigma_fixed = s)),
              chains = CHAINS, cores = CHAINS, log_lik = TRUE, seed = SEED)

    # rethinking reports PSIS and WAIC on the deviance scale, where lower is
    # better. Convert to elpd = -PSIS/2 so every curve is maximised.
    ps <- suppressWarnings(PSIS(m, pointwise = TRUE))
    grid$psis  <- c(grid$psis,  -sum(ps$PSIS) / 2)
    grid$bad_k <- c(grid$bad_k, sum(ps$k > 0.7))
    grid$waic  <- c(grid$waic,  -suppressWarnings(WAIC(m)$WAIC) / 2)
    grid$abar  <- c(grid$abar,  list(extract.samples(m)$a_bar))
  }
  saveRDS(grid, cache_file)
}

# --- direct leave-one-cluster-out --------------------------------------------
# PSIS is on thin ice here. Every listing has exactly ONE binomial observation,
# so dropping listing i removes all information about a[i], and the importance
# weights become wildly heavy-tailed (see the bad_k column).
#
# The same fact makes the honest answer easy. With a[i] unconstrained by the
# remaining data, the leave-one-out prediction is just the prediction for a
# brand new listing:
#
#   p(P_i | y_-i, sigma) = INT Binomial(P_i | N_i, logistic(a))
#                              Normal(a | a_bar, sigma) da
#
# averaged over the posterior of a_bar. No importance sampling anywhere. It
# handles a[i] exactly; its one approximation is reusing the full-data
# posterior of a_bar instead of refitting without listing i.

loco_elpd <- function(abar, sigma, z) {
  p <- inv_logit(rep(abar, length.out = length(z)) + z * sigma)
  sum(vapply(seq_along(dat$P),
             function(i) log_mean_exp(dbinom(dat$P[i], dat$N[i], p, log = TRUE)),
             numeric(1)))
}

# Common random numbers across the grid, so the curve is smooth and differences
# between grid points are not swamped by Monte Carlo noise.
#
# REPS multiplies the ~2000 posterior draws. 50 is already far more than the
# peak needs -- CHECK_REPS below reproduces the same peak with a fifth of the
# sample and independent draws -- and each extra unit costs 200 listings x 2000
# densities x 12 grid points, which adds up fast.
REPS       <- 50
CHECK_REPS <- 10

n_draws <- length(grid$abar[[1]])
set.seed(SEED)
z_main  <- rnorm(n_draws * REPS)
set.seed(SEED * 100)
z_check <- rnorm(n_draws * CHECK_REPS)

score <- function(z) {
  vapply(seq_along(sigma_grid),
         function(i) loco_elpd(grid$abar[[i]], sigma_grid[i], z), numeric(1))
}
grid$loco  <- score(z_main)
grid$check <- score(z_check)

# --- where does each criterion peak? -----------------------------------------

peak <- function(y) sigma_grid[which.max(y)]

dens_sigma <- density(post$sigma)
sigma_mode <- dens_sigma$x[which.max(dens_sigma$y)]
sigma_pi   <- PI(post$sigma, prob = 0.89)

saveRDS(c(grid, list(mode = sigma_mode, pi = sigma_pi, dens = dens_sigma)),
        cache_file)

tbl("06-sigma-grid.txt",
    data.frame(sigma      = round(sigma_grid, 3),
               elpd_psis  = round(grid$psis, 2),
               bad_k      = grid$bad_k,
               elpd_waic  = round(grid$waic, 2),
               elpd_loco  = round(grid$loco, 2),
               loco_check = round(grid$check, 2)),
    header = c("Cross-validation over a grid of FIXED sigma",
               "elpd, higher is better. bad_k counts listings with Pareto k > 0.7.",
               "",
               sprintf("peak: PSIS %.3f | WAIC %.3f | direct %.3f | check %.3f",
                       peak(grid$psis), peak(grid$waic),
                       peak(grid$loco), peak(grid$check)),
               sprintf("sigma ~ dexp(1): mode %.3f, 89%% [%.3f, %.3f]",
                       sigma_mode, sigma_pi[1], sigma_pi[2])))
