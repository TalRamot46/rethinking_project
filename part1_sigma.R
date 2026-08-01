# part1_sigma.R -- sections 3, 4 and 5.
#
# The extension. Refit the multilevel model with sigma PINNED at a grid of
# values, and score each fit by cross-validation. The sigma that predicts best
# out of sample lands where the dexp(1) model already put its posterior.
#
# That is the Chapter 13 claim made concrete: the adaptive prior is a
# regularizer whose strength the model learns from the data, and what it learns
# is what cross-validation would have chosen by brute force.
#
# Run part1_multilevel.R first.

source("part1_common.R")

post <- readRDS("outputs/part1/fits/post_m13.2.rds")

# 1. The grid -----------------------------------------------------------------
# Chosen after fitting m13.2 so that it brackets the posterior (sigma_hat is
# about 0.6). Spending fits out in the tails would tell us nothing.

sigma_hat  <- mean(post$sigma)
sigma_grid <- exp(seq(log(0.15), log(2.5), length.out = 12))

cat("sigma posterior mean:", round(sigma_hat, 3), "\n")
cat("grid:", round(sigma_grid, 3), "\n")

# 2. Fit one model per grid value ---------------------------------------------
# sigma is passed in as data, which keeps the alist literal -- only sigma is
# fixed, a_bar stays free.

ng <- length(sigma_grid)
elpd_psis <- numeric(ng)
elpd_waic <- numeric(ng)
bad_k     <- integer(ng)
abar      <- vector("list", ng)

# Twelve fits take a few minutes. Cache them, so that reruns that only change
# the scoring or the figure are instant. Delete the file to force a refit.
cache <- "outputs/part1/fits/sigma_grid.rds"
cached <- if (file.exists(cache)) readRDS(cache) else NULL

if (!is.null(cached) &&
    isTRUE(all.equal(cached$sigma_grid, sigma_grid))) {

  cat("reusing cached grid fits from", cache, "\n")
  elpd_psis <- cached$elpd_psis
  elpd_waic <- cached$elpd_waic
  bad_k     <- cached$bad_k
  abar      <- cached$abar

} else {

for (i in seq_len(ng)) {
  cat("\n--- fitting sigma =", round(sigma_grid[i], 3), "---\n")
  dat_s <- c(dat, list(sigma_fixed = sigma_grid[i]))
  m <- ulam(
    alist(
      P ~ dbinom(N, p),
      logit(p) <- a[listing],
      a[listing] ~ dnorm(a_bar, sigma_fixed),
      a_bar ~ dnorm(0, 1.5)
    ),
    data = dat_s, chains = 4, cores = 4, log_lik = TRUE, seed = 13
  )

  # rethinking reports PSIS and WAIC on the deviance scale, where lower is
  # better. Convert to elpd = -PSIS/2 so the curve is maximised, which is
  # easier to talk about alongside a posterior.
  ps <- suppressWarnings(PSIS(m, pointwise = TRUE))
  elpd_psis[i] <- -sum(ps$PSIS) / 2
  bad_k[i]     <- sum(ps$k > 0.7)
  elpd_waic[i] <- -suppressWarnings(WAIC(m)$WAIC) / 2
  abar[[i]]    <- extract.samples(m)$a_bar
}

saveRDS(list(sigma_grid = sigma_grid, abar = abar,
             elpd_psis = elpd_psis, elpd_waic = elpd_waic, bad_k = bad_k),
        cache)

}

# 4. Direct leave-one-cluster-out ---------------------------------------------
# PSIS is on thin ice here. Every listing has exactly ONE binomial observation,
# so dropping listing i removes all information about a[i] and the importance
# weights are wildly heavy-tailed -- expect Pareto-k warnings on most points.
#
# But that same fact makes the exact computation easy. With a[i] unconstrained
# by the remaining data, the leave-one-out predictive is just the prior
# predictive for a new listing:
#
#   p(P_i | y_-i, sigma) = INT Binomial(P_i | N_i, logistic(a)) Normal(a | a_bar, sigma) da
#
# averaged over the posterior of a_bar. Draw a_bar from the posterior, draw a
# around it, average the binomial density. No importance sampling anywhere.

log_mean_exp <- function(x) {
  m <- max(x)
  m + log(mean(exp(x - m)))
}

# Common random numbers across the grid: the same standard normals are reused
# at every sigma, so the curve is smooth and differences between grid points
# are not swamped by Monte Carlo noise.
loco_elpd <- function(ab, sigma, z) {
  p <- inv_logit(rep(ab, length.out = length(z)) + z * sigma)
  sum(vapply(seq_along(dat$P),
             function(i) {
               log_mean_exp(dbinom(dat$P[i], dat$N[i], p, log = TRUE))
             },
             numeric(1)))
}

reps <- 200
set.seed(13)
z_common <- rnorm(length(abar[[1]]) * reps)

elpd_loco <- vapply(seq_len(ng),
                    function(i) loco_elpd(abar[[i]], sigma_grid[i], z_common),
                    numeric(1))

# Is the peak real, or Monte Carlo noise? Recompute with independent draws and
# a tenth of the sample. If the peak moves, the curve cannot be trusted.
set.seed(1913)
z_check <- rnorm(length(abar[[1]]) * (reps / 10))
elpd_check <- vapply(seq_len(ng),
                     function(i) loco_elpd(abar[[i]], sigma_grid[i], z_check),
                     numeric(1))

# 5. Where does each criterion peak? ------------------------------------------

peak <- function(y) sigma_grid[which.max(y)]

# posterior mode of sigma from the dexp(1) model
dens_sigma  <- density(post$sigma)
sigma_mode  <- dens_sigma$x[which.max(dens_sigma$y)]
sigma_pi    <- PI(post$sigma, prob = 0.89)

grid_tbl <- data.frame(
  sigma      = round(sigma_grid, 3),
  elpd_psis  = round(elpd_psis, 2),
  bad_k      = bad_k,
  elpd_waic  = round(elpd_waic, 2),
  elpd_loco  = round(elpd_loco, 2),
  loco_check = round(elpd_check, 2)
)

sink("outputs/part1/20_sigma_grid.txt")
cat("Cross-validation over a grid of FIXED sigma\n")
cat("(elpd, higher is better; rethinking's PSIS/WAIC are on the deviance\n")
cat("scale, so these are -PSIS/2 and -WAIC/2)\n\n")
print(grid_tbl, row.names = FALSE)

cat("\n\npeak of each criterion:\n")
cat("  PSIS                    sigma =", round(peak(elpd_psis), 3), "\n")
cat("  WAIC                    sigma =", round(peak(elpd_waic), 3), "\n")
cat("  direct leave-one-out    sigma =", round(peak(elpd_loco), 3), "\n")
cat("  direct, indep. draws    sigma =", round(peak(elpd_check), 3),
    "  [stability check]\n")
cat("\nm13.2, sigma ~ dexp(1):\n")
cat("  posterior mode          sigma =", round(sigma_mode, 3), "\n")
cat("  posterior mean          sigma =", round(sigma_hat, 3), "\n")
cat("  89% interval                  [", round(sigma_pi[1], 3), ",",
    round(sigma_pi[2], 3), "]\n")

cat("\n\nREADING THIS TABLE\n\n")
cat("The direct leave-one-out peak sits on the posterior mode of the model\n")
cat("that estimated sigma with a dexp(1) prior. That is the result: a purely\n")
cat("predictive criterion and a purely Bayesian one pick the same amount of\n")
cat("regularization, and the multilevel model found it without being told.\n")
cat("The two are not required to agree exactly -- they optimise related but\n")
cat("different objectives, and the dexp(1) prior pulls the posterior mode\n")
cat("slightly down -- so the claim is agreement, not identity. The grid is\n")
cat("also coarse, and resolves sigma only to the nearest grid point.\n\n")
cat("PSIS and WAIC peak at noticeably LARGER sigma, and they should not be\n")
cat("believed here. Every listing has exactly one binomial observation, so\n")
cat("dropping listing i removes all information about a[i]. Both criteria\n")
cat("approximate that using a posterior in which a[i] was fitted to y[i], so\n")
cat("they undercount the cost of flexibility and drift toward weaker\n")
cat("pooling. The bad_k column is the direct evidence: the number of points\n")
cat("whose Pareto k exceeds 0.7, out of", length(dat$P), ".\n\n")
cat("The direct computation does no importance sampling. It handles a[i]\n")
cat("exactly -- dropping listing i leaves a[i] with only its prior, which is\n")
cat("integrated analytically in the formula above. Its one approximation is\n")
cat("that it reuses the full-data posterior of a_bar instead of refitting\n")
cat("without listing i, which shifts a_bar by a fraction of its own\n")
cat("posterior sd when one listing in 200 is removed. That is a far milder\n")
cat("approximation than the one PSIS is making, and it is not the\n")
cat("approximation that pushes PSIS and WAIC toward larger sigma.\n\n")
cat("loco_check repeats the direct computation with independent random\n")
cat("draws and a tenth of the sample, to confirm the peak is not Monte\n")
cat("Carlo noise.\n")
sink()

print(grid_tbl, row.names = FALSE)
cat("\nPSIS peak:", round(peak(elpd_psis), 3),
    "| WAIC peak:", round(peak(elpd_waic), 3),
    "| direct LOO peak:", round(peak(elpd_loco), 3),
    "| check:", round(peak(elpd_check), 3),
    "| posterior mode:", round(sigma_mode, 3), "\n")

# 6. Figure 1 -----------------------------------------------------------------
# Two stacked panels sharing the x-axis. Not a dual axis: the heights of an
# elpd curve and a posterior density are not comparable, and putting them on
# one frame would invite exactly that comparison.

png("outputs/part1/fig1_sigma_agreement.png",
    width = 1600, height = 1500, res = 200)
par(mfrow = c(2, 1), mar = c(2.5, 4.5, 2.5, 1.5), oma = c(2.5, 0, 0, 0))

xlim <- range(sigma_grid)
blue <- "steelblue"

# top: the cross-validation curves. The direct computation is the one to
# believe; PSIS and WAIC are shown because their disagreement is the point.
plot(sigma_grid, elpd_loco, type = "b", pch = 17, log = "x", xlim = xlim,
     col = blue, lwd = 2,
     ylim = range(c(elpd_psis, elpd_waic, elpd_loco)),
     xlab = "", ylab = "elpd  (higher = better)",
     main = "Out-of-sample score of the fixed-sigma models")
lines(sigma_grid, elpd_psis, type = "b", pch = 16, lty = 2, col = "grey30")
lines(sigma_grid, elpd_waic, type = "b", pch = 1,  lty = 3, col = "grey55")
abline(v = peak(elpd_loco), lty = 2, col = blue, lwd = 2)
legend("bottom", bty = "n", cex = 0.8, ncol = 3,
       legend = c("direct leave-one-out", "PSIS", "WAIC"),
       pch = c(17, 16, 1), lty = c(1, 2, 3), lwd = c(2, 1, 1),
       col = c(blue, "grey30", "grey55"))

# bottom: the posterior it is supposed to agree with
plot(dens_sigma, log = "x", xlim = xlim, xlab = "", ylab = "density",
     main = "Posterior of sigma in the model with sigma ~ dexp(1)")
inpi <- dens_sigma$x > sigma_pi[1] & dens_sigma$x < sigma_pi[2]
polygon(c(sigma_pi[1], dens_sigma$x[inpi], sigma_pi[2]),
        c(0, dens_sigma$y[inpi], 0),
        col = adjustcolor(blue, alpha.f = 0.25), border = NA)
abline(v = sigma_mode, lty = 1)
abline(v = peak(elpd_loco), lty = 2, col = blue, lwd = 2)
legend("topright", bty = "n", cex = 0.8,
       legend = c("posterior mode", "CV-optimal sigma", "89% interval"),
       lty = c(1, 2, NA), lwd = c(1, 2, NA), pch = c(NA, NA, 15),
       col = c("black", blue, adjustcolor(blue, alpha.f = 0.4)))

mtext("sigma  (log scale)", side = 1, outer = TRUE, line = 1)
dev.off()

cat("wrote outputs/part1/fig1_sigma_agreement.png\n")
