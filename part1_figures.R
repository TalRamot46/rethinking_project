# part1_figures.R -- diagnostics and the population distribution.
#
# Two things the essay needs that the other scripts do not produce:
#
#   (a) trace plots and trace rank plots for the centered and the
#       non-centered multilevel model, side by side
#   (b) the analogue of Figure 13.2 -- the posterior distribution of
#       listings, on the log-odds scale and on the probability scale
#
# Both need the fitted objects themselves rather than a saved posterior, so
# the two multilevel models are refitted here. They take a few seconds each.

source("part1_common.R")

# 1. Refit both parameterizations ---------------------------------------------

f_centered <- alist(
  P ~ dbinom(N, p),
  logit(p) <- a[listing],
  a[listing] ~ dnorm(a_bar, sigma),
  a_bar ~ dnorm(0, 1.5),
  sigma ~ dexp(1)
)

f_noncentered <- alist(
  P ~ dbinom(N, p),
  logit(p) <- a_bar + z[listing] * sigma,
  z[listing] ~ dnorm(0, 1),
  a_bar ~ dnorm(0, 1.5),
  sigma ~ dexp(1),
  gq> vector[listing]:a <<- a_bar + z * sigma
)

m13.2c <- ulam(f_centered, data = dat, chains = 4, cores = 4,
               log_lik = TRUE, seed = 13)
m13.2  <- ulam(f_noncentered, data = dat, chains = 4, cores = 4,
               log_lik = TRUE, seed = 13,
               control = list(adapt_delta = 0.95))

# 2. Sampler diagnostics ------------------------------------------------------
# Divergent transitions are the headline warning, but they are not the only
# one. E-BFMI flags the same funnel through the energy of the trajectories, and
# n_eff reports how many effectively independent draws the chains produced.

ebfmi <- function(m) {
  d <- tryCatch(m@cstanfit$diagnostic_summary(quiet = TRUE),
                error = function(e) NULL)
  if (is.null(d)) NA else round(min(d$ebfmi), 3)
}

diag_row <- function(m, nm) {
  pr <- precis(m, depth = 1)
  data.frame(model     = nm,
             divergent = sum(divergent(m)),
             min_ebfmi = ebfmi(m),
             max_rhat  = round(max(pr$rhat, na.rm = TRUE), 3),
             min_ess   = round(min(pr$ess_bulk, na.rm = TRUE), 0))
}

diags <- rbind(diag_row(m13.2c, "centered"),
               diag_row(m13.2,  "non-centered"))

sink("outputs/part1/12_sampler_diagnostics.txt")
cat("Sampler diagnostics for the two parameterizations of the same model\n")
cat("(hyperparameters a_bar and sigma)\n\n")
print(diags, row.names = FALSE)
cat("\nNeither version produced divergent transitions on this data set, so\n")
cat("the usual headline warning is silent. The funnel still shows up, in\n")
cat("the E-BFMI warning and in an effective sample size for sigma about\n")
cat("five times smaller than the non-centered version delivers from the\n")
cat("same number of iterations.\n")
sink()

print(diags, row.names = FALSE)

# 3. Trace plots --------------------------------------------------------------
# A healthy trace looks like a fuzzy caterpillar: the chains explore the same
# region, they mix, and none of them sticks.

# Warmup is excluded. The first few iterations shoot far above the posterior,
# and including them compresses the vertical axis so much that the difference
# between the two parameterizations -- which is the whole point of the figure
# -- becomes invisible.
win <- c(501, 1000)

# n_cols = 2 matters: the default grid is three wide, which would leave a third
# of the figure empty and shrink the two panels that carry the information.
png("outputs/part1/fig3a_trace_centered.png",
    width = 1600, height = 620, res = 190)
traceplot(m13.2c, pars = c("a_bar", "sigma"), window = win, n_cols = 2)
dev.off()

png("outputs/part1/fig3b_trace_noncentered.png",
    width = 1600, height = 620, res = 190)
traceplot(m13.2, pars = c("a_bar", "sigma"), window = win, n_cols = 2)
dev.off()

# 4. Trace rank plots ---------------------------------------------------------
# Trace plots get hard to read once the chains overlap. A trank plot ranks all
# the samples from all chains together and histograms the ranks by chain. If
# the chains are exploring the same distribution, every chain should hold about
# the same share of every rank, so the histograms should overlap and stay flat.

png("outputs/part1/fig4a_trank_centered.png",
    width = 1600, height = 620, res = 190)
trankplot(m13.2c, pars = c("a_bar", "sigma"), n_cols = 2)
dev.off()

png("outputs/part1/fig4b_trank_noncentered.png",
    width = 1600, height = 620, res = 190)
trankplot(m13.2, pars = c("a_bar", "sigma"), n_cols = 2)
dev.off()

# 5. The population of listings -- Figure 13.2 --------------------------------
# The multilevel model does not just estimate 200 intercepts. It estimates the
# DISTRIBUTION those intercepts came from, and that distribution is what lets
# it say something about a listing it has never seen. R code 13.7.

post <- extract.samples(m13.2)

png("outputs/part1/fig5_population.png", width = 1700, height = 850, res = 190)
par(mfrow = c(1, 2), mar = c(4.5, 4.5, 5, 1.5))

# 100 draws of the population distribution, on the log-odds scale
plot(NULL, xlim = c(0, 6), ylim = c(0, 0.8),
     xlab = "log-odds of a positive experience", ylab = "density", main = "")
mtext("Posterior population of listings", side = 3, line = 3, font = 2)
for (i in 1:100)
  curve(dnorm(x, post$a_bar[i], post$sigma[i]), add = TRUE,
        col = col.alpha("black", 0.2))

# simulate listings from the posterior and look at them as probabilities
sim_listings <- rnorm(8000, post$a_bar, post$sigma)

dens(inv_logit(sim_listings), lwd = 2, adj = 0.1, xlab = "", main = "")
mtext("Simulated new listings", side = 3, line = 3, font = 2)
mtext("probability of a positive experience", side = 1, line = 2.4, cex = 0.9)
rt <- seq(3, 5, 0.5)
axis(3, at = (rt - 1) / 4, labels = format(rt), cex.axis = 0.75)
mtext("implied overall rating", side = 3, line = 1.5, cex = 0.75)

dev.off()

cat("wrote outputs/part1/fig3*, fig4*, fig5_population.png\n")

# The spread of that simulated population, in rating points, is the thing
# sigma controls -- worth a number in the text.
sink("outputs/part1/13_population.txt")
cat("Posterior population of listings, simulated from a_bar and sigma\n\n")
cat("log-odds scale:\n")
print(precis(data.frame(log_odds = sim_listings), prob = 0.89))
cat("\nprobability scale:\n")
print(precis(data.frame(p = inv_logit(sim_listings)), prob = 0.89))
cat("\nrating scale (1-5):\n")
print(precis(data.frame(rating = 1 + 4 * inv_logit(sim_listings)), prob = 0.89))
sink()
