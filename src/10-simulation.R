# Simulate from a KNOWN model and see which estimator recovers it.
#
# Everything up to here compared models on real data, where the truth is not
# available and WAIC has to stand in for it. Here the truth is chosen, so the
# error of each estimator can be measured directly. This is section 13.2 of the
# book -- R code 13.9 to 13.20 -- with listings in place of ponds.
#
# Two scenarios are run, because the answer depends on something the real data
# cannot tell us for free:
#
#   fitted  a_bar = 2.88, sigma = 0.62  -- the values estimated from the real
#           listings, so this is the problem actually at hand
#   book    a_bar = 1.50, sigma = 1.50  -- the book's own settings, where
#           listings differ much more from one another
#
# Partial pooling wins in both. Complete pooling is close behind in the first
# and far behind in the second, which is the point: how much pooling helps
# depends on how much the clusters genuinely vary.

source("src/00-setup.R")

n_listing <- 60
Ni        <- as.integer(rep(c(2, 5, 15, 50), each = 15))

simulate_scenario <- function(a_bar, sigma) {
  set.seed(SEED)
  sim <- data.frame(listing = 1:n_listing, Ni = Ni,
                    true_a = rnorm(n_listing, a_bar, sigma))
  sim$Pi <- rbinom(n_listing, prob = inv_logit(sim$true_a), size = sim$Ni)

  # the three estimators
  sim$p_true     <- inv_logit(sim$true_a)
  sim$p_nopool   <- sim$Pi / sim$Ni             # the empirical rating, no more
  sim$p_complete <- sum(sim$Pi) / sum(sim$Ni)   # one number for every listing

  m <- ulam(
    alist(
      Pi ~ dbinom(Ni, p),
      logit(p) <- a[listing],
      a[listing] ~ dnorm(a_bar, sigma),
      a_bar ~ dnorm(0, 1.5),
      sigma ~ dexp(1)
    ),
    data = list(Pi = sim$Pi, Ni = sim$Ni, listing = sim$listing),
    chains = CHAINS, cores = CHAINS, seed = SEED
  )
  sim$p_partpool <- apply(inv_logit(extract.samples(m)$a), 2, mean)

  sim$nopool_error   <- abs(sim$p_nopool   - sim$p_true)
  sim$partpool_error <- abs(sim$p_partpool - sim$p_true)
  sim$complete_error <- abs(sim$p_complete - sim$p_true)
  sim
}

scenarios <- list(
  fitted = list(a_bar = 2.88, sigma = 0.62,
                title = "fitted:  a_bar = 2.88, sigma = 0.62"),
  book   = list(a_bar = 1.50, sigma = 1.50,
                title = "book:  a_bar = 1.50, sigma = 1.50")
)

for (nm in names(scenarios))
  scenarios[[nm]]$sim <- simulate_scenario(scenarios[[nm]]$a_bar,
                                           scenarios[[nm]]$sigma)

# --- the figure --------------------------------------------------------------
# Points are per-listing absolute error; the horizontal segments are the mean
# within each review-count band, which is what the comparison rests on.

panel <- function(s) {
  sim   <- s$sim
  edges <- c(0, cumsum(rle(sim$Ni)$lengths))
  mean_by_band <- function(e) tapply(e, sim$Ni, mean)

  plot(1:n_listing, sim$nopool_error, pch = 16, col = rangi2, xaxt = "n",
       xlab = "listing  (sorted by number of reviews)",
       ylab = "absolute error", main = "",
       ylim = c(0, max(sim$nopool_error, sim$partpool_error)))
  mtext(s$title, side = 3, line = 0.8, font = 2, cex = 0.9)
  points(1:n_listing, sim$partpool_error)

  for (k in seq_along(unique(sim$Ni))) {
    x0 <- edges[k] + 0.5
    x1 <- edges[k + 1] + 0.5
    segments(x0, mean_by_band(sim$nopool_error)[k],
             x1, mean_by_band(sim$nopool_error)[k], col = rangi2, lwd = 2)
    segments(x0, mean_by_band(sim$partpool_error)[k],
             x1, mean_by_band(sim$partpool_error)[k], lty = 2, lwd = 2)
    abline(v = x1, lwd = 0.5, col = "grey80")
  }
  axis(1, at = (head(edges, -1) + tail(edges, -1)) / 2,
       labels = paste(unique(sim$Ni), "rev"))
}

fig("08-simulation-error.png", width = 1900, height = 900, {
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 4, 1))
  panel(scenarios$fitted)
  legend("topright", bty = "n", cex = 0.85,
         legend = c("no pooling", "partial pooling"),
         pch = c(16, 1), lty = c(1, 2), lwd = 2, col = c(rangi2, "black"))
  panel(scenarios$book)
  mtext("Simulated listings: error against the known truth",
        side = 3, line = -1.5, outer = TRUE, font = 2, cex = 1.05)
})

# --- the table ---------------------------------------------------------------

summarise <- function(name) {
  sim <- scenarios[[name]]$sim
  by  <- function(e) as.numeric(round(tapply(e, sim$Ni, mean), 4))
  data.frame(scenario         = name,
             reviews          = sort(unique(sim$Ni)),
             complete_pooling = by(sim$complete_error),
             no_pooling       = by(sim$nopool_error),
             partial_pooling  = by(sim$partpool_error))
}

overall <- function(name) {
  sim <- scenarios[[name]]$sim
  sprintf("  %-7s complete %.4f | no pooling %.4f | partial %.4f",
          name, mean(sim$complete_error), mean(sim$nopool_error),
          mean(sim$partpool_error))
}

tbl("07-simulation-error.txt",
    do.call(rbind, lapply(names(scenarios), summarise)),
    header = c(
      "Mean absolute error against the KNOWN true probability",
      sprintf("%d simulated listings, 15 at each of %s reviews",
              n_listing, paste(unique(Ni), collapse = "/")),
      "",
      "overall:",
      overall("fitted"),
      overall("book")))
