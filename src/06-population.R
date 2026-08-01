# The population of listings -- the analogue of Figure 13.2.
#
# A multilevel model does not only estimate 200 intercepts. It estimates the
# distribution those intercepts came from, and that is what lets it say
# anything about a listing it has never seen. R code 13.7.

source("src/02-sample.R")
post <- readRDS(file.path(CACHE, "post-partial.rds"))

set.seed(SEED)
sim <- rnorm(8000, post$a_bar, post$sigma)   # 8000 imaginary new listings

fig("05-population.png", width = 1700, height = 850, {
  par(mfrow = c(1, 2), mar = c(4.5, 4.5, 5, 1.5))

  # 100 draws of the population distribution, on the log-odds scale
  plot(NULL, xlim = c(0, 6), ylim = c(0, 0.8), main = "",
       xlab = "log-odds of a positive experience", ylab = "density")
  mtext("Posterior population of listings", side = 3, line = 3, font = 2)
  for (i in 1:100)
    curve(dnorm(x, post$a_bar[i], post$sigma[i]),
          add = TRUE, col = col.alpha("black", 0.2))

  # the same population as probabilities. A symmetric bell on the log-odds
  # scale becomes strongly skewed once squeezed through the inverse logit near
  # the top of its range.
  dens(inv_logit(sim), lwd = 2, adj = 0.1, xlab = "", main = "")
  mtext("Simulated new listings", side = 3, line = 3, font = 2)
  mtext("probability of a positive experience", side = 1, line = 2.4, cex = 0.9)
  stars <- seq(3, 5, 0.5)
  axis(3, at = (stars - 1) / 4, labels = format(stars), cex.axis = 0.75)
  mtext("implied overall rating", side = 3, line = 1.5, cex = 0.75)
})

tbl("04-population.txt",
    precis(data.frame(log_odds = sim,
                      p        = inv_logit(sim),
                      rating   = 1 + 4 * inv_logit(sim)), prob = 0.89),
    header = "8000 listings simulated from the posterior of a_bar and sigma")
