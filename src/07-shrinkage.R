# Shrinkage, in the style of Figure 13.1.
#
# Listings sorted by review count; the empirical rating and the posterior mean,
# joined by a segment so the size and direction of the move are visible.

source("src/02-sample.R")
post <- readRDS(file.path(CACHE, "post-partial.rds"))

d$p_emp <- (d$review_scores_rating - 1) / 4
d$p_est <- apply(inv_logit(post$a), 2, mean)
p_bar   <- mean(inv_logit(post$a_bar))

# 200 listings is too dense to read once each point is joined to its partner.
# Thin to 12 per band for the figure only; the model is still fitted on all 200.
set.seed(SEED)
shown <- d %>%
  group_by(bin) %>%
  slice_sample(n = 12) %>%
  ungroup() %>%
  arrange(number_of_reviews) %>%
  mutate(x = row_number())

edges   <- which(diff(as.integer(shown$bin)) != 0) + 0.5
centres <- tapply(shown$x, shown$bin, mean)

fig("06-shrinkage.png", width = 1800, height = 1100, {
  par(mar = c(4.5, 4.5, 4, 4.5))
  plot(shown$x, shown$p_emp, ylim = c(min(shown$p_emp) - 0.06, 1.01),
       pch = 16, col = rangi2, xaxt = "n", main = "",
       xlab = "listing  (sorted by number of reviews)",
       ylab = "proportion of positive experiences")
  mtext("Partial pooling shrinks the listings with least evidence",
        side = 3, line = 2.2, font = 2, cex = 1.05)

  segments(shown$x, shown$p_emp, shown$x, shown$p_est, col = "grey60", lwd = 0.8)
  points(shown$x, shown$p_emp, pch = 16, col = rangi2)
  points(shown$x, shown$p_est, pch = 1)

  abline(h = p_bar, lty = 2)                       # what everything is pulled to
  abline(v = edges, lwd = 0.5, col = "grey70")
  axis(1, at = round(centres), labels = round(centres))
  mtext(paste(levels(shown$bin), "rev"), side = 3, at = centres,
        line = 0.3, cex = 0.7, col = "grey30")

  stars <- seq(3, 5, 0.5)                          # same axis in rating units
  axis(4, at = (stars - 1) / 4, labels = format(stars))
  mtext("overall rating (1-5)", side = 4, line = 2.8, cex = 0.9)

  legend("bottomright", bty = "n", cex = 0.8,
         legend = c("empirical rating", "posterior mean", "population mean"),
         pch = c(16, 1, NA), lty = c(NA, NA, 2),
         col = c(rangi2, "black", "black"))
})

# Two measures, because the obvious one misleads on its own.
#
# shift is not monotone in the review count: a one-review listing is almost
# always rated exactly 5.0, which already sits close to the population mean, so
# it has little distance to travel however hard it is pulled.
#
# shrink_factor compares the spread before and after pooling. It is a ratio of
# means rather than a mean of ratios, which matters because the per-listing
# ratio has a near-zero denominator for exactly those low-N listings.
tbl("05-shrinkage.txt",
    d %>%
      mutate(shift    = abs(p_est - p_emp) * 4,
             dist_emp = abs(p_emp - p_bar) * 4,
             dist_est = abs(p_est - p_bar) * 4) %>%
      group_by(bin) %>%
      summarise(listings      = n(),
                mean_shift    = round(mean(shift), 3),
                spread_before = round(mean(dist_emp), 3),
                spread_after  = round(mean(dist_est), 3),
                shrink_factor = round(1 - mean(dist_est) / mean(dist_emp), 3),
                .groups = "drop") %>%
      as.data.frame(),
    header = c("Shrinkage by review count -- all units in rating points",
               sprintf("population mean rating: %.3f", 1 + 4 * p_bar)))
