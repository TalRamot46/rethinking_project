# part1_shrinkage.R -- section 6.
#
# Shrinkage, in the style of Figure 13.1. Listings on the horizontal axis
# sorted by number of reviews; the empirical rating and the multilevel
# posterior mean on the vertical.
#
# Run part1_multilevel.R first.

source("part1_common.R")

post <- readRDS("outputs/part1/fits/post_m13.2.rds")

d$p_emp <- (d$review_scores_rating - 1) / 4
d$p_est <- apply(inv_logit(post$a), 2, mean)

# 200 listings is too dense to read once each point is joined to its partner.
# Thin to 12 per review-count bin for the figure only; the model is still the
# one fitted on all 200.
set.seed(13)
p <- d %>%
  group_by(bin) %>%
  slice_sample(n = 12) %>%
  ungroup() %>%
  arrange(number_of_reviews)

p$x <- 1:nrow(p)

# bin boundaries and label positions
edges  <- which(diff(as.integer(p$bin)) != 0) + 0.5
centres <- tapply(p$x, p$bin, mean)

p_bar <- mean(inv_logit(post$a_bar))

png("outputs/part1/fig2_shrinkage.png", width = 1800, height = 1100, res = 200)
par(mar = c(4.5, 4.5, 4, 4.5))

# The book plots the full 0-1 range, but no listing here falls below about 0.5
# and half the panel would be empty. Zoom to the data.
ylim <- c(min(p$p_emp) - 0.06, 1.01)

plot(p$x, p$p_emp, ylim = ylim, pch = 16, col = rangi2, xaxt = "n",
     xlab = "listing  (sorted by number of reviews)",
     ylab = "proportion of positive experiences",
     main = "")
mtext("Partial pooling shrinks the listings with least evidence",
      side = 3, line = 2.2, font = 2, cex = 1.05)

# each listing's move from empirical to posterior mean
segments(p$x, p$p_emp, p$x, p$p_est, col = "grey60", lwd = 0.8)
points(p$x, p$p_emp, pch = 16, col = rangi2)
points(p$x, p$p_est, pch = 1)

# the population mean everything is pulled toward
abline(h = p_bar, lty = 2)

# dividers between review-count groups, labelled in the top margin so they
# cannot collide with the legend
abline(v = edges, lwd = 0.5, col = "grey70")
axis(1, at = round(centres), labels = round(centres))
mtext(paste(levels(p$bin), "rev"), side = 3, at = centres, line = 0.3,
      cex = 0.7, col = "grey30")

# same axis in rating units, since that is what a reader recognises. Not a
# second variable -- the same quantity in the units the data came in.
rt <- seq(3, 5, 0.5)
axis(4, at = (rt - 1) / 4, labels = format(rt))
mtext("overall rating (1-5)", side = 4, line = 2.8, cex = 0.9)

legend("bottomright", bty = "n", cex = 0.8, bg = "white",
       legend = c("empirical rating", "posterior mean (m13.2)",
                  "population mean"),
       pch = c(16, 1, NA), lty = c(NA, NA, 2),
       col = c(rangi2, "black", "black"))

dev.off()

cat("wrote outputs/part1/fig2_shrinkage.png\n")

# How far did pooling move each listing, by review count? ---------------------

# Three measures, because the obvious one is misleading on its own.
#
# shift -- how far the estimate moved, in rating points. NOT monotone in the
#   review count: a listing with one review is almost always rated 5.0, and 5.0
#   is only 0.2 rating points from the population mean of 4.79, so there is
#   barely any distance to travel however hard it is pulled.
#
# spread_emp / spread_est -- mean distance from the population mean, before and
#   after pooling. This is the honest version of the same idea, and it needs no
#   division by a per-listing quantity that can be near zero.
#
# shrink_factor = 1 - spread_est/spread_emp -- the fraction of the population
#   spread that pooling removes. A ratio of means, not a mean of ratios: the
#   latter is unstable here precisely because most low-N listings sit a hair
#   above the population mean.

shrink_tbl <- d %>%
  mutate(
    shift    = abs(p_est - p_emp) * 4,
    dist_emp = abs(p_emp - p_bar) * 4,
    dist_est = abs(p_est - p_bar) * 4
  ) %>%
  group_by(bin) %>%
  summarise(
    listings      = n(),
    mean_shift    = round(mean(shift), 3),
    max_shift     = round(max(shift), 3),
    spread_emp    = round(mean(dist_emp), 3),
    spread_est    = round(mean(dist_est), 3),
    shrink_factor = round(1 - mean(dist_est) / mean(dist_emp), 3),
    .groups = "drop"
  )

sink("outputs/part1/30_shrinkage_by_n.txt")
cat("Shrinkage by review count -- all units in rating points\n\n")
cat("mean_shift / max_shift : distance the estimate moved\n")
cat("spread_emp             : mean distance from the population mean, raw\n")
cat("spread_est             : mean distance from the population mean, pooled\n")
cat("shrink_factor          : 1 - spread_est/spread_emp\n\n")
print(as.data.frame(shrink_tbl), row.names = FALSE)
cat("\npopulation mean rating implied by a_bar:", round(1 + 4 * p_bar, 3), "\n")
cat("\nmean_shift is not monotone in the review count, and that is not a bug.\n")
cat("A one-review listing is nearly always rated exactly 5.0, which already\n")
cat("sits close to the population mean, so it has little distance to move\n")
cat("even though nearly all of that distance is taken away from it.\n")
cat("shrink_factor is the measure that answers the intended question.\n")
sink()

print(as.data.frame(shrink_tbl), row.names = FALSE)
