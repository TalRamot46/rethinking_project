# Two stacked panels sharing the sigma axis.
#
# Deliberately not a dual axis: the heights of an elpd curve and a posterior
# density are not comparable, and one frame would invite exactly that
# comparison. The agreement to look for is horizontal, not vertical.

source("src/00-setup.R")

g    <- readRDS(file.path(CACHE, "sigma-grid.rds"))
peak <- function(y) g$sigma_grid[which.max(y)]
best <- peak(g$loco)
blue <- "steelblue"

fig("07-sigma-agreement.png", width = 1600, height = 1500, {
  par(mfrow = c(2, 1), mar = c(2.5, 4.5, 2.5, 1.5), oma = c(2.5, 0, 0, 0))
  xlim <- range(g$sigma_grid)

  # top: the cross-validation curves. The direct computation is the one to
  # believe; PSIS and WAIC are shown because their disagreement is the point.
  plot(g$sigma_grid, g$loco, type = "b", pch = 17, log = "x", xlim = xlim,
       col = blue, lwd = 2, ylim = range(c(g$psis, g$waic, g$loco)),
       xlab = "", ylab = "elpd  (higher = better)",
       main = "Out-of-sample score of the fixed-sigma models")
  lines(g$sigma_grid, g$psis, type = "b", pch = 16, lty = 2, col = "grey30")
  lines(g$sigma_grid, g$waic, type = "b", pch = 1,  lty = 3, col = "grey55")
  abline(v = best, lty = 2, col = blue, lwd = 2)
  legend("bottom", bty = "n", cex = 0.8, ncol = 3,
         legend = c("direct leave-one-out", "PSIS", "WAIC"),
         pch = c(17, 16, 1), lty = c(1, 2, 3), lwd = c(2, 1, 1),
         col = c(blue, "grey30", "grey55"))

  # bottom: the posterior it is supposed to agree with
  plot(g$dens, log = "x", xlim = xlim, xlab = "", ylab = "density",
       main = "Posterior of sigma in the model with sigma ~ dexp(1)")
  inside <- g$dens$x > g$pi[1] & g$dens$x < g$pi[2]
  polygon(c(g$pi[1], g$dens$x[inside], g$pi[2]),
          c(0, g$dens$y[inside], 0),
          col = adjustcolor(blue, alpha.f = 0.25), border = NA)
  abline(v = g$mode, lty = 1)
  abline(v = best, lty = 2, col = blue, lwd = 2)
  legend("topright", bty = "n", cex = 0.8,
         legend = c("posterior mode", "CV-optimal sigma", "89% interval"),
         lty = c(1, 2, NA), lwd = c(1, 2, NA), pch = c(NA, NA, 15),
         col = c("black", blue, adjustcolor(blue, alpha.f = 0.4)))

  mtext("sigma  (log scale)", side = 1, outer = TRUE, line = 1)
})
