# Shared setup. Every other script starts by sourcing this file.

library(tidyverse)
library(rethinking)

# --- knobs -------------------------------------------------------------------

SEED       <- 13
N_PER_BIN  <- 40                                   # 5 bins x 40 = 200 listings
BIN_BREAKS <- c(0, 2, 5, 10, 50, Inf)
BIN_LABELS <- c("1-2", "3-5", "6-10", "11-50", "50+")
CHAINS     <- 4

# --- paths -------------------------------------------------------------------

RAW    <- "data/raw"  # untouched Inside Airbnb exports (gitignored)
DATA   <- "data"      # derived data                    (gitignored)
CACHE  <- ".cache"    # fitted posteriors               (gitignored)
FIGS   <- "figures"   # committed
TABLES <- "tables"    # committed

for (p in c(DATA, RAW, CACHE, FIGS, TABLES))
  dir.create(p, showWarnings = FALSE, recursive = TRUE)

# cmdstan is several times faster than rstan here; fall back if it is absent
set_ulam_cmdstan(
  requireNamespace("cmdstanr", quietly = TRUE) &&
    !is.null(tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL))
)

# --- helpers -----------------------------------------------------------------

#' Fit a model from R/03-models.R by name.
#' Every fit is seeded, so reruns reproduce the numbers quoted in the essay.
fit_model <- function(name, data = dat, ...) {
  ulam(models[[name]], data = data, chains = CHAINS, cores = CHAINS,
       log_lik = TRUE, seed = SEED, ...)
}

#' Write a plot to figures/ and nothing else.
fig <- function(name, expr, width = 1600, height = 900) {
  png(file.path(FIGS, name), width = width, height = height, res = 190)
  on.exit(dev.off())
  force(expr)
  invisible(name)
}

#' Print a table to the console and save the same text to tables/.
tbl <- function(name, x, header = NULL) {
  # plain data frames carry no meaningful row names here; precis objects do
  body  <- if (class(x)[1] == "data.frame") {
    capture.output(print(x, row.names = FALSE))
  } else {
    capture.output(print(x))
  }
  lines <- c(if (!is.null(header)) c(header, ""), body)
  writeLines(lines, file.path(TABLES, name))
  cat(lines, sep = "\n")
  cat("\n")
  invisible(x)
}

#' log(mean(exp(x))), without overflowing for very small densities.
log_mean_exp <- function(x) {
  m <- max(x)
  m + log(mean(exp(x - m)))
}
