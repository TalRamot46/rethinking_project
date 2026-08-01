# part1_multilevel.R -- section 2.
#
# No pooling vs partial pooling, and the WAIC comparison. The point of the
# comparison is the pWAIC column: the multilevel model has MORE actual
# parameters and FEWER effective ones.
#
# Run prep_data.R first.

source("part1_common.R")

cat("listings:", nrow(d),
    "| reviews:", sum(d$number_of_reviews),
    "| median N:", median(d$number_of_reviews), "\n")

saveRDS(d, "outputs/part1/fits/d.rds")

# 1. No pooling ---------------------------------------------------------------
# Each listing gets its own intercept with a fixed prior. The prior is the
# book's (R code 13.2). Note that a rating of 4.78 means p = 0.945, i.e.
# logit(p) = 2.9 -- so Normal(0,1.5) is not weak on this scale. It pulls every
# listing toward p = 0.5. m13.1b checks that the comparison is not an artifact
# of that.

m13.1 <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(0, 1.5)
  ),
  data = dat, chains = 4, cores = 4, log_lik = TRUE, seed = 13
)

m13.1b <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(0, 5)
  ),
  data = dat, chains = 4, cores = 4, log_lik = TRUE, seed = 13
)

# 2. Partial pooling ----------------------------------------------------------
# The prior over intercepts is itself estimated. Centered form first, as in
# R code 13.3.

m13.2c <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(a_bar, sigma),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1)
  ),
  data = dat, chains = 4, cores = 4, log_lik = TRUE, seed = 13
)

# The centered version samples the funnel badly: E-BFMI below 0.3 on every
# chain and an effective sample size for sigma in the low hundreds. Everything
# downstream depends on the sigma posterior, so refit non-centered
# (R code 13.22). Same model, better geometry.

m13.2 <- ulam(
  alist(
    P ~ dbinom(N, p),
    logit(p) <- a_bar + z[listing] * sigma,
    z[listing] ~ dnorm(0, 1),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1),
    gq> vector[listing]:a <<- a_bar + z * sigma
  ),
  data = dat, chains = 4, cores = 4, log_lik = TRUE, seed = 13,
  control = list(adapt_delta = 0.95)
)

# Save the POSTERIOR, not the fit object. Under cmdstan the draws live in
# temporary csv files that are deleted when the R session ends, so a saved
# ulam object reloads without its samples and extract.samples() fails on it.
saveRDS(extract.samples(m13.2), "outputs/part1/fits/post_m13.2.rds")

# 3. Comparison ---------------------------------------------------------------

cmp <- compare(m13.1, m13.1b, m13.2)

sink("outputs/part1/10_compare.txt")
cat("WAIC comparison --", nrow(d), "listings\n\n")
cat("m13.1  no pooling,  a[listing] ~ Normal(0, 1.5)   [book prior, R code 13.2]\n")
cat("m13.1b no pooling,  a[listing] ~ Normal(0, 5)     [sensitivity: flat]\n")
cat("m13.2  multilevel,  a[listing] ~ Normal(a_bar, sigma), non-centered\n\n")
print(round(cmp, 2))
cat("\nactual number of parameters:\n")
cat("  m13.1  =", nrow(d), "\n")
cat("  m13.1b =", nrow(d), "\n")
cat("  m13.2  =", nrow(d) + 2, "\n")
cat("\nThe multilevel model has the MOST actual parameters and the FEWEST\n")
cat("effective ones. That is the whole lesson of the chapter.\n")
sink()

print(round(cmp, 2))

# 4. Diagnostics: centered vs non-centered ------------------------------------

diag_row <- function(m, nm) {
  pr <- precis(m, depth = 1)
  data.frame(
    model      = nm,
    divergent  = sum(divergent(m)),
    max_rhat   = round(max(pr$rhat,     na.rm = TRUE), 3),
    min_ess    = round(min(pr$ess_bulk, na.rm = TRUE), 0)
  )
}

diags <- rbind(
  diag_row(m13.2c, "m13.2c centered"),
  diag_row(m13.2,  "m13.2  non-centered")
)

sink("outputs/part1/11_precis_m13.2.txt")
cat("Centered vs non-centered -- same model, different geometry\n")
cat("(rhat and ess over the hyperparameters a_bar and sigma)\n\n")
print(diags, row.names = FALSE)
cat("\n\nm13.2c centered -- hyperparameters\n\n")
print(precis(m13.2c, depth = 1))
cat("\n\nm13.2 non-centered -- hyperparameters\n\n")
print(precis(m13.2, depth = 1))
cat("\n\nm13.2 non-centered -- all parameters\n\n")
print(precis(m13.2, depth = 2, pars = "a"))
sink()

print(diags, row.names = FALSE)
print(precis(m13.2, depth = 1))
