# The four models, as formulas only. Fitting happens in 04-fit.R.
#
# Keeping the formulas together makes the one thing that actually differs
# between them -- what the intercepts are allowed to know about each other --
# easy to read off.

models <- list(

  # Complete pooling. One intercept for every listing at once: the model says
  # all listings are the same, and any difference between them is noise. The
  # opposite extreme from no pooling, and the other thing partial pooling sits
  # between.
  complete_pooling = alist(
    P ~ dbinom(N, p),
    logit(p) <- a,
    a ~ dnorm(0, 1.5)
  ),

  # No pooling. Each listing gets its own intercept under a fixed prior, the
  # book's (R code 13.2). Note this prior is not weak on this scale: ratings
  # sit near 4.79, so logit(p) ~ 2.9, and Normal(0,1.5) pulls every listing
  # toward p = 0.5.
  no_pooling = alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(0, 1.5)
  ),

  # Same, with a genuinely flat prior. A sensitivity check that the comparison
  # is not an artifact of the prior width above.
  no_pooling_flat = alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(0, 5)
  ),

  # Partial pooling, centered -- the prior over intercepts is itself estimated
  # (R code 13.3). Samples the funnel badly; kept to show why.
  partial_centered = alist(
    P ~ dbinom(N, p),
    logit(p) <- a[listing],
    a[listing] ~ dnorm(a_bar, sigma),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1)
  ),

  # The same model non-centered (R code 13.22): sample standardized offsets z
  # and rebuild a = a_bar + z*sigma. Identical posterior, far better geometry.
  partial = alist(
    P ~ dbinom(N, p),
    logit(p) <- a_bar + z[listing] * sigma,
    z[listing] ~ dnorm(0, 1),
    a_bar ~ dnorm(0, 1.5),
    sigma ~ dexp(1),
    gq> vector[listing]:a <<- a_bar + z * sigma
  )
)
