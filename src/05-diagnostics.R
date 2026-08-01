# Centered vs non-centered: same model, different geometry.
#
# Neither version produces divergent transitions on this data set, so the
# warning usually associated with the funnel never fires. The funnel shows up
# anyway -- in the energy diagnostic and in the effective sample size for
# sigma, which is the parameter everything downstream depends on.

if (!exists("fits")) source("src/04-fit.R")

ebfmi <- function(m) {
  d <- tryCatch(m@cstanfit$diagnostic_summary(quiet = TRUE),
                error = function(e) NULL)
  if (is.null(d)) NA_real_ else round(min(d$ebfmi), 3)
}

diagnose <- function(m, label) {
  p <- precis(m, depth = 1)
  data.frame(model     = label,
             divergent = sum(divergent(m)),
             min_ebfmi = ebfmi(m),
             max_rhat  = round(max(p$rhat, na.rm = TRUE), 3),
             min_ess   = round(min(p$ess_bulk, na.rm = TRUE)))
}

tbl("03-diagnostics.txt",
    rbind(diagnose(fits$partial_centered, "centered"),
          diagnose(fits$partial,          "non-centered")),
    header = c("Sampler diagnostics over a_bar and sigma",
               "E-BFMI below 0.3 is a warning; n_eff is effectively",
               "independent draws, not iterations."))

# Trace plots. A healthy trace is a fuzzy caterpillar: chains in the same band,
# wiggling fast, none sticking. Warmup is excluded -- the first iterations
# shoot far above the posterior and would compress the axis so much that the
# difference between the two parameterizations became invisible.
# n_cols = 2 stops the default three-wide grid leaving a third of the figure
# empty.
trace_pars <- c("a_bar", "sigma")

fig("01-trace-centered.png", height = 620,
    traceplot(fits$partial_centered, pars = trace_pars,
              window = c(501, 1000), n_cols = 2))

fig("02-trace-noncentered.png", height = 620,
    traceplot(fits$partial, pars = trace_pars,
              window = c(501, 1000), n_cols = 2))

# Trace rank plots. Trace plots get hard to read once four chains overlap. A
# trank plot ranks all samples from all chains together and histograms the
# ranks per chain: if every chain explores the same distribution, each should
# hold about its fair share of every rank, so the histograms overlap and stay
# flat.
fig("03-trank-centered.png", height = 620,
    trankplot(fits$partial_centered, pars = trace_pars, n_cols = 2))

fig("04-trank-noncentered.png", height = 620,
    trankplot(fits$partial, pars = trace_pars, n_cols = 2))
