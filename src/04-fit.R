# Fit the four models and compare them.
#
# The point of the comparison is the pWAIC column: the multilevel model has
# MORE actual parameters than the no-pooling models and FEWER effective ones.

source("src/02-sample.R")
source("src/03-models.R")

fits <- list(
  complete_pooling = fit_model("complete_pooling"),
  no_pooling       = fit_model("no_pooling"),
  no_pooling_flat  = fit_model("no_pooling_flat"),
  partial_centered = fit_model("partial_centered"),
  partial          = fit_model("partial", control = list(adapt_delta = 0.95))
)

# Under cmdstan the draws live in temporary csv files that are deleted when the
# session ends, so a saved ulam object reloads without its samples. Save the
# posterior itself, which is what the later scripts actually need.
saveRDS(extract.samples(fits$partial), file.path(CACHE, "post-partial.rds"))

cmp <- round(compare(fits$complete_pooling, fits$no_pooling,
                     fits$no_pooling_flat, fits$partial), 2)

# compare() carries the model names as row names, and round() drops the class
# that would print them. Make the name a column so it survives.
cmp <- data.frame(model = sub("^fits\\$", "", rownames(cmp)),
                  cmp, row.names = NULL)

tbl("01-compare.txt", cmp,
    header = c(
      sprintf("WAIC comparison -- %d listings, %d reviews",
              nrow(d), sum(d$number_of_reviews)),
      "",
      "complete_pooling  logit(p) <- a                        1 parameter",
      "no_pooling        a[listing] ~ Normal(0, 1.5)        200 parameters",
      "no_pooling_flat   a[listing] ~ Normal(0, 5)          200 parameters",
      "partial           a[listing] ~ Normal(a_bar, sigma)  202 parameters"))

tbl("02-precis-partial.txt", precis(fits$partial, depth = 1),
    header = "Hyperparameters of the multilevel model (non-centered)")
