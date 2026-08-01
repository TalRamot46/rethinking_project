# part1_common.R -- sourced by every part 1 script.
#
# Builds the subsample and the data list. Kept in one place so that the models,
# the sigma grid and the figures all use the identical 200 listings.
#
# The framing: an overall rating y in [1,5] is read as the fraction of customers
# who had a positive experience, p = (y-1)/4. With N reviews that gives a
# success count P = round(p*N) out of N trials -- the tadpole survival data of
# Chapter 13, with listings for tanks and reviews for tadpoles.

library(tidyverse)
library(rethinking)

dir.create("outputs/part1/fits", showWarnings = FALSE, recursive = TRUE)

# cmdstan is much faster than rstan here; fall back if it is not installed
if (requireNamespace("cmdstanr", quietly = TRUE) &&
    !is.null(tryCatch(cmdstanr::cmdstan_version(), error = function(e) NULL))) {
  set_ulam_cmdstan(TRUE)
} else {
  set_ulam_cmdstan(FALSE)
}

d <- read_csv("data/ratings_small.csv", show_col_types = FALSE)

# Stratify the subsample across review counts. A plain random sample would be
# swamped by the middle of the distribution and leave too few low-N listings,
# which are the ones that show shrinkage.
set.seed(13)
d <- d %>%
  mutate(bin = cut(number_of_reviews,
                   breaks = c(0, 2, 5, 10, 50, Inf),
                   labels = c("1-2", "3-5", "6-10", "11-50", "50+"))) %>%
  group_by(bin) %>%
  slice_sample(n = 40) %>%
  ungroup() %>%
  arrange(number_of_reviews)

d$P       <- round((d$review_scores_rating - 1) / 4 * d$number_of_reviews)
d$listing <- 1:nrow(d)

dat <- list(
  P       = d$P,
  N       = d$number_of_reviews,
  listing = d$listing
)