library(tidyverse)
library(rethinking)

# 1. Load and clean data
df <- read_csv("listings.csv") %>%
  select(id, host_id, review_scores_rating, number_of_reviews) %>%
  filter(!is.na(review_scores_rating), number_of_reviews > 0) %>%
  mutate(
    # Standardize ratings for easier prior specification
    rating_std = (review_scores_rating - mean(review_scores_rating)) / sd(review_scores_rating),
    log_reviews = log(number_of_reviews)
  ) %>%
  slice_sample(n = 1000)

# 2. Clean index variables for rethinking
df$listing_idx <- as.integer(as.factor(df$id))
df$host_idx    <- as.integer(as.factor(df$host_id))

# 3. Data list for ulam
dat <- list(
  rating      = df$rating_std,
  n_reviews   = df$number_of_reviews,
  listing_idx = df$listing_idx,
  host_idx    = df$host_idx,
  log_rev     = df$log_reviews - mean(df$log_reviews)
)

# 4. Multilevel Model with Variance Weighted by Review Count
f_rating_shrinkage <- alist(
  # Likelihood: observational variance shrinks with sample size n_reviews
  rating ~ dnorm(mu, sigma / sqrt(n_reviews)),
  
  # Linear model with listing intercept and review count effect
  mu <- a_listing[listing_idx] + b_rev * log_rev,
  
  # Adaptive prior over listings (Partial Pooling)
  a_listing[listing_idx] ~ dnorm(a_bar, sigma_listing),
  
  # Priors
  a_bar ~ dnorm(0, 0.5),
  b_rev ~ dnorm(0, 0.5),
  sigma_listing ~ dexp(1),
  sigma ~ dexp(1)
)

# 5. Fit model
m_rating <- ulam(
  f_rating_shrinkage,
  data = dat,
  chains = 4,
  cores = 4,
  iter = 2000
)

# 6. Extract posterior and compare raw ratings vs shrunken predictions
post <- extract.samples(m_rating)