library(tidyverse)
library(brms)

# 1. Load and clean data
# Make sure "listings.csv" is in your working directory, or specify the full path
df <- read_csv("listings.csv") %>%
  select(host_profile_id, review_scores_rating, number_of_reviews) %>%
  filter(!is.na(review_scores_rating), number_of_reviews > 0) %>%
  # Ensure host_profile_id is treated as a categorical factor for varying intercepts
  mutate(host_profile_id = as.factor(host_profile_id)) %>%
  # Filter down to top N rows to keep sampling quick
  slice_sample(n = 2000)

# 2. Set Bayesian Priors
# Chapter 13 focuses on partial pooling via varying intercepts (1 | host_id)
priors <- c(
  prior(normal(4.5, 0.5), class = Intercept), # Prior on average rating
  prior(exponential(1), class = sd),           # Prior on standard deviation across hosts
  prior(exponential(1), class = sigma)        # Prior on residual variation within hosts
)

# 3. Fit Varying Intercept Model via brms (uses Stan backend under the hood)
m_partial_pooling <- brm(
  formula = review_scores_rating ~ 1 + (1 | host_profile_id),
  data = df,
  family = gaussian(),
  prior = priors,
  chains = 4,
  cores = 4,
  iter = 2000,
  warmup = 1000
)

# 4. View Model Summary
summary(m_partial_pooling)