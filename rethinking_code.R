library(tidyverse)
library(rethinking)

# 1. Load and clean data
df <- read_csv("listings.csv") %>%
  # Select required columns (review_scores_rating exists in this detailed dataset)
  select(host_id, price, review_scores_rating, number_of_reviews) %>%
  filter(!is.na(price), !is.na(review_scores_rating), number_of_reviews > 0) %>%
  mutate(
    # Clean price: remove '$' and ',' then parse to numeric
    price_clean = parse_number(as.character(price)),
    log_price   = log(price_clean)
  ) %>%
  filter(price_clean > 0) %>%
  slice_sample(n = 2000)

# 2. Rethinking requires CLEAN integer index numbers (1, 2, 3, ... N)
df$host_idx <- as.integer(as.factor(df$host_id))

# 3. Create list for ulam
dat <- list(
  log_price = df$log_price,
  host_idx  = df$host_idx
)

# 4. Chapter 13 Multilevel Model Formula
f_multilevel <- alist(
  log_price ~ dnorm(mu, sigma),
  
  # Linear model: unique intercept 'a' per host
  mu <- a[host_idx],
  
  # Adaptive Prior (Varying Intercepts / Partial Pooling)
  a[host_idx] ~ dnorm(a_bar, sigma_host),
  
  # Hyper-priors
  a_bar ~ dnorm(4.5, 1),
  sigma_host ~ dexp(1),
  
  # Residual variation
  sigma ~ dexp(1)
)

# 5. Fit model via ulam (Stan HMC engine)
m_rethinking <- ulam(
  f_multilevel,
  data = dat,
  chains = 4,
  cores = 4,
  iter = 2000
)

# 6. View estimates (depth = 2 prints varying host intercepts alongside a_bar and sigma)
sink("precis_output.txt")
print(precis(m_rethinking, depth = 2)) # or depth = 1
sink()