# Build the analysis sample and the data list ulam wants.
#
# An overall rating y in [1,5] is read as the fraction of customers who had a
# positive experience, p = (y-1)/4. With N reviews that gives a success count
# P = round(p*N) out of N trials -- the tadpole data of Chapter 13, with
# listings in place of tanks and reviews in place of tadpoles.

source("src/00-setup.R")

# Stratify across review counts. A plain random sample would be swamped by the
# middle of the distribution and would leave too few one-review listings, which
# are precisely the ones partial pooling is supposed to do something about.
set.seed(SEED)
d <- read_csv(file.path(DATA, "ratings.csv"), show_col_types = FALSE) %>%
  mutate(bin = cut(number_of_reviews, BIN_BREAKS, labels = BIN_LABELS)) %>%
  group_by(bin) %>%
  slice_sample(n = N_PER_BIN) %>%
  ungroup() %>%
  arrange(number_of_reviews) %>%
  mutate(P       = round((review_scores_rating - 1) / 4 * number_of_reviews),
         listing = row_number())

dat <- list(P       = d$P,
            N       = d$number_of_reviews,
            listing = d$listing)
