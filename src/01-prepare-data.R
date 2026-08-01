# Reduce the raw Inside Airbnb export to the four columns everything else
# needs. Run once. data/raw/listings.csv is 213 MiB; data/ratings.csv is
# under 2 MB.

source("src/00-setup.R")
library(data.table)

raw <- fread(file.path(RAW, "listings.csv"),
             select = c("id", "host_id",
                        "review_scores_rating", "number_of_reviews")) %>%
  as_tibble() %>%
  mutate(review_scores_rating = as.numeric(review_scores_rating),
         number_of_reviews    = as.integer(number_of_reviews))

# A listing needs at least one review to have a rating, and the rating must be
# on the 1-5 scale. A handful of rows fall below 1 and are dropped.
d <- raw %>%
  filter(!is.na(review_scores_rating),
         number_of_reviews > 0,
         between(review_scores_rating, 1, 5))

write_csv(d, file.path(DATA, "ratings.csv"))

# The table that motivates the whole exercise: as review counts grow the mean
# rating barely moves while the spread collapses. That is sampling noise, not
# real differences in quality.
noise <- d %>%
  mutate(bin = cut(number_of_reviews, BIN_BREAKS, labels = BIN_LABELS)) %>%
  group_by(bin) %>%
  summarise(listings    = n(),
            mean_rating = round(mean(review_scores_rating), 3),
            sd_rating   = round(sd(review_scores_rating), 3),
            .groups = "drop")

tbl("00-noise-by-reviews.txt", as.data.frame(noise),
    header = c("Rating noise by review count -- all usable listings",
               sprintf("%d of %d raw rows usable", nrow(d), nrow(raw))))
