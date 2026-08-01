# prep_data.R -- run once.
#
# listings.csv is 222 MB. Everything downstream needs four columns. This reads
# the big file once, filters, and writes data/ratings_small.csv (a few hundred
# KB) plus the table that motivates the whole exercise.

library(tidyverse)
library(data.table)

dir.create("data", showWarnings = FALSE)
dir.create("outputs/part1", showWarnings = FALSE, recursive = TRUE)

# 1. Read only the columns we need -------------------------------------------

d <- fread("listings.csv",
           select = c("id", "host_id", "review_scores_rating", "number_of_reviews")) %>%
  as_tibble() %>%
  mutate(
    review_scores_rating = as.numeric(review_scores_rating),
    number_of_reviews    = as.integer(number_of_reviews)
  )

cat("rows in listings.csv:", nrow(d), "\n")

# 2. Filter -------------------------------------------------------------------
# A listing needs at least one review to have a rating, and the rating must be
# on the 1-5 scale. A handful of rows fall below 1; they are dropped rather
# than clamped, and the count is reported.

n_low <- sum(d$review_scores_rating < 1, na.rm = TRUE)

d <- d %>%
  filter(!is.na(review_scores_rating),
         number_of_reviews > 0,
         review_scores_rating >= 1,
         review_scores_rating <= 5)

cat("dropped for rating < 1:", n_low, "\n")
cat("usable listings:", nrow(d), "\n")

# 3. The motivating table -----------------------------------------------------
# Why the raw average cannot be taken at face value: the spread of ratings
# collapses as review counts grow, while the mean barely moves. That is the
# signature of sampling noise, not of real quality differences.

noise_tbl <- d %>%
  mutate(bin = cut(number_of_reviews,
                   breaks = c(0, 2, 5, 10, 50, Inf),
                   labels = c("1-2", "3-5", "6-10", "11-50", "50+"))) %>%
  group_by(bin) %>%
  summarise(
    listings    = n(),
    mean_rating = round(mean(review_scores_rating), 3),
    sd_rating   = round(sd(review_scores_rating), 3),
    .groups = "drop"
  )

print(noise_tbl)

sink("outputs/part1/00_noise_by_reviews.txt")
cat("Rating noise by review count -- all usable listings\n\n")
print(as.data.frame(noise_tbl))
cat("\nlistings with exactly 1 review:", sum(d$number_of_reviews == 1), "\n")
cat("  of those, rated exactly 5.0:",
    sum(d$number_of_reviews == 1 & d$review_scores_rating == 5), "\n")
sink()

# 4. Write the small file -----------------------------------------------------

write_csv(d, "data/ratings_small.csv")
cat("wrote data/ratings_small.csv\n")
