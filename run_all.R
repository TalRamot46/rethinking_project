# Run the whole analysis. From the repository root:
#
#   Rscript run_all.R
#
# 01-prepare-data.R needs listings.csv in the working directory and is skipped
# once data/ratings.csv exists. Everything after it runs in a few minutes; the
# twelve fixed-sigma fits in step 08 are cached in .cache/.

source("src/00-setup.R")

if (!file.exists(file.path(DATA, "ratings.csv"))) {
  source("src/01-prepare-data.R")
} else {
  cat("data/ratings.csv exists -- skipping 01-prepare-data.R\n")
}

source("src/04-fit.R")          # also sources 02-sample.R and 03-models.R
source("src/05-diagnostics.R")  # reuses the fits from 04
source("src/06-population.R")
source("src/07-shrinkage.R")
source("src/08-sigma-grid.R")
source("src/09-sigma-figure.R")
source("src/10-simulation.R")

cat("\nDone. Figures in", FIGS, "and tables in", TABLES, "\n")
