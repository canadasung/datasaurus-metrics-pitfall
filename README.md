# datasaurus-metrics-pitfall

### The Plots
![Datasaurus Dozen](output/01-dataset-facets.png)

### The metrics of the datasets for each plot
![Summary Table](output/00-summary-table.png)

---

Different plots, same metrics, WHY?

I find this discrepancy interesting that we could be tricked by the data when they have the same metrics such as mean, median, standard deviation, but the plots for the data would be dramatically different. This project aims to explore this phenomenon and understand the implications for predictive modeling. This is also a reminder to myself to be aware of similar pitfall.

This is a learning project that replicates the analysis from Julia Silge's blog post
["Multiclass predictive modeling for the Datasaurus Dozen"](https://juliasilge.com/blog/datasaurus-multiclass/). 

## About

The Datasaurus Dozen is a collection of 13 datasets that share nearly
identical summary statistics (mean, standard deviation, correlation) but
look completely different when plotted. This project replicates a random
forest classifier that attempts to predict which of the 13 datasets a given
point belongs to, using `tidymodels`.

## Setup

This project uses [renv](https://rstudio.github.io/renv/) for dependency
management.

1. Clone the repository
2. Open the project in Positron (or RStudio)
3. Run the following in the R console to install the required packages:

   ```r
   renv::restore()
   ```

## Repository structure

- `R/` -- helper functions used in the analysis
- `analysis/` -- main analysis script(s) replicating the blog post
- `output/` -- saved plots and other generated artifacts
- `renv.lock` -- pinned package versions for reproducibility

## Reproducing the analysis

Run the script(s) in `analysis/` from the project root, for example:

```r
source("analysis/01-datasaurus-multiclass.R")
```
