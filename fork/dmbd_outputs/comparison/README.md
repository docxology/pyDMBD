# DMBD Dataset Comparison

This directory contains visualizations and analysis comparing the three datasets used for testing the Dynamic Markov Blanket Discovery (DMBD) model:

1. **Forager**: A 5-dimensional dataset from a foraging agent simulation
2. **Synthetic**: A 3-dimensional dataset with designed dependencies
3. **Binary**: A 3-dimensional binary dataset with clear Markov blanket structure

## Files in this Directory

- `dataset_comparison_report.md`: Summary report of the comparison findings
- `dataset_comparison_stats.csv`: Statistical properties of each dataset dimension
- `dataset_comparison_heatmap.png`: Heatmap visualizations of means, standard deviations, and skewness
- `dataset_comparison_distributions.png`: Histograms showing the distribution of values in each dimension
- `dataset_comparison_autocorrelations.png`: Autocorrelation plots showing temporal dependencies
- `*_correlation_matrix.png`: Correlation matrices for each dataset showing relationships between dimensions
- `debug_log.txt`: Log of the comparison process

## Key Insights

### Statistical Properties

The datasets vary significantly in their statistical properties:

- **Forager**: Mixed distributions with some dimensions normalized (mean ≈ 0, std ≈ 1) and others bounded [0,1]
- **Synthetic**: Continuous values with means near 0 and standard deviations around 0.21-0.25
- **Binary**: Binary values (0/1) with means around 0.42-0.47 and standard deviations around 0.49

### Correlations

The correlation matrices reveal different dependency structures:

- **Forager**: Complex correlations between dimensions reflecting the agent's behavior
- **Synthetic**: Controlled correlations based on the designed dependencies
- **Binary**: Clear correlations matching the intended Markov blanket structure

### Temporal Dependencies

The autocorrelation plots show how each dimension depends on its past values:

- **Forager**: Strong temporal dependencies in some dimensions
- **Synthetic**: Moderate temporal dependencies
- **Binary**: Weaker temporal dependencies due to the binary nature of the data

## Implications for DMBD

Despite the differences in complexity and statistical properties, the DMBD model failed to converge on all three datasets. This suggests that the convergence issues are fundamental to the model or its implementation rather than being data-dependent.

The binary dataset's failure is particularly concerning as it represents the simplest possible case with clear dependencies. This points to potential issues in the DMBD update algorithm that need to be addressed in future work.

## Usage

These visualizations can be used to:

1. Understand the characteristics of each dataset
2. Compare the complexity and structure across datasets
3. Identify potential issues that might affect DMBD convergence
4. Guide future improvements to the DMBD model

For more detailed analysis, refer to the main report at `/home/trim/Documents/GitHub/pyDMBD/fork/dmbd_outputs/REPORT.md`. 