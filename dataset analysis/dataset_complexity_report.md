# Dataset Complexity Analysis Report

Generated on: 2025-07-19 21:38:19

## Executive Summary

| Dataset | Complexity Level | Separation Ratio | Silhouette Score | Recommended Strategy |
|---------|------------------|------------------|------------------|---------------------|
| dtd | HIGH | 0.809 | -0.191 | Self-Distillation |
| flowers | LOW | 2.478 | 0.597 | Standard Training |

## Detailed Analysis

### DTD

**Basic Statistics:**
- Dataset Size: 3,760 samples
- Number of Classes: 47
- Feature Dimension: 512

**Complexity Metrics:**
- Intrinsic Dimensionality (95%): 0.469
- Complexity Score: 1.236
- Separation Ratio: 0.809
- Silhouette Score: -0.191

**Training Recommendations:**
- Hard Dataset - Use Self-Distillation + Center Optimization

### FLOWERS

**Basic Statistics:**
- Dataset Size: 4,093 samples
- Number of Classes: 102
- Feature Dimension: 512

**Complexity Metrics:**
- Intrinsic Dimensionality (95%): 0.320
- Complexity Score: 0.403
- Separation Ratio: 2.478
- Silhouette Score: 0.597

**Training Recommendations:**
- Relatively Simple Dataset
-   - Current Config Should Be Sufficient
-   - Consider Lower Learning Rate

