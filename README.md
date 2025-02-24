## Summary
This project involves implementing the Sessa Empirical Estimator (SEE) for estimating prescription durations in a pharmacoepidemiological context. Originally written in R, the SEE will be ported to Python, and applied to a chosen dataset. The standard SEE approach uses K-Means clustering to group similar prescription intervals, but this project will also explore an alternative clustering algorithm to see if it produces different insights. Finally, the results and performance from both clustering methods will be compared, highlighting any differences in exposure classification or prescription duration estimates.

## Contributors;
- Pantino, Prince Isaac
- Singh, Fitzsixto Angelo

## Objectives

#### 1. Replicate the SEE Methodology:
- Convert the SEE R code into Python and validate its functionality on a dataset.
- Implement K-Means clustering to estimate prescription durations and identify exposure windows.

#### 2. Explore an Alternative Clustering Algorithm:
- Substitute K-Means with a different clustering approach (e.g., DBSCAN, Hierarchical Clustering, or GMM).
- Compare the resulting clusters, prescription durations, and exposure periods to those from K-Means.

#### 3. Evaluate and Compare Results:
- Generate metrics and visualizations to illustrate how each clustering algorithm performs.
- Discuss strengths, weaknesses, and potential applications of each approach.

#### 4. Document Findings and Insights:
- Summarize the differences in performance and outcomes between K-Means-based SEE and the alternative method.
- Provide clear recommendations on how clustering choice can affect pharmacoepidemiological analyses.