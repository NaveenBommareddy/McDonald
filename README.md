McDonald's Market Segmentation - Python Implementation

This project replicates the McDonald's fast food market segmentation case study originally presented in the book Market Segmentation Analysis by Dolnicar, Grün, and Leisch. It uses Python libraries to execute all key segmentation steps.

Overview

The goal is to segment consumers based on their perception of McDonald's brand image using clustering and classification techniques. The pipeline includes:

Data preprocessing

Dimensionality reduction (PCA)

Clustering (K-Means, Hierarchical, Gaussian Mixture Models)

Segment profiling

Segment prediction (Logistic Regression, Decision Trees)

Requirements

Make sure the following Python packages are installed:

pip install pandas numpy matplotlib seaborn scikit-learn scipy

Dataset

This project expects a CSV file named mcdonalds.csv, which includes:

11 binary (Yes/No) brand image attributes

A numeric "Like" score column for McDonald's

Example columns:

Yummy,Convenient,Spicy,Fattening,Greasy,Fast,Cheap,Tasty,Expensive,Healthy,Disgusting,Like

How to Use

Place your dataset as mcdonalds.csv in the same directory.

Run the Python script (mcdonalds_segmentation_python.py).

Outputs include:

PCA plots

Dendrogram for hierarchical clustering

K-Means cluster assignments

GMM segment assignments

Logistic regression and decision tree models

Segment-wise box plots for Like scores

Output Summary

Visual PCA mapping to reduce dimensionality

Segment membership for each consumer

Tree-based explanation of segment drivers

Performance evaluation using silhouette score

Notes

This project is a Python adaptation. Some visual output may differ slightly from the R version due to library defaults.

For advanced analysis, extend this with other libraries like plotly, xgboost, or dash for interactivity.

License

MIT License. Attribution encouraged for academic or educational reuse.
