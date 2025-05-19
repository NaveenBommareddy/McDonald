MCDONALD'S MARKET SEGMENTATION - PYTHON IMPLEMENTATION:

This project replicates the McDonald's fast food market segmentation case study originally presented in the book Market Segmentation Analysis by Dolnicar, Grün, and Leisch. It uses Python libraries to execute all key segmentation steps.
Overview
The goal is to segment consumers based on their perception of McDonald's brand image using clustering and classification techniques. The pipeline includes:
•	Data preprocessing
•	Dimensionality reduction (PCA)
•	Clustering (K-Means, Hierarchical, Gaussian Mixture Models)
•	Segment profiling
•	Segment prediction (Logistic Regression, Decision Trees)

REQUIREMENTS:

Make sure the following Python packages are installed:
pip install pandas numpy matplotlib seaborn scikit-learn scipy
Dataset
This project expects a CSV file named mcdonalds.csv, which includes:
•	11 binary (Yes/No) brand image attributes
•	A numeric "Like" score column for McDonald's
Example columns:
Yummy, Convenient, Spicy, Fattening, Greasy, Fast, Cheap, Tasty, Expensive, Healthy, Disgusting. 

HOW TO USE:

1.	Place your dataset as mcdonalds.csv in the same directory.
2.	Run the Python script (mcdonalds_segmentation_python.py).
3.	Outputs include:
o	PCA plots
o	Dendrogram for hierarchical clustering
o	K-Means cluster assignments
o	GMM segment assignments
o	Logistic regression and decision tree models
o	Segment-wise box plots for Like scores

OUTPUT SUMMARY:

•	Visual PCA mapping to reduce dimensionality
•	Segment membership for each consumer
•	Tree-based explanation of segment drivers
•	Performance evaluation using silhouette score

NOTES

•	This project is a Python adaptation. Some visual output may differ slightly from the R version due to library defaults.
•	For advanced analysis, extend this with other libraries like plotly, xg boost, or dash for interactivity.

LICENSE:

MIT License. Attribution encouraged for academic or educational reuse.
