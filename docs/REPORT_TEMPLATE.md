# Final Project Report

## Title
E-Commerce Purchase Intent Prediction with Explainable AI

## Abstract
This project develops an end-to-end machine-learning workflow to estimate whether an online shopping session will result in a purchase. The pipeline covers data validation, exploratory analysis, preprocessing, model comparison, evaluation, and delivery through an API and dashboard. Replace this paragraph with a concise summary of the actual dataset, selected model, and measured results after running training.

## 1. Problem Statement
E-commerce teams need to understand which sessions are more likely to convert. The objective is to estimate purchase intent from session-level browsing and context features and communicate model performance transparently.

## 2. Objectives
- Prepare and validate the public Online Shoppers Purchasing Intention dataset.
- Explore purchase-rate and behaviour patterns.
- Compare six classification algorithms using imbalance-aware metrics.
- Package preprocessing and model artifacts for repeatable inference.
- Expose predictions through FastAPI and a Streamlit dashboard.
- Document limitations and reproducibility steps.

## 3. Dataset
Source: UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset. Record the actual row count, columns, target balance, missing values, and data provenance after running the project. State clearly if synthetic fallback data was used; synthetic metrics are not evidence of performance on the real dataset.

## 4. Methodology
Use a stratified train/test split. Fit imputation, scaling, and one-hot encoding only within the training pipeline. Compare Logistic Regression, Decision Tree, Random Forest, Extra Trees, HistGradientBoosting, and Gradient Boosting. Report precision, recall, F1, ROC-AUC and PR-AUC. Discuss threshold choice in terms of false positives and false negatives.

## 5. Results
Paste the real contents of `artifacts/model_comparison.csv` here. Include a chart and confusion matrix. Do not invent results. State which model was selected and why.

## 6. Explainability and Leakage Considerations
Feature importance indicates model reliance, not causality. Consider using SHAP for global/local explanations. Investigate whether `PageValues` is available at the time the prediction is intended to be made; if not, exclude it and compare performance.

## 7. Application
Describe the FastAPI `/predict` endpoint and Streamlit dashboard. Add screenshots of the running application and Swagger docs.

## 8. Testing and Reproducibility
Report Python version, dependency installation, pytest results, training command, and Git commit/repository link.

## 9. Limitations and Future Work
Discuss historical data, drift, threshold costs, data privacy, calibration, cross-validation, SHAP/LIME, MLflow, containerization, and deployment monitoring.

## 10. Conclusion
Summarize only verified results from your run and suggest realistic next steps.

## References
1. UCI Machine Learning Repository, Online Shoppers Purchasing Intention Dataset.
2. scikit-learn documentation: https://scikit-learn.org/stable/
3. FastAPI documentation: https://fastapi.tiangolo.com/
4. Streamlit documentation: https://docs.streamlit.io/
