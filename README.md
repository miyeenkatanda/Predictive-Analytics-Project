# Predictive-Analytics-Project
🧠 Predictive Analytics Project
Credit Default Prediction using Machine Learning (CatBoost, XGBoost, LightGBM)
📋 Overview

This project applies supervised machine learning to predict the likelihood of customer default using the dataset cleaned_casedata.csv.
Three models — CatBoost, XGBoost, and LightGBM — were trained and compared to identify the most effective algorithm for accurate and interpretable credit risk prediction.

🎯 Objectives

Build a predictive model to estimate the probability of customer default.

Compare performance across multiple boosting-based algorithms.

Use SHAP and feature importance to interpret model behavior.

🧩 Data Preparation

Input File: cleaned_casedata.csv

Key preprocessing steps:

Dropped redundant columns (Unnamed: 0, EXT_SOURCE_2, EXT_SOURCE_3)

Created an average external score variable EXT_SOURCE_AVG

Handled missing values:

Categorical → 'Missing'

Numerical → median values

Split data into train (70%) and test (30%) using stratified sampling

⚙️ Models Evaluated
Model	Holdout Accuracy	Holdout ROC–AUC	Notes
CatBoost	0.735	0.724	Handles categorical variables natively; stable and interpretable
XGBoost	0.921	0.717	Strong accuracy, slightly lower AUC; efficient tree boosting
LightGBM	0.819	0.708	Fast and memory-efficient, but slightly less accurate
🏆 Model Selection

All models performed well, but CatBoost was selected as the final model because it:

Is specifically optimized for categorical variables,

Achieved balanced performance across accuracy and AUC,

Provided more interpretable results through native feature importance and SHAP integration.

📈 Model Evaluation

Each model was trained with consistent hyperparameters (200 iterations, depth = 6, learning rate = 0.1).
Evaluation metrics included:

Accuracy (classification performance)

ROC–AUC (discrimination ability)

Feature Importance and SHAP for interpretability

Generated plots:

ROC Curve

Top 15 Feature Importances

SHAP Summary Plot

🔍 Explainability with SHAP

SHAP (SHapley Additive Explanations) was used to interpret feature influence.

The SHAP summary plot visualized the most impactful predictors and their direction of effect.

High external source values and longer credit tenure were associated with lower default risk.

🧾 Key Insights

External credit sources and credit amount were strong predictors of default.

Higher external source averages correlated with better repayment behavior.

The model enables explainable, data-driven credit risk assessments.

🧰 Dependencies

Install required packages:

pip install numpy pandas scikit-learn catboost xgboost lightgbm shap matplotlib

▶️ How to Run

Clone the repository:

git clone https://github.com/miyeenkatanda/Predictive-Analytics-Project.git


Place cleaned_casedata.csv in the project folder.

Run the script:

python predictive_model.py


View printed metrics and generated plots.

📊 Project Structure
Predictive-Analytics-Project/
│
├── cleaned_casedata.csv
├── predictive_model.py
├── README.md
│
├── Outputs/
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── shap_summary.png
│
└── requirements.txt
