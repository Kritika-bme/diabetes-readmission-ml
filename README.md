# Diabetes Hospital Readmission Prediction

## Overview
An end-to-end machine learning pipeline to predict 30-day hospital readmission 
in diabetic patients, with explainability analysis using SHAP values to identify 
key clinical risk factors.

## Problem Statement
Unplanned 30-day readmissions of diabetic patients cost the US healthcare system 
an estimated $26 billion annually. This project develops a binary classification 
model to identify high-risk patients at the point of discharge using clinical data 
from 130 US hospitals (1999-2008), enabling targeted early intervention.

## Results
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.647 |
| Random Forest | 0.638 |
| Gradient Boosting (Tuned) | 0.665 |
| XGBoost | 0.6645 |

Results consistent with published literature on this dataset (AUC 0.63-0.72).

## SHAP Explainability Findings
Top clinical drivers of 30-day readmission risk:
1. num_lab_procedures — strongest predictor (more tests = sicker patient)
2. num_medications — higher medication count = more complex condition
3. time_in_hospital — longer stay = more serious illness
4. age_numeric — older patients at higher risk
5. number_inpatient — prior hospitalizations predict future ones

SHAP analysis reveals the model is clinically interpretable —
findings align with established medical knowledge on diabetic readmission risk.

## Methodology

### 1. Exploratory Data Analysis
- Identified class imbalance of 11.4% requiring oversampling strategy
- Detected missing values encoded as '?' across 7 columns
- Analyzed distributions of age, gender, medications, and hospital stay duration
- Generated correlation heatmap to identify multicollinear features

### 2. Data Cleaning
- Replaced '?' placeholders with NaN for proper missing value handling
- Dropped columns exceeding 40% missingness threshold (weight, medical_specialty)
- Excluded patients with discharge dispositions indicating death or hospice
  transfer (IDs 11, 13, 14, 19, 20, 21) to prevent target leakage
- Removed duplicate encounter records
- Final clean dataset: 99,323 encounters, 43 features

### 3. Feature Engineering
- Converted age range strings to numeric midpoints for ordinal modeling
- Engineered medication instability feature by counting per-patient
  drug dosage changes (Up/Down) as a proxy for clinical instability
- Mapped 600+ unique ICD-9 diagnostic codes into 9 clinically relevant
  categories using clinical grouping logic
- Applied label encoding to binary categorical variables
- Applied one-hot encoding to multi-class categorical variables

### 4. Modeling
- Addressed class imbalance via random oversampling of minority class
  in training set only (test set kept at natural distribution)
- Trained and compared four classifiers: Logistic Regression,
  Random Forest, Gradient Boosting, XGBoost
- Tuned Gradient Boosting using RandomizedSearchCV with 3-fold
  cross validation across 20 parameter combinations
- Evaluated using ROC-AUC, Precision, Recall, and F1-Score

### 5. Explainability
- Applied SHAP (SHapley Additive exPlanations) to XGBoost model
- Generated feature importance, beeswarm, and waterfall plots
- Identified top clinical drivers of readmission risk
- Demonstrates transparency in high-stakes medical AI decisions

## Dataset
- Source: UCI Machine Learning Repository
- 101,766 patient encounters, 50 clinical and administrative features
- Target: Binary classification (<30 day readmission = 1, otherwise = 0)
- Link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Technologies
- Python 3.13
- pandas, numpy, scikit-learn
- XGBoost, SHAP
- matplotlib, seaborn
- Jupyter Notebook

## How to Reproduce
1. Clone this repository
2. pip install -r requirements.txt
3. Download diabetic_data.csv from UCI link above
4. Place file in project root directory
5. Run 01_load_data.ipynb sequentially from top to bottom

## Author
Kritika
B.Tech Biomedical Engineering



