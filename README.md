# Diabetes Hospital Readmission Prediction

## Problem Statement
Unplanned 30-day readmissions of diabetic patients cost the US healthcare system 
an estimated $26 billion annually. This project develops a binary classification 
model to identify high-risk patients at the point of discharge using clinical data 
from 130 US hospitals (1999-2008), enabling targeted early intervention.

## Dataset
- Source: UCI Machine Learning Repository
- 101,766 patient encounters, 50 clinical and administrative features
- Target: Binary classification (<30 day readmission = 1, otherwise = 0)
- Class imbalance: 11.4% positive class (readmitted within 30 days)
- Link: https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008

## Project Structure
diabetes_project/
    01_load_data.ipynb    (full pipeline: EDA, cleaning, modeling, evaluation)
    diabetic_data.csv     (download from UCI link above)
    README.md
    requirements.txt

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
  transfer (IDs 11, 13, 14, 19, 20, 21) to prevent target leakage — 
  deceased patients cannot be readmitted and would introduce noise
- Removed duplicate encounter records
- Final clean dataset: 99,323 encounters, 43 features

### 3. Feature Engineering
- Converted age range strings to numeric midpoints for ordinal modeling
- Engineered medication instability feature by counting per-patient 
  drug dosage changes (Up/Down) as a proxy for clinical instability
- Mapped 600+ unique ICD-9 diagnostic codes into 9 clinically relevant 
  categories (Circulatory, Respiratory, Digestive, Diabetes, Injury, 
  Musculoskeletal, Genitourinary, Neoplasms, Other) using clinical 
  grouping logic — reducing dimensionality while preserving medical meaning
- Applied label encoding to binary categorical variables
- Applied one-hot encoding to multi-class categorical variables

### 4. Modeling and Evaluation
- Addressed class imbalance via random oversampling of minority class 
  in training set only (test set kept at natural distribution)
- Trained and compared three classifiers: Logistic Regression, 
  Random Forest, Gradient Boosting
- Tuned Gradient Boosting using RandomizedSearchCV with 3-fold 
  cross validation across 20 parameter combinations
- Evaluated using ROC-AUC, Precision, Recall, and F1-Score

## Results
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.647 |
| Random Forest | 0.638 |
| Gradient Boosting (Tuned) | 0.665 |

## Key Findings
- Number of lab procedures was the strongest predictor (importance: 0.133)
- Number of medications and time in hospital ranked 2nd and 3rd
- Tuned model achieves 60% recall on the minority class
- Results consistent with published literature on this dataset (AUC 0.63-0.72)

## Clinical Significance
A recall of 0.60 means the model correctly flags 60 out of every 100 
high-risk patients before discharge. In a clinical workflow, these patients 
can receive enhanced post-discharge follow-up, reducing preventable readmissions 
and improving patient outcomes.

## Technologies
- Python 3.13
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Jupyter Notebook

## How to Reproduce
1. Clone this repository
2. pip install -r requirements.txt
3. Download diabetic_data.csv from the UCI link above
4. Place the file in the project root directory
5. Run 01_load_data.ipynb sequentially from top to bottom

## Author
Kritika
Biomedical Engineering, 2nd Year

