# Credit Risk Scoring System Using Multivariate Statistical Analysis and Machine Learning

A comprehensive Data Science project that develops an interpretable **Credit Risk Scoring System** using **statistical analysis**, **Principal Component Analysis (PCA)**, and **Machine Learning** techniques on the **Home Credit Default Risk** dataset.

The project aims to predict the probability of loan default, generate standardized credit risk scores, and evaluate model performance and fairness across applicant segments.

---

## Project Overview

Financial institutions must accurately evaluate the creditworthiness of loan applicants before approving loans. Traditional manual assessments are time-consuming and may produce inconsistent decisions. This project leverages statistical techniques and machine learning algorithms to automate credit risk assessment while maintaining interpretability.

The workflow includes:

- Data preprocessing and feature engineering
- Exploratory Data Analysis (EDA)
- Statistical hypothesis testing
- Principal Component Analysis (PCA)
- Multiple machine learning classification models
- Credit risk score generation
- Model comparison and evaluation
- Fairness and stability analysis

---

## Objectives

- Clean and preprocess applicant financial data.
- Engineer meaningful features from multiple Home Credit datasets.
- Reduce feature dimensionality using Principal Component Analysis (PCA).
- Perform statistical hypothesis testing on applicant attributes.
- Build and compare multiple machine learning models.
- Generate standardized credit risk scores.
- Evaluate discrimination, calibration, fairness, and stability of the models.

---

## Dataset

This project uses the **Home Credit Default Risk** dataset available on Kaggle.

### Dataset Files

- HC_application_train.csv
- HC_bureau.csv
- HC_bureau_balance.csv
- HC_previous_application.csv
- HC_POS_CASH_balance.csv
- HC_installments_payments.csv
- HC_credit_card_balance.csv
- HomeCredit_columns_description.csv

The datasets contain applicant demographic information, loan history, bureau records, installment payments, credit card balances, and previous loan applications.

---

## Technologies Used

### Programming Languages

- Python

### Libraries

- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- XGBoost
- LightGBM

### Development Environment

- Jupyter Notebook
- Visual Studio Code

---

## Machine Learning Models

The project compares the performance of multiple supervised learning algorithms:

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Naive Bayes
- Random Forest
- XGBoost
- LightGBM

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

---

## Statistical Methods

The project incorporates statistical analysis through:

- Chi-Square Test
- Independent Sample t-Test
- ANOVA
- Correlation Analysis
- Principal Component Analysis (PCA)

---

## Project Structure

```
Credit_Risk_Scoring/
│
├── data/
│   ├── HC_application_train.csv
│   ├── HC_bureau.csv
│   ├── HC_bureau_balance.csv
│   ├── HC_previous_application.csv
│   ├── HC_POS_CASH_balance.csv
│   ├── HC_installments_payments.csv
│   ├── HC_credit_card_balance.csv
│   └── HomeCredit_columns_description.csv
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   ├── 03_Exploratory_Data_Analysis.ipynb
│   ├── 04_Hypothesis_Testing.ipynb
│   ├── 05_Feature_Engineering_and_PCA.ipynb
│   ├── 06_Logistic_Regression.ipynb
│   ├── 07_Decision_Tree.ipynb
│   ├── 08_KNN.ipynb
│   ├── 09_Naive_Bayes.ipynb
│   ├── 10_Random_Forest.ipynb
│   ├── 11_XGBoost.ipynb
│   ├── 12_LightGBM.ipynb
│   ├── 13_Credit_Risk_Scoring.ipynb
│   └── 14_Fairness_Analysis.ipynb
│
├── models/
├── outputs/
├── figures/
├── report/
├── README.md
└── requirements.txt
```

---

## Project Workflow

```
Data Collection
        │
        ▼
Data Understanding
        │
        ▼
Data Cleaning
        │
        ▼
Feature Engineering
        │
        ▼
Dataset Merging
        │
        ▼
Exploratory Data Analysis
        │
        ▼
Hypothesis Testing
        │
        ▼
Feature Scaling
        │
        ▼
Principal Component Analysis
        │
        ▼
Machine Learning Models
        │
        ▼
Model Evaluation
        │
        ▼
Credit Risk Score Generation
        │
        ▼
Fairness Analysis
```

---

## Expected Outcomes

- Cleaned and integrated applicant dataset.
- PCA-reduced feature set.
- Comparative analysis of multiple classification models.
- Standardized credit risk score (300–900 scale).
- Risk categorization for loan approval.
- Fairness and stability evaluation across applicant groups.

---

## Repository Setup

### Clone the Repository

```bash
git clone https://github.com/ahamril2265/Credit_Risk_Scoring.git
cd Credit_Risk_Scoring
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

---

## Results

The project will compare multiple machine learning models and identify the best-performing classifier based on predictive performance and interpretability. The selected model will be used to generate standardized credit scores and classify applicants into different credit risk categories.

---

## Future Enhancements

- Hyperparameter optimization using Optuna
- Explainable AI using SHAP and LIME
- Probability calibration
- Interactive dashboard using Streamlit
- Model deployment with FastAPI
- Automated retraining pipeline

---

## Course Outcomes

| Course Outcome | Description |
|----------------|-------------|
| CO3 | Hypothesis testing using Chi-Square, t-Test, and ANOVA |
| CO5 | Multivariate statistical analysis and PCA |
| CO6 | Statistical computing using Python |
| CO4 (Data Analytics) | Supervised Machine Learning for credit risk prediction |

---

## License

This project is intended for academic and educational purposes.

---

## Acknowledgements

- Kaggle – Home Credit Default Risk Dataset
- Scikit-learn
- XGBoost
- LightGBM
- Home Credit Group
