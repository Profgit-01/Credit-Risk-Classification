# Credit Risk Classification using CIBIL Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Profgit-01/Credit-Risk-Classification/blob/main/Credit_Risk_Modeling-Prediction.ipynb)

A machine learning project that predicts credit risk for bank customers and classifies them into approval categories **P1 to P4**, using real-world Indian bank data combined with external CIBIL credit bureau data.

## Problem Statement

Banks need to decide quickly and consistently whether a customer is a safe candidate for a loan. This project builds a classification model that uses a customer's internal bank records and external CIBIL credit history to assign them to one of four approval categories (P1, P2, P3, P4), supporting faster and more reliable loan decisions.

## Dataset

- **Internal bank data**: customer account and profile details.
- **External CIBIL data**: credit bureau history for the same customers.
- **Unseen dataset**: held-out data used to test the final model.
- Around **65,000+ rows** across the internal, external and unseen datasets.



## Approach

1. **Data cleaning**: handled `-99999` placeholder values and null values.
2. **Merging**: combined internal and external datasets.
3. **Feature selection**: Chi-Square test (categorical), VIF (multicollinearity) and ANOVA (numerical).
4. **Encoding**: ordinal encoding for `EDUCATION`, one-hot encoding for other categorical features.
5. **Scaling**: `StandardScaler` on key numeric features.
6. **Modeling**: trained and compared Decision Tree, Random Forest and XGBoost.
7. **Hyperparameter tuning**: `GridSearchCV` with cross-validation, plus a custom parameter grid.
8. **Validation**: hypothesis testing to confirm key predictors of default.
9. **Prediction**: applied the final model to the unseen dataset.

## Results

| Model | Notes |
|---|---|
| Decision Tree | Baseline |
| Random Forest | Compared against XGBoost |
| **XGBoost** | Best performing model after tuning |

- Final accuracy: **~79%**
- Best XGBoost parameters: `colsample_bytree=0.9`, `learning_rate=1`, `max_depth=3`, `alpha=10`, `n_estimators=100`

## Tech Stack

- **Language**: Python
- **Libraries**: NumPy, Pandas, scikit-learn, XGBoost, Statsmodels, SciPy, Matplotlib, OpenPyXL
- **Environment**: Google Colab

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/Profgit-01/Credit-Risk-Classification.git
   cd Credit-Risk-Classification
   ```
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn xgboost statsmodels scipy matplotlib openpyxl
   ```
3. Add the dataset files to the project folder (update the file paths in the notebook if needed).
4. Open `Credit_Risk_Modeling-Prediction.ipynb` in Jupyter or upload it to [Google Colab](https://colab.research.google.com/) and run all cells.

## Repository Structure

```
├── Credit_Risk_Modeling-Prediction.ipynb   # Main notebook
├── README.md
└── requirements.txt                        # Optional
```

## Future Improvements

- Handle class imbalance (e.g. SMOTE or class weights)
- Try additional models such as LightGBM
- Add feature importance and model explainability (e.g. SHAP)
- Deploy as a simple web app for loan officers

## Author

**Vishal Kurdiya**
M.Sc. Applied Mathematics, NIT Warangal
