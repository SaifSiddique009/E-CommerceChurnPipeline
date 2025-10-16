# Customer Churn Prediction - Machine Learning Pipeline

## Project Overview

This project implements a complete machine learning pipeline to predict customer churn based on customer data. The pipeline includes data exploration, preprocessing, feature engineering, model training, and comprehensive evaluation.

The goal of this project is to predict whether a customeer will churn (leave) based on their behavioral and demographic data.

## Dataset Description

**Dataset**: Customer Churn Dataset
- **Total Records**: 1,000 customers
- **Features**: 15 columns (14 features + 1 target)
- **Target Variable**: `Target_Churn` (Binary: True/False)

### Features:

**Numerical Features (10):**
- `Age`: Customer age
- `Annual_Income`: Annual income
- `Total_Spend`: Total amount spent
- `Years_as_Customer`: Customer tenure
- `Num_of_Purchases`: Total purchases made
- `Average_Transaction_Amount`: Average transaction value
- `Num_of_Returns`: Number of returns
- `Num_of_Support_Contacts`: Support interactions
- `Satisfaction_Score`: Customer satisfaction rating
- `Last_Purchase_Days_Ago`: Days since last purchase

**Categorical Features (3):**
- `Gender`: Male, Female, Other
- `Promotion_Response`: Responded, Ignored, Unsubscribed
- `Email_Opt_In`: True/False

## Project Structure

```
E-CommerceChurnPipeline/
├── data/
│   ├── dataset.csv              # Original dataset
│   └── processed/               # Processed train/val/test splits
├── notebooks/
│   └── churn_prediction_analysis.ipynb  # Interactive Jupyter notebook
├── scripts/
│   ├── eda.py                   # Exploratory Data Analysis
│   ├── preprocessing.py         # Data preprocessing
│   ├── feature_engineering.py   # Feature engineering
│   ├── modeling.py              # Model training
│   ├── evaluation.py            # Model evaluation
│   └── main.py                  # Main execution script
├── visualizations/              # All EDA and evaluation plots
├── models/                      # Trained model files
├── reports/                     # Evaluation reports
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Setup Instructions

### Installation

1. **Clone the repository and navigate to the project directory**

```bash
git clone https://github.com/SaifSiddique009/E-CommerceChurnPipeline.git
cd E-CommerceChurnPipeline
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Complete Pipeline

Execute the entire pipeline from data loading to model evaluation:

```bash
cd scripts
python main.py
```

### Option 2: Run Individual Modules

Run each module separately for step-by-step execution:

```bash
cd scripts

# Step 1: Exploratory Data Analysis
python eda.py

# Step 2: Data Preprocessing
python preprocessing.py

# Step 3: Feature Engineering
python feature_engineering.py

# Step 4: Model Training
python modeling.py

# Step 5: Model Evaluation
python evaluation.py
```

### Option 3: Use Jupyter Notebook

For interactive analysis:

```bash
jupyter notebook notebooks/churn_prediction_analysis.ipynb
```

## Models Implemented

### Baseline Models
1. **Logistic Regression** - Simple linear classifier
2. **Random Forest** - Ensemble of decision trees
3. **XGBoost** - Gradient boosting algorithm
4. **LightGBM** - Fast gradient boosting framework

### Advanced Models
5. **Artificial Neural Network (ANN)** - Deep learning model using TensorFlow/Keras
   - Architecture: 4 hidden layers (128, 64, 32, 16 neurons)
   - Dropout layers for regularization
   - Early stopping and learning rate reduction

## Pipeline Steps

### 1. Exploratory Data Analysis (EDA)
- Data quality checks (missing values, duplicates)
- Target variable distribution analysis
- Numerical feature distributions and outlier detection
- Categorical feature analysis
- Correlation analysis
- Bivariate analysis (features vs target)
- **Outputs**: 8 visualization files + EDA summary report

### 2. Data Preprocessing
- Remove unnecessary columns (Customer_ID)
- **Encoding Strategy**:
  - **One-Hot Encoding** for Gender and Promotion_Response (no ordinal relationship)
  - Binary encoding for Email_Opt_In and Target_Churn
- Feature scaling using StandardScaler
- Data splitting: 70% train, 15% validation, 15% test
- Class imbalance handling using SMOTE
- **Outputs**: Processed datasets + scaler object

### 3. Feature Engineering
- Created interaction features:
  - `Spend_per_Purchase`
  - `Return_Rate`
  - `Support_per_Year`
  - `Purchase_Frequency`
  - `Income_to_Spend_Ratio`
  - `Recency_Score`
- Feature importance analysis
- **Outputs**: Engineered datasets + feature importance plot

### 4. Model Training
- Train all baseline models
- Train advanced ANN model
- **Outputs**: Trained model files (.pkl, .keras)

### 5. Model Evaluation
- Performance metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-Score
  - ROC-AUC
- Confusion matrices
- ROC curves
- Detailed classification reports
- **Outputs**: Comparison visualizations + detailed reports

## Key Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.53 | 0.55 | 0.68 | 0.61 | 0.52 |
| Random Forest | 0.53 | 0.55 | 0.68 | 0.61 | 0.56 |
| XGBoost | 0.54 | 0.54 | 0.84 | 0.66 | 0.53 |
| LightGBM | 0.53 | 0.53 | 0.89 | 0.67 | 0.49 |
| ANN | 0.46 | 0.49 | 0.59 | 0.54 | 0.49 |

### Key Insights

1. **Dataset Characteristics**:
   - **Size**: 1,000 customers (15 features)
   - **Target Distribution**: Relatively balanced (52.6% churned)
   - **Data Quality**: No missing values, no duplicates
   - **Feature Correlations**: Weak correlations with target variable

2. **Top Features Affecting Churn**:
   - **Support_per_Year** (Engineered) - Highest importance
  - **Average_Transaction_Amount** - Transaction behavior
  - **Return_Rate** (Engineered) - Customer satisfaction
  - **Spend_per_Purchase** (Engineered) - Purchase patterns
  - **Num_of_Purchases** - Total purchases made

3. **Encoding Decisions**:
   - **One-Hot Encoding** used for Gender and Promotion_Response because:
     - No ordinal/hierarchical relationship between categories
     - Prevents model from assuming false ordering
     - Preserves independence of categories

## Accuracy Analysis

**Achieved**: 54% accuracy (XGBoost)

### Possible Reason of Lower Performance

1. **Weak Feature-Target Correlation**
   - Highest absolute correlation with target: <0.04
   - Features have limited predictive power
   
2. **Dataset Characteristics**
   - Small sample size (1,000 customers)
   - Limited feature diversity
   - May lack critical churn indicators

3. **What Was Done to Improve**
   - Feature engineering (6 new features)
   - Advanced model (ANN with 4 layers)
   - Proper preprocessing and scaling

### Recommendations for Improvement

1. **Data Collection**
   - Customer engagement metrics (clicks, time on site)
   - Email open/click rates
   - Customer service interaction quality
   - Product usage frequency
   - Payment history patterns

2. **Feature Engineering**
   - Time-series patterns
   - Customer lifetime value
   - Cohort analysis features
   - Seasonal patterns

## Output Files

### Visualizations (`/visualizations/`)
1. `01_target_distribution.png` - Target variable balance
2. `02_numerical_distributions.png` - Feature distributions
3. `03_numerical_boxplots.png` - Outlier detection
4. `04_categorical_distributions.png` - Categorical features
5. `05_correlation_matrix.png` - Feature correlations
6. `06_top_correlations.png` - Top correlated features
7. `07_bivariate_numerical.png` - Numerical vs target
8. `08_bivariate_categorical.png` - Categorical vs target
9. `09_feature_importance.png` - Feature importance scores
10. `10_model_comparison.png` - Model performance comparison
11. `11_confusion_matrices.png` - All confusion matrices
12. `12_roc_curves.png` - ROC curves comparison

### Models (`/models/`)
- `logistic_regression.pkl`
- `random_forest.pkl`
- `xgboost.pkl`
- `lightgbm.pkl`
- `ann_model.keras`
- `scaler.pkl`
- `ensemble.pkl` (if created)

### Reports (`/reports/`)
- `model_comparison.csv` - Metrics comparison table
- Classification reports for each model (`.txt` files)
- `EDA_SUMMARY_REPORT.txt` - EDA findings summary

## Future Model Improvement Strategy

1. **Hyperparameter Tuning**:
   - GridSearchCV for Random Forest and XGBoost
   - Optimizes model parameters for better performance

2. **Feature Engineering**:
   - Already implemented with 6 derived features
   - Can be extended with domain knowledge

3. **Ensemble Methods**:
   - Voting Classifier combining best models
   - Leverages strengths of multiple algorithms

---
