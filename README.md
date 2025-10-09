# Credit Risk Prediction using Decision Tree

This project implements a machine learning solution to predict credit risk for loan applicants using a Decision Tree algorithm.

## Overview

The Decision Tree classifier analyzes various features of loan applicants to predict whether they pose a high or low credit risk. This helps financial institutions make informed lending decisions.

## Features

- **Comprehensive Feature Set**: Analyzes age, income, loan amount, credit score, employment years, existing loans, education, marital status, and home ownership
- **Model Training & Evaluation**: Trains a Decision Tree classifier with performance metrics
- **Visualization**: Generates decision tree visualization and confusion matrix
- **Prediction Engine**: Predicts credit risk for new loan applicants
- **Feature Importance**: Identifies which factors most influence credit risk

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the credit risk predictor:
```bash
python credit_risk_predictor.py
```

The script will:
1. Generate a sample dataset of 1000 loan applicants
2. Train a Decision Tree classifier
3. Evaluate the model with accuracy, confusion matrix, and classification report
4. Visualize the decision tree and results
5. Make predictions for sample new applicants

## Output Files

- `credit_risk_dataset.csv`: Generated sample dataset
- `decision_tree_visualization.png`: Visual representation of the decision tree and confusion matrix

## Model Features

The model considers the following features:
- **Age**: Applicant's age
- **Income**: Annual income
- **Loan Amount**: Requested loan amount
- **Credit Score**: Credit score (300-850)
- **Employment Years**: Years of employment
- **Existing Loans**: Number of existing loans
- **Education**: Education level (High School, Bachelor, Master, PhD)
- **Marital Status**: Single, Married, or Divorced
- **Home Ownership**: Rent, Own, or Mortgage

## Example Predictions

The script includes examples of both high-risk and low-risk applicants to demonstrate the model's capabilities.

## Model Performance

The model provides:
- Accuracy score
- Confusion matrix
- Precision, recall, and F1-score for each risk category
- Feature importance rankings

## Customization

You can customize the model by adjusting:
- `max_depth`: Maximum depth of the decision tree
- `min_samples_split`: Minimum samples required to split a node
- `min_samples_leaf`: Minimum samples required at a leaf node
- Dataset size and features

## Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
