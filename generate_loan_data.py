"""
Script to generate synthetic loan applicant data for credit risk prediction
"""
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import random

def generate_loan_dataset(n_samples=1000, random_state=42):
    """
    Generate synthetic loan applicant dataset with realistic features
    """
    np.random.seed(random_state)
    random.seed(random_state)
    
    # Generate base features using make_classification
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=random_state,
        class_sep=0.8
    )
    
    # Create meaningful feature names and transform to realistic ranges
    data = {}
    
    # Age (18-80)
    data['age'] = np.clip(25 + X[:, 0] * 15, 18, 80).astype(int)
    
    # Annual Income (20k-200k)
    data['annual_income'] = np.clip(50000 + X[:, 1] * 40000, 20000, 200000).astype(int)
    
    # Employment Length in years (0-40)
    data['employment_length'] = np.clip(5 + X[:, 2] * 8, 0, 40).astype(int)
    
    # Loan Amount (1k-100k)
    data['loan_amount'] = np.clip(15000 + X[:, 3] * 25000, 1000, 100000).astype(int)
    
    # Credit Score (300-850)
    data['credit_score'] = np.clip(650 + X[:, 4] * 100, 300, 850).astype(int)
    
    # Debt-to-Income Ratio (0-1)
    data['debt_to_income'] = np.clip(0.3 + X[:, 5] * 0.4, 0, 1).round(3)
    
    # Number of Credit Lines (0-20)
    data['num_credit_lines'] = np.clip(3 + X[:, 6] * 5, 0, 20).astype(int)
    
    # Years of Credit History (0-50)
    data['credit_history_years'] = np.clip(5 + X[:, 7] * 10, 0, 50).astype(int)
    
    # Categorical features
    education_levels = ['High School', 'Bachelor', 'Master', 'PhD', 'Associate']
    data['education'] = [random.choice(education_levels) for _ in range(n_samples)]
    
    home_ownership = ['Rent', 'Own', 'Mortgage', 'Other']
    data['home_ownership'] = [random.choice(home_ownership) for _ in range(n_samples)]
    
    loan_purpose = ['debt_consolidation', 'home_improvement', 'major_purchase', 
                   'medical', 'vacation', 'wedding', 'car', 'business']
    data['loan_purpose'] = [random.choice(loan_purpose) for _ in range(n_samples)]
    
    # Target variable: 0 = Low Risk (Good), 1 = High Risk (Bad)
    data['credit_risk'] = y
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic correlations
    # Higher income generally means lower risk
    high_income_mask = df['annual_income'] > 80000
    df.loc[high_income_mask, 'credit_risk'] = np.where(
        np.random.random(sum(high_income_mask)) < 0.7, 0, df.loc[high_income_mask, 'credit_risk']
    )
    
    # Higher credit score means lower risk
    high_credit_mask = df['credit_score'] > 750
    df.loc[high_credit_mask, 'credit_risk'] = np.where(
        np.random.random(sum(high_credit_mask)) < 0.8, 0, df.loc[high_credit_mask, 'credit_risk']
    )
    
    # High debt-to-income ratio increases risk
    high_dti_mask = df['debt_to_income'] > 0.6
    df.loc[high_dti_mask, 'credit_risk'] = np.where(
        np.random.random(sum(high_dti_mask)) < 0.7, 1, df.loc[high_dti_mask, 'credit_risk']
    )
    
    return df

if __name__ == "__main__":
    # Generate dataset
    loan_data = generate_loan_dataset(n_samples=1000)
    
    # Save to CSV
    loan_data.to_csv('loan_data.csv', index=False)
    
    print("Dataset generated successfully!")
    print(f"Shape: {loan_data.shape}")
    print("\nFirst few rows:")
    print(loan_data.head())
    print("\nDataset info:")
    print(loan_data.info())
    print("\nTarget distribution:")
    print(loan_data['credit_risk'].value_counts())