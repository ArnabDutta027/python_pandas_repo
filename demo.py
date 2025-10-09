"""
Simple demo script for credit risk prediction
"""
from credit_risk_predictor import CreditRiskPredictor

def demo_predictions():
    """Demo with different applicant profiles"""
    
    # Initialize and train the model
    predictor = CreditRiskPredictor()
    predictor.load_data()
    predictor.preprocess_data()
    predictor.train_decision_tree(optimize_hyperparameters=False)  # Faster training
    
    print("\n" + "="*60)
    print("CREDIT RISK PREDICTION DEMO")
    print("="*60)
    
    # Test cases with different risk profiles
    test_cases = [
        {
            'name': 'Low Risk Applicant',
            'data': {
                'age': 45,
                'annual_income': 120000,
                'employment_length': 15,
                'loan_amount': 20000,
                'credit_score': 780,
                'debt_to_income': 0.25,
                'num_credit_lines': 8,
                'credit_history_years': 20,
                'education': 'Master',
                'home_ownership': 'Own',
                'loan_purpose': 'home_improvement'
            }
        },
        {
            'name': 'High Risk Applicant',
            'data': {
                'age': 25,
                'annual_income': 35000,
                'employment_length': 2,
                'loan_amount': 30000,
                'credit_score': 580,
                'debt_to_income': 0.75,
                'num_credit_lines': 2,
                'credit_history_years': 3,
                'education': 'High School',
                'home_ownership': 'Rent',
                'loan_purpose': 'vacation'
            }
        },
        {
            'name': 'Medium Risk Applicant',
            'data': {
                'age': 35,
                'annual_income': 65000,
                'employment_length': 7,
                'loan_amount': 25000,
                'credit_score': 680,
                'debt_to_income': 0.45,
                'num_credit_lines': 5,
                'credit_history_years': 10,
                'education': 'Bachelor',
                'home_ownership': 'Mortgage',
                'loan_purpose': 'debt_consolidation'
            }
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'-'*50}")
        print(f"TEST CASE {i}: {test_case['name']}")
        print(f"{'-'*50}")
        
        prediction, probability = predictor.predict_single_applicant(test_case['data'])
        
        # Risk assessment
        risk_level = "HIGH" if prediction == 1 else "LOW"
        confidence = max(probability) * 100
        
        print(f"RISK ASSESSMENT: {risk_level} RISK")
        print(f"CONFIDENCE: {confidence:.1f}%")
        
        if prediction == 1:
            print("⚠️  RECOMMENDATION: Loan application requires careful review")
            print("   Consider additional documentation or collateral")
        else:
            print("✅ RECOMMENDATION: Loan application can be approved")
            print("   Applicant shows good creditworthiness")

if __name__ == "__main__":
    demo_predictions()