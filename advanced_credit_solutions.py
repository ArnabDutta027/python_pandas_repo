"""
Advanced Credit Risk Solutions and Mitigation Strategies

This module provides sophisticated solutions for managing credit risk,
including machine learning models, portfolio management, and regulatory compliance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedCreditRiskSolutions:
    """Advanced solutions for credit risk management"""
    
    def __init__(self):
        self.models = {}
        self.risk_thresholds = {
            'conservative': 0.15,
            'moderate': 0.25,
            'aggressive': 0.35
        }
    
    def generate_synthetic_data(self, n_samples: int = 10000) -> pd.DataFrame:
        """Generate synthetic loan data for demonstration"""
        np.random.seed(42)
        
        data = {
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'annual_income': np.random.lognormal(10.8, 0.6, n_samples).clip(20000, 500000),
            'debt_to_income': np.random.beta(2, 5, n_samples).clip(0, 0.8),
            'employment_years': np.random.exponential(3, n_samples).clip(0, 40),
            'loan_amount': np.random.lognormal(10, 0.8, n_samples).clip(1000, 1000000),
            'age': np.random.normal(40, 12, n_samples).clip(18, 80),
            'num_credit_accounts': np.random.poisson(8, n_samples).clip(0, 30),
            'payment_history_months': np.random.exponential(36, n_samples).clip(0, 300),
            'previous_defaults': np.random.poisson(0.3, n_samples).clip(0, 10),
            'bankruptcy_history': np.random.binomial(1, 0.05, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create loan_to_income ratio
        df['loan_to_income'] = df['loan_amount'] / df['annual_income']
        
        # Create default probability based on features (synthetic target)
        default_prob = (
            0.3 * (850 - df['credit_score']) / 550 +  # Credit score impact
            0.25 * df['debt_to_income'] +              # DTI impact
            0.2 * np.maximum(0, 3 - df['employment_years']) / 3 +  # Employment stability
            0.15 * np.minimum(df['loan_to_income'] / 5, 1) +       # Loan size impact
            0.1 * df['previous_defaults'] / 5 +         # Default history
            0.2 * df['bankruptcy_history']              # Bankruptcy impact
        )
        
        # Add some noise and create binary default outcome
        default_prob += np.random.normal(0, 0.1, n_samples)
        df['default'] = (default_prob + np.random.normal(0, 0.05, n_samples) > 0.4).astype(int)
        
        return df
    
    def train_ml_models(self, data: pd.DataFrame) -> Dict[str, float]:
        """Train machine learning models for credit risk prediction"""
        
        # Prepare features
        feature_columns = [
            'credit_score', 'annual_income', 'debt_to_income', 'employment_years',
            'loan_amount', 'age', 'num_credit_accounts', 'payment_history_months',
            'previous_defaults', 'bankruptcy_history', 'loan_to_income'
        ]
        
        X = data[feature_columns]
        y = data['default']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100, 
            max_depth=10, 
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        gb_model.fit(X_train, y_train)
        
        # Store models
        self.models['random_forest'] = rf_model
        self.models['gradient_boosting'] = gb_model
        self.feature_columns = feature_columns
        
        # Evaluate models
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        gb_pred_proba = gb_model.predict_proba(X_test)[:, 1]
        
        rf_auc = roc_auc_score(y_test, rf_pred_proba)
        gb_auc = roc_auc_score(y_test, gb_pred_proba)
        
        return {
            'random_forest_auc': rf_auc,
            'gradient_boosting_auc': gb_auc,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_default_risk(self, applicant_data: Dict) -> Dict:
        """Predict default risk for a new applicant"""
        
        if not self.models:
            raise ValueError("Models not trained. Call train_ml_models first.")
        
        # Prepare input data
        input_df = pd.DataFrame([applicant_data])
        
        # Ensure all required features are present
        for col in self.feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0  # Default value for missing features
        
        X = input_df[self.feature_columns]
        
        # Get predictions from both models
        rf_prob = self.models['random_forest'].predict_proba(X)[0, 1]
        gb_prob = self.models['gradient_boosting'].predict_proba(X)[0, 1]
        
        # Ensemble prediction (average)
        ensemble_prob = (rf_prob + gb_prob) / 2
        
        # Determine risk category
        if ensemble_prob <= self.risk_thresholds['conservative']:
            risk_category = 'Low Risk'
            recommendation = 'Approve with standard terms'
        elif ensemble_prob <= self.risk_thresholds['moderate']:
            risk_category = 'Medium Risk'
            recommendation = 'Approve with enhanced monitoring'
        elif ensemble_prob <= self.risk_thresholds['aggressive']:
            risk_category = 'High Risk'
            recommendation = 'Conditional approval with strict terms'
        else:
            risk_category = 'Very High Risk'
            recommendation = 'Decline application'
        
        return {
            'default_probability': ensemble_prob,
            'rf_probability': rf_prob,
            'gb_probability': gb_prob,
            'risk_category': risk_category,
            'recommendation': recommendation
        }
    
    def portfolio_risk_analysis(self, loan_portfolio: pd.DataFrame) -> Dict:
        """Analyze risk across a loan portfolio"""
        
        # Calculate portfolio metrics
        total_loans = len(loan_portfolio)
        total_exposure = loan_portfolio['loan_amount'].sum()
        
        # Risk distribution
        risk_distribution = {}
        expected_losses = []
        
        for _, loan in loan_portfolio.iterrows():
            prediction = self.predict_default_risk(loan.to_dict())
            default_prob = prediction['default_probability']
            expected_loss = loan['loan_amount'] * default_prob
            expected_losses.append(expected_loss)
            
            risk_cat = prediction['risk_category']
            if risk_cat not in risk_distribution:
                risk_distribution[risk_cat] = {'count': 0, 'exposure': 0}
            
            risk_distribution[risk_cat]['count'] += 1
            risk_distribution[risk_cat]['exposure'] += loan['loan_amount']
        
        total_expected_loss = sum(expected_losses)
        portfolio_default_rate = total_expected_loss / total_exposure
        
        # Calculate Value at Risk (VaR) at 95% confidence level
        expected_losses_sorted = sorted(expected_losses, reverse=True)
        var_95_index = int(0.05 * len(expected_losses_sorted))
        var_95 = expected_losses_sorted[var_95_index] if var_95_index < len(expected_losses_sorted) else 0
        
        return {
            'total_loans': total_loans,
            'total_exposure': total_exposure,
            'expected_total_loss': total_expected_loss,
            'portfolio_default_rate': portfolio_default_rate,
            'value_at_risk_95': var_95,
            'risk_distribution': risk_distribution
        }
    
    def generate_risk_mitigation_strategies(self, applicant_data: Dict, 
                                         prediction: Dict) -> List[str]:
        """Generate specific risk mitigation strategies"""
        
        strategies = []
        default_prob = prediction['default_probability']
        
        # Credit score based strategies
        if applicant_data.get('credit_score', 700) < 650:
            strategies.extend([
                "Require co-signer with credit score > 700",
                "Implement credit improvement program with quarterly reviews",
                "Offer secured loan option with collateral"
            ])
        
        # Income and employment strategies
        if applicant_data.get('employment_years', 5) < 2:
            strategies.extend([
                "Require employment verification and offer letter",
                "Implement probationary period with payment monitoring",
                "Consider income stability insurance"
            ])
        
        # Debt-to-income strategies
        if applicant_data.get('debt_to_income', 0.3) > 0.4:
            strategies.extend([
                "Require debt consolidation plan",
                "Implement automatic payment setup",
                "Offer financial counseling services"
            ])
        
        # Loan amount strategies
        loan_to_income = applicant_data.get('loan_to_income', 1)
        if loan_to_income > 3:
            strategies.extend([
                "Reduce loan amount by 20-30%",
                "Extend loan term to reduce monthly payments",
                "Require additional collateral or guarantees"
            ])
        
        # Default history strategies
        if applicant_data.get('previous_defaults', 0) > 0:
            strategies.extend([
                "Implement early warning system for payment delays",
                "Require explanation and remediation plan for previous defaults",
                "Set up automatic payment with penalty for manual payments"
            ])
        
        # High-risk specific strategies
        if default_prob > 0.3:
            strategies.extend([
                "Require comprehensive financial audit",
                "Implement weekly payment monitoring",
                "Consider loan insurance or guarantee programs",
                "Establish emergency contact and intervention protocols"
            ])
        
        return strategies

def demonstrate_advanced_solutions():
    """Demonstrate advanced credit risk solutions"""
    
    print("=" * 80)
    print("ADVANCED CREDIT RISK SOLUTIONS AND MITIGATION STRATEGIES")
    print("=" * 80)
    
    # Initialize the system
    risk_system = AdvancedCreditRiskSolutions()
    
    # Generate synthetic data and train models
    print("\n1. GENERATING SYNTHETIC LOAN DATA AND TRAINING ML MODELS")
    print("-" * 60)
    
    loan_data = risk_system.generate_synthetic_data(10000)
    print(f"Generated {len(loan_data)} synthetic loan records")
    
    # Train models
    model_performance = risk_system.train_ml_models(loan_data)
    print(f"Random Forest AUC: {model_performance['random_forest_auc']:.3f}")
    print(f"Gradient Boosting AUC: {model_performance['gradient_boosting_auc']:.3f}")
    
    # Test with sample applicants
    print("\n2. ADVANCED RISK PREDICTION FOR SAMPLE APPLICANTS")
    print("-" * 60)
    
    sample_applicants = [
        {
            'name': 'High-Risk Applicant',
            'credit_score': 580,
            'annual_income': 35000,
            'debt_to_income': 0.55,
            'employment_years': 0.8,
            'loan_amount': 25000,
            'age': 25,
            'num_credit_accounts': 15,
            'payment_history_months': 6,
            'previous_defaults': 3,
            'bankruptcy_history': 1,
            'loan_to_income': 0.71
        },
        {
            'name': 'Medium-Risk Applicant',
            'credit_score': 680,
            'annual_income': 65000,
            'debt_to_income': 0.35,
            'employment_years': 3.5,
            'loan_amount': 30000,
            'age': 32,
            'num_credit_accounts': 8,
            'payment_history_months': 42,
            'previous_defaults': 1,
            'bankruptcy_history': 0,
            'loan_to_income': 0.46
        },
        {
            'name': 'Low-Risk Applicant',
            'credit_score': 750,
            'annual_income': 95000,
            'debt_to_income': 0.22,
            'employment_years': 7,
            'loan_amount': 40000,
            'age': 38,
            'num_credit_accounts': 6,
            'payment_history_months': 84,
            'previous_defaults': 0,
            'bankruptcy_history': 0,
            'loan_to_income': 0.42
        }
    ]
    
    for applicant in sample_applicants:
        name = applicant.pop('name')
        print(f"\n{name}:")
        
        # Get prediction
        prediction = risk_system.predict_default_risk(applicant)
        print(f"  Default Probability: {prediction['default_probability']:.1%}")
        print(f"  Risk Category: {prediction['risk_category']}")
        print(f"  Recommendation: {prediction['recommendation']}")
        
        # Get mitigation strategies
        strategies = risk_system.generate_risk_mitigation_strategies(applicant, prediction)
        print(f"  Mitigation Strategies:")
        for strategy in strategies[:3]:  # Show top 3 strategies
            print(f"    • {strategy}")
    
    # Portfolio analysis
    print("\n3. PORTFOLIO RISK ANALYSIS")
    print("-" * 60)
    
    # Create a sample portfolio
    portfolio_sample = loan_data.sample(100).copy()
    portfolio_analysis = risk_system.portfolio_risk_analysis(portfolio_sample)
    
    print(f"Portfolio Size: {portfolio_analysis['total_loans']} loans")
    print(f"Total Exposure: ${portfolio_analysis['total_exposure']:,.0f}")
    print(f"Expected Total Loss: ${portfolio_analysis['expected_total_loss']:,.0f}")
    print(f"Portfolio Default Rate: {portfolio_analysis['portfolio_default_rate']:.2%}")
    print(f"Value at Risk (95%): ${portfolio_analysis['value_at_risk_95']:,.0f}")
    
    print(f"\nRisk Distribution:")
    for risk_level, metrics in portfolio_analysis['risk_distribution'].items():
        percentage = (metrics['count'] / portfolio_analysis['total_loans']) * 100
        print(f"  {risk_level}: {metrics['count']} loans ({percentage:.1f}%) - "
              f"${metrics['exposure']:,.0f} exposure")
    
    print("\n4. REGULATORY COMPLIANCE AND BEST PRACTICES")
    print("-" * 60)
    print("• Implement Fair Credit Reporting Act (FCRA) compliance")
    print("• Ensure Equal Credit Opportunity Act (ECOA) adherence")
    print("• Maintain Basel III capital adequacy ratios")
    print("• Regular model validation and back-testing")
    print("• Stress testing under adverse economic scenarios")
    print("• Documentation of all risk assessment decisions")
    print("• Regular audit trails and model governance")
    
    print("\n5. TECHNOLOGY INTEGRATION RECOMMENDATIONS")
    print("-" * 60)
    print("• Real-time credit scoring APIs")
    print("• Automated decision engines with human oversight")
    print("• Machine learning model monitoring and drift detection")
    print("• Integration with credit bureaus and alternative data sources")
    print("• Blockchain for loan origination and servicing transparency")
    print("• AI-powered fraud detection systems")
    print("• Customer communication automation for risk mitigation")

if __name__ == "__main__":
    demonstrate_advanced_solutions()