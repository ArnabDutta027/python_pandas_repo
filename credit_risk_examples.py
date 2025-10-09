"""
Real-World Credit Risk Examples and Solutions for Loan Applicants

This module demonstrates common credit risk scenarios that lenders face
and provides practical solutions for assessment and mitigation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum

class RiskLevel(Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    VERY_HIGH = "Very High"

class LoanType(Enum):
    PERSONAL = "Personal Loan"
    MORTGAGE = "Mortgage"
    AUTO = "Auto Loan"
    BUSINESS = "Business Loan"
    CREDIT_CARD = "Credit Card"

@dataclass
class LoanApplicant:
    """Represents a loan applicant with their financial profile"""
    name: str
    age: int
    annual_income: float
    credit_score: int
    employment_years: float
    debt_to_income_ratio: float
    loan_amount: float
    loan_type: LoanType
    collateral_value: Optional[float] = None
    previous_defaults: int = 0
    bankruptcy_history: bool = False
    payment_history_months: int = 0

@dataclass
class CreditRiskAssessment:
    """Credit risk assessment result"""
    applicant_name: str
    risk_level: RiskLevel
    credit_score_impact: int
    probability_of_default: float
    recommended_interest_rate: float
    loan_decision: str
    risk_factors: List[str]
    mitigation_strategies: List[str]

class CreditRiskAnalyzer:
    """Analyzes credit risk for loan applications"""
    
    def __init__(self):
        # Risk weights for different factors
        self.risk_weights = {
            'credit_score': 0.30,
            'debt_to_income': 0.25,
            'employment_stability': 0.20,
            'loan_to_income': 0.15,
            'payment_history': 0.10
        }
        
        # Base interest rates by loan type
        self.base_rates = {
            LoanType.PERSONAL: 0.08,
            LoanType.MORTGAGE: 0.04,
            LoanType.AUTO: 0.06,
            LoanType.BUSINESS: 0.10,
            LoanType.CREDIT_CARD: 0.18
        }
    
    def assess_credit_risk(self, applicant: LoanApplicant) -> CreditRiskAssessment:
        """Comprehensive credit risk assessment"""
        
        risk_factors = []
        mitigation_strategies = []
        risk_score = 0
        
        # 1. Credit Score Analysis
        credit_risk, credit_factors, credit_mitigations = self._assess_credit_score(applicant)
        risk_score += credit_risk * self.risk_weights['credit_score']
        risk_factors.extend(credit_factors)
        mitigation_strategies.extend(credit_mitigations)
        
        # 2. Debt-to-Income Ratio
        dti_risk, dti_factors, dti_mitigations = self._assess_debt_to_income(applicant)
        risk_score += dti_risk * self.risk_weights['debt_to_income']
        risk_factors.extend(dti_factors)
        mitigation_strategies.extend(dti_mitigations)
        
        # 3. Employment Stability
        emp_risk, emp_factors, emp_mitigations = self._assess_employment(applicant)
        risk_score += emp_risk * self.risk_weights['employment_stability']
        risk_factors.extend(emp_factors)
        mitigation_strategies.extend(emp_mitigations)
        
        # 4. Loan-to-Income Ratio
        lti_risk, lti_factors, lti_mitigations = self._assess_loan_to_income(applicant)
        risk_score += lti_risk * self.risk_weights['loan_to_income']
        risk_factors.extend(lti_factors)
        mitigation_strategies.extend(lti_mitigations)
        
        # 5. Payment History
        payment_risk, payment_factors, payment_mitigations = self._assess_payment_history(applicant)
        risk_score += payment_risk * self.risk_weights['payment_history']
        risk_factors.extend(payment_factors)
        mitigation_strategies.extend(payment_mitigations)
        
        # Calculate final metrics
        risk_level = self._determine_risk_level(risk_score)
        probability_of_default = self._calculate_default_probability(risk_score, applicant)
        interest_rate = self._calculate_interest_rate(risk_score, applicant.loan_type)
        loan_decision = self._make_loan_decision(risk_level, probability_of_default)
        
        return CreditRiskAssessment(
            applicant_name=applicant.name,
            risk_level=risk_level,
            credit_score_impact=int(risk_score * 100),
            probability_of_default=probability_of_default,
            recommended_interest_rate=interest_rate,
            loan_decision=loan_decision,
            risk_factors=risk_factors,
            mitigation_strategies=mitigation_strategies
        )
    
    def _assess_credit_score(self, applicant: LoanApplicant) -> Tuple[float, List[str], List[str]]:
        """Assess risk based on credit score"""
        factors = []
        mitigations = []
        
        if applicant.credit_score < 580:
            risk = 1.0
            factors.append(f"Very poor credit score ({applicant.credit_score})")
            mitigations.extend([
                "Require co-signer with good credit",
                "Offer secured loan with collateral",
                "Implement credit counseling program"
            ])
        elif applicant.credit_score < 670:
            risk = 0.7
            factors.append(f"Fair credit score ({applicant.credit_score})")
            mitigations.extend([
                "Higher down payment required",
                "Shorter loan term to reduce risk"
            ])
        elif applicant.credit_score < 740:
            risk = 0.4
            factors.append(f"Good credit score ({applicant.credit_score})")
            mitigations.append("Standard loan terms with monitoring")
        else:
            risk = 0.1
            # No risk factors for excellent credit
        
        return risk, factors, mitigations
    
    def _assess_debt_to_income(self, applicant: LoanApplicant) -> Tuple[float, List[str], List[str]]:
        """Assess risk based on debt-to-income ratio"""
        factors = []
        mitigations = []
        
        if applicant.debt_to_income_ratio > 0.43:
            risk = 1.0
            factors.append(f"High debt-to-income ratio ({applicant.debt_to_income_ratio:.1%})")
            mitigations.extend([
                "Require debt consolidation before approval",
                "Lower loan amount to reduce monthly payments",
                "Extend loan term to reduce monthly burden"
            ])
        elif applicant.debt_to_income_ratio > 0.36:
            risk = 0.6
            factors.append(f"Elevated debt-to-income ratio ({applicant.debt_to_income_ratio:.1%})")
            mitigations.extend([
                "Require proof of stable income",
                "Monthly payment monitoring"
            ])
        elif applicant.debt_to_income_ratio > 0.28:
            risk = 0.3
            factors.append(f"Moderate debt-to-income ratio ({applicant.debt_to_income_ratio:.1%})")
            mitigations.append("Regular financial health check-ins")
        else:
            risk = 0.1
        
        return risk, factors, mitigations
    
    def _assess_employment(self, applicant: LoanApplicant) -> Tuple[float, List[str], List[str]]:
        """Assess risk based on employment stability"""
        factors = []
        mitigations = []
        
        if applicant.employment_years < 1:
            risk = 0.8
            factors.append(f"Short employment history ({applicant.employment_years} years)")
            mitigations.extend([
                "Require employment verification letter",
                "Higher interest rate for new employment risk",
                "Probationary period with payment monitoring"
            ])
        elif applicant.employment_years < 2:
            risk = 0.5
            factors.append(f"Limited employment history ({applicant.employment_years} years)")
            mitigations.append("Quarterly employment verification")
        elif applicant.employment_years < 5:
            risk = 0.2
            # Stable employment, minimal risk
        else:
            risk = 0.1
            # Very stable employment
        
        return risk, factors, mitigations
    
    def _assess_loan_to_income(self, applicant: LoanApplicant) -> Tuple[float, List[str], List[str]]:
        """Assess risk based on loan amount relative to income"""
        factors = []
        mitigations = []
        loan_to_income = applicant.loan_amount / applicant.annual_income
        
        if loan_to_income > 5:
            risk = 1.0
            factors.append(f"Very high loan-to-income ratio ({loan_to_income:.1f}x)")
            mitigations.extend([
                "Require substantial collateral",
                "Co-signer mandatory",
                "Reduce loan amount significantly"
            ])
        elif loan_to_income > 3:
            risk = 0.7
            factors.append(f"High loan-to-income ratio ({loan_to_income:.1f}x)")
            mitigations.extend([
                "Require additional income documentation",
                "Consider collateral requirements"
            ])
        elif loan_to_income > 2:
            risk = 0.4
            factors.append(f"Moderate loan-to-income ratio ({loan_to_income:.1f}x)")
            mitigations.append("Enhanced income verification")
        else:
            risk = 0.1
        
        return risk, factors, mitigations
    
    def _assess_payment_history(self, applicant: LoanApplicant) -> Tuple[float, List[str], List[str]]:
        """Assess risk based on payment history and defaults"""
        factors = []
        mitigations = []
        
        if applicant.bankruptcy_history:
            risk = 0.9
            factors.append("Previous bankruptcy on record")
            mitigations.extend([
                "Require 2+ years since bankruptcy discharge",
                "Secured loan only",
                "Financial counseling mandatory"
            ])
        elif applicant.previous_defaults > 2:
            risk = 0.8
            factors.append(f"Multiple previous defaults ({applicant.previous_defaults})")
            mitigations.extend([
                "Require explanation letters for each default",
                "Higher interest rate premium",
                "Shorter loan term"
            ])
        elif applicant.previous_defaults > 0:
            risk = 0.5
            factors.append(f"Previous default history ({applicant.previous_defaults})")
            mitigations.append("Enhanced monitoring and early intervention program")
        elif applicant.payment_history_months < 12:
            risk = 0.4
            factors.append("Limited payment history available")
            mitigations.append("Start with smaller loan amount to build relationship")
        else:
            risk = 0.1
        
        return risk, factors, mitigations
    
    def _determine_risk_level(self, risk_score: float) -> RiskLevel:
        """Determine overall risk level"""
        if risk_score >= 0.7:
            return RiskLevel.VERY_HIGH
        elif risk_score >= 0.5:
            return RiskLevel.HIGH
        elif risk_score >= 0.3:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _calculate_default_probability(self, risk_score: float, applicant: LoanApplicant) -> float:
        """Calculate probability of default based on risk factors"""
        base_probability = risk_score * 0.15  # Base 15% max default rate
        
        # Adjust for loan type risk
        loan_type_multipliers = {
            LoanType.MORTGAGE: 0.8,    # Lower risk due to collateral
            LoanType.AUTO: 0.9,        # Moderate risk
            LoanType.PERSONAL: 1.2,    # Higher risk, unsecured
            LoanType.BUSINESS: 1.3,    # Higher risk, variable income
            LoanType.CREDIT_CARD: 1.5  # Highest risk, revolving credit
        }
        
        multiplier = loan_type_multipliers.get(applicant.loan_type, 1.0)
        return min(base_probability * multiplier, 0.25)  # Cap at 25%
    
    def _calculate_interest_rate(self, risk_score: float, loan_type: LoanType) -> float:
        """Calculate recommended interest rate"""
        base_rate = self.base_rates[loan_type]
        risk_premium = risk_score * 0.08  # Up to 8% risk premium
        return base_rate + risk_premium
    
    def _make_loan_decision(self, risk_level: RiskLevel, probability_of_default: float) -> str:
        """Make final loan decision"""
        if risk_level == RiskLevel.VERY_HIGH or probability_of_default > 0.20:
            return "DECLINE - Risk too high"
        elif risk_level == RiskLevel.HIGH:
            return "CONDITIONAL APPROVAL - With enhanced terms and monitoring"
        elif risk_level == RiskLevel.MEDIUM:
            return "APPROVE - With standard risk mitigation measures"
        else:
            return "APPROVE - Standard terms"

def create_sample_applicants() -> List[LoanApplicant]:
    """Create sample loan applicants representing different risk scenarios"""
    
    return [
        # Scenario 1: High Risk - Recent Graduate with Student Loans
        LoanApplicant(
            name="Sarah Johnson - Recent Graduate",
            age=23,
            annual_income=45000,
            credit_score=640,
            employment_years=0.5,
            debt_to_income_ratio=0.45,  # High due to student loans
            loan_amount=25000,
            loan_type=LoanType.PERSONAL,
            previous_defaults=0,
            payment_history_months=8
        ),
        
        # Scenario 2: Medium Risk - Mid-Career Professional
        LoanApplicant(
            name="Michael Chen - Software Engineer",
            age=32,
            annual_income=85000,
            credit_score=720,
            employment_years=3.5,
            debt_to_income_ratio=0.32,
            loan_amount=35000,
            loan_type=LoanType.AUTO,
            collateral_value=40000,
            previous_defaults=0,
            payment_history_months=48
        ),
        
        # Scenario 3: Low Risk - Established Professional
        LoanApplicant(
            name="Jennifer Martinez - Doctor",
            age=38,
            annual_income=180000,
            credit_score=780,
            employment_years=8,
            debt_to_income_ratio=0.25,
            loan_amount=500000,
            loan_type=LoanType.MORTGAGE,
            collateral_value=600000,
            previous_defaults=0,
            payment_history_months=120
        ),
        
        # Scenario 4: Very High Risk - Previous Bankruptcy
        LoanApplicant(
            name="Robert Williams - Small Business Owner",
            age=45,
            annual_income=60000,
            credit_score=580,
            employment_years=2,
            debt_to_income_ratio=0.48,
            loan_amount=50000,
            loan_type=LoanType.BUSINESS,
            previous_defaults=1,
            bankruptcy_history=True,
            payment_history_months=24
        ),
        
        # Scenario 5: Medium-High Risk - Gig Economy Worker
        LoanApplicant(
            name="Lisa Thompson - Freelance Designer",
            age=29,
            annual_income=42000,
            credit_score=680,
            employment_years=1.5,
            debt_to_income_ratio=0.38,
            loan_amount=15000,
            loan_type=LoanType.PERSONAL,
            previous_defaults=1,
            payment_history_months=36
        )
    ]

def demonstrate_credit_risk_analysis():
    """Demonstrate credit risk analysis with real-world examples"""
    
    print("=" * 80)
    print("REAL-WORLD CREDIT RISK ANALYSIS FOR LOAN APPLICANTS")
    print("=" * 80)
    print()
    
    analyzer = CreditRiskAnalyzer()
    applicants = create_sample_applicants()
    
    for i, applicant in enumerate(applicants, 1):
        print(f"\n{'='*60}")
        print(f"SCENARIO {i}: {applicant.name}")
        print(f"{'='*60}")
        
        # Display applicant profile
        print(f"\nAPPLICANT PROFILE:")
        print(f"Age: {applicant.age}")
        print(f"Annual Income: ${applicant.annual_income:,.0f}")
        print(f"Credit Score: {applicant.credit_score}")
        print(f"Employment Years: {applicant.employment_years}")
        print(f"Debt-to-Income Ratio: {applicant.debt_to_income_ratio:.1%}")
        print(f"Loan Amount: ${applicant.loan_amount:,.0f}")
        print(f"Loan Type: {applicant.loan_type.value}")
        if applicant.collateral_value:
            print(f"Collateral Value: ${applicant.collateral_value:,.0f}")
        if applicant.previous_defaults > 0:
            print(f"Previous Defaults: {applicant.previous_defaults}")
        if applicant.bankruptcy_history:
            print(f"Bankruptcy History: Yes")
        
        # Perform risk assessment
        assessment = analyzer.assess_credit_risk(applicant)
        
        # Display results
        print(f"\nRISK ASSESSMENT RESULTS:")
        print(f"Risk Level: {assessment.risk_level.value}")
        print(f"Probability of Default: {assessment.probability_of_default:.1%}")
        print(f"Recommended Interest Rate: {assessment.recommended_interest_rate:.2%}")
        print(f"Loan Decision: {assessment.loan_decision}")
        
        print(f"\nIDENTIFIED RISK FACTORS:")
        for factor in assessment.risk_factors:
            print(f"  • {factor}")
        
        print(f"\nRECOMMENDED MITIGATION STRATEGIES:")
        for strategy in assessment.mitigation_strategies:
            print(f"  • {strategy}")
        
        print(f"\n{'-'*60}")

if __name__ == "__main__":
    demonstrate_credit_risk_analysis()