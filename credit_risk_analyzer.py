"""
Credit Risk Analyzer - Real-World Examples and Solutions
Demonstrates credit risk assessment for loan applicants with practical solutions
"""

from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum
import json


class RiskLevel(Enum):
    LOW = "Low Risk"
    MODERATE = "Moderate Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"


class LoanType(Enum):
    MORTGAGE = "Mortgage"
    AUTO = "Auto Loan"
    PERSONAL = "Personal Loan"
    BUSINESS = "Business Loan"


@dataclass
class Applicant:
    name: str
    annual_income: float
    monthly_income: float
    credit_score: int
    existing_monthly_debts: float
    employment_type: str
    employment_years: float
    requested_loan_amount: float
    loan_type: LoanType
    down_payment: float = 0
    additional_info: Dict = None


@dataclass
class RiskAssessment:
    applicant_name: str
    risk_level: RiskLevel
    dti_ratio: float
    ltv_ratio: float
    approval_recommendation: str
    issues_identified: List[str]
    solutions_proposed: List[str]
    modified_terms: Dict


class CreditRiskAnalyzer:
    """Analyzes credit risk and proposes solutions for loan applicants"""
    
    # Industry standard thresholds
    MAX_DTI_RATIO = 43.0  # Maximum debt-to-income ratio
    MIN_CREDIT_SCORE = 620  # Minimum acceptable credit score
    MAX_LTV_RATIO = 80.0  # Maximum loan-to-value ratio for mortgages
    
    def calculate_dti(self, monthly_income: float, total_monthly_debts: float) -> float:
        """Calculate debt-to-income ratio"""
        if monthly_income <= 0:
            return float('inf')
        return (total_monthly_debts / monthly_income) * 100
    
    def calculate_ltv(self, loan_amount: float, collateral_value: float) -> float:
        """Calculate loan-to-value ratio"""
        if collateral_value <= 0:
            return 100.0
        return (loan_amount / collateral_value) * 100
    
    def estimate_monthly_payment(self, loan_amount: float, loan_type: LoanType, 
                                 annual_rate: float = 6.5, years: int = 30) -> float:
        """Estimate monthly loan payment"""
        if loan_type == LoanType.MORTGAGE:
            # Mortgage calculation (principal + interest)
            monthly_rate = annual_rate / 100 / 12
            num_payments = years * 12
            payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                     ((1 + monthly_rate)**num_payments - 1)
            return payment
        elif loan_type == LoanType.AUTO:
            # Auto loan (typically 5-7 years)
            monthly_rate = (annual_rate + 1) / 100 / 12
            num_payments = 5 * 12
            payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                     ((1 + monthly_rate)**num_payments - 1)
            return payment
        else:
            # Personal/Business loan (typically 3-5 years)
            monthly_rate = (annual_rate + 2) / 100 / 12
            num_payments = 5 * 12
            payment = loan_amount * (monthly_rate * (1 + monthly_rate)**num_payments) / \
                     ((1 + monthly_rate)**num_payments - 1)
            return payment
    
    def assess_credit_score_risk(self, credit_score: int) -> Tuple[RiskLevel, str]:
        """Assess risk based on credit score"""
        if credit_score >= 740:
            return RiskLevel.LOW, "Excellent credit score - very low default risk"
        elif credit_score >= 670:
            return RiskLevel.LOW, "Good credit score - low default risk"
        elif credit_score >= 620:
            return RiskLevel.MODERATE, "Fair credit score - moderate default risk"
        elif credit_score >= 580:
            return RiskLevel.HIGH, "Subprime credit score - high default risk"
        else:
            return RiskLevel.CRITICAL, "Poor credit score - very high default risk"
    
    def analyze_applicant(self, applicant: Applicant) -> RiskAssessment:
        """Complete risk analysis with solutions"""
        
        issues = []
        solutions = []
        modified_terms = {}
        
        # Calculate estimated monthly payment
        estimated_payment = self.estimate_monthly_payment(
            applicant.requested_loan_amount, 
            applicant.loan_type
        )
        
        # Calculate DTI with new loan
        total_monthly_debt = applicant.existing_monthly_debts + estimated_payment
        dti_ratio = self.calculate_dti(applicant.monthly_income, total_monthly_debt)
        
        # Calculate LTV if applicable
        ltv_ratio = 0
        if applicant.loan_type == LoanType.MORTGAGE or applicant.loan_type == LoanType.AUTO:
            collateral_value = applicant.requested_loan_amount + applicant.down_payment
            ltv_ratio = self.calculate_ltv(applicant.requested_loan_amount, collateral_value)
        
        # Assess credit score
        credit_risk, credit_msg = self.assess_credit_score_risk(applicant.credit_score)
        
        # Initialize risk level
        risk_level = RiskLevel.LOW
        
        # Check DTI ratio
        if dti_ratio > self.MAX_DTI_RATIO:
            risk_level = RiskLevel.HIGH if dti_ratio < 50 else RiskLevel.CRITICAL
            issues.append(f"DTI ratio of {dti_ratio:.1f}% exceeds maximum {self.MAX_DTI_RATIO}%")
            
            # Propose solutions
            # Solution 1: Reduce loan amount
            max_payment = (applicant.monthly_income * (self.MAX_DTI_RATIO / 100)) - \
                         applicant.existing_monthly_debts
            if max_payment > 0:
                # Reverse calculate max loan amount
                max_loan = max_payment * 150  # Rough estimate
                solutions.append(
                    f"Reduce loan amount from ${applicant.requested_loan_amount:,.0f} to "
                    f"${max_loan:,.0f} to meet DTI requirements"
                )
                modified_terms['max_loan_amount'] = max_loan
            
            # Solution 2: Increase down payment
            if applicant.loan_type in [LoanType.MORTGAGE, LoanType.AUTO]:
                increased_down = applicant.down_payment * 1.5
                new_loan = applicant.requested_loan_amount - (increased_down - applicant.down_payment)
                solutions.append(
                    f"Increase down payment from ${applicant.down_payment:,.0f} to "
                    f"${increased_down:,.0f} to reduce monthly payment"
                )
                modified_terms['recommended_down_payment'] = increased_down
            
            # Solution 3: Add co-borrower
            required_income = (total_monthly_debt / self.MAX_DTI_RATIO) * 100
            additional_income_needed = required_income - applicant.monthly_income
            solutions.append(
                f"Add co-borrower with minimum monthly income of ${additional_income_needed:,.0f}"
            )
        
        # Check credit score
        if applicant.credit_score < self.MIN_CREDIT_SCORE:
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.HIGH
            else:
                risk_level = RiskLevel.CRITICAL
            
            issues.append(f"Credit score {applicant.credit_score} below minimum {self.MIN_CREDIT_SCORE}")
            solutions.append("Complete credit rehabilitation program for 6-12 months")
            solutions.append("Consider secured loan option with collateral")
            solutions.append("Resolve any outstanding collections or delinquencies")
            modified_terms['required_collateral'] = applicant.requested_loan_amount * 1.2
        
        # Check LTV ratio
        if ltv_ratio > self.MAX_LTV_RATIO and ltv_ratio > 0:
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MODERATE
            
            issues.append(f"LTV ratio of {ltv_ratio:.1f}% exceeds maximum {self.MAX_LTV_RATIO}%")
            required_down = applicant.requested_loan_amount * 0.25  # 25% down for 75% LTV
            solutions.append(
                f"Increase down payment to ${required_down:,.0f} for 75% LTV ratio"
            )
            modified_terms['required_down_payment_ltv'] = required_down
        
        # Check employment stability
        if applicant.employment_years < 2:
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MODERATE
            
            issues.append(f"Limited employment history ({applicant.employment_years} years)")
            solutions.append("Provide additional income verification (tax returns, bank statements)")
            solutions.append("Consider smaller loan amount with option to refinance in 12 months")
        
        # Check self-employment
        if applicant.employment_type.lower() == "self-employed":
            if risk_level == RiskLevel.LOW:
                risk_level = RiskLevel.MODERATE
            
            issues.append("Self-employment income volatility risk")
            solutions.append("Require 3 years of tax returns instead of 2")
            solutions.append("Maintain 12 months of reserves in liquid assets")
            solutions.append("Use conservative income calculation (lowest 2-year average)")
            modified_terms['reserve_requirement_months'] = 12
        
        # Determine approval recommendation
        if risk_level == RiskLevel.CRITICAL:
            approval = "DENY - Critical risk factors present. Recommend reapplying after addressing issues."
        elif risk_level == RiskLevel.HIGH:
            approval = "CONDITIONAL - High risk. Approve only with significant modifications and enhanced terms."
            modified_terms['interest_rate_adjustment'] = "+2.5% to base rate"
        elif risk_level == RiskLevel.MODERATE:
            approval = "APPROVE WITH CONDITIONS - Moderate risk. Require some modifications."
            modified_terms['interest_rate_adjustment'] = "+1.0% to base rate"
        else:
            approval = "APPROVE - Low risk applicant meeting all standards."
            modified_terms['interest_rate_adjustment'] = "Standard rate"
        
        # Add credit score insight
        issues.append(credit_msg)
        
        return RiskAssessment(
            applicant_name=applicant.name,
            risk_level=risk_level,
            dti_ratio=dti_ratio,
            ltv_ratio=ltv_ratio,
            approval_recommendation=approval,
            issues_identified=issues,
            solutions_proposed=solutions,
            modified_terms=modified_terms
        )
    
    def print_assessment(self, assessment: RiskAssessment):
        """Print formatted risk assessment"""
        print(f"\n{'='*80}")
        print(f"CREDIT RISK ASSESSMENT: {assessment.applicant_name}")
        print(f"{'='*80}")
        print(f"\nRISK LEVEL: {assessment.risk_level.value}")
        print(f"DTI Ratio: {assessment.dti_ratio:.1f}%")
        if assessment.ltv_ratio > 0:
            print(f"LTV Ratio: {assessment.ltv_ratio:.1f}%")
        
        print(f"\n{'-'*80}")
        print("ISSUES IDENTIFIED:")
        for i, issue in enumerate(assessment.issues_identified, 1):
            print(f"  {i}. {issue}")
        
        if assessment.solutions_proposed:
            print(f"\n{'-'*80}")
            print("PROPOSED SOLUTIONS:")
            for i, solution in enumerate(assessment.solutions_proposed, 1):
                print(f"  {i}. {solution}")
        
        if assessment.modified_terms:
            print(f"\n{'-'*80}")
            print("MODIFIED TERMS:")
            for key, value in assessment.modified_terms.items():
                if isinstance(value, float):
                    print(f"  - {key.replace('_', ' ').title()}: ${value:,.2f}")
                else:
                    print(f"  - {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n{'-'*80}")
        print(f"RECOMMENDATION: {assessment.approval_recommendation}")
        print(f"{'='*80}\n")


def main():
    """Run credit risk analysis on real-world examples"""
    
    analyzer = CreditRiskAnalyzer()
    
    # Example 1: High DTI Ratio
    print("\n" + "="*80)
    print("EXAMPLE 1: HIGH DEBT-TO-INCOME RATIO")
    print("="*80)
    
    applicant1 = Applicant(
        name="Sarah Johnson",
        annual_income=60000,
        monthly_income=5000,
        credit_score=720,
        existing_monthly_debts=1700,  # Auto $400 + CC $800 + Student $500
        employment_type="W-2 Employee",
        employment_years=5,
        requested_loan_amount=250000,
        loan_type=LoanType.MORTGAGE,
        down_payment=25000,
        additional_info={"occupation": "Marketing Manager"}
    )
    
    assessment1 = analyzer.analyze_applicant(applicant1)
    analyzer.print_assessment(assessment1)
    
    # Example 2: Poor Credit History
    print("\n" + "="*80)
    print("EXAMPLE 2: POOR CREDIT HISTORY")
    print("="*80)
    
    applicant2 = Applicant(
        name="Michael Chen",
        annual_income=75000,
        monthly_income=6250,
        credit_score=580,
        existing_monthly_debts=800,
        employment_type="W-2 Employee",
        employment_years=3,
        requested_loan_amount=25000,
        loan_type=LoanType.PERSONAL,
        additional_info={
            "late_payments": 3,
            "collections": 1,
            "recent_unemployment": "1 year ago"
        }
    )
    
    assessment2 = analyzer.analyze_applicant(applicant2)
    analyzer.print_assessment(assessment2)
    
    # Example 3: Self-Employed with Variable Income
    print("\n" + "="*80)
    print("EXAMPLE 3: SELF-EMPLOYED APPLICANT")
    print("="*80)
    
    applicant3 = Applicant(
        name="David Martinez",
        annual_income=90000,  # Conservative estimate
        monthly_income=7500,
        credit_score=720,
        existing_monthly_debts=1200,
        employment_type="Self-Employed",
        employment_years=4,
        requested_loan_amount=400000,
        loan_type=LoanType.MORTGAGE,
        down_payment=80000,
        additional_info={
            "income_2022": 120000,
            "income_2023": 85000,
            "income_2024_projected": 95000
        }
    )
    
    assessment3 = analyzer.analyze_applicant(applicant3)
    analyzer.print_assessment(assessment3)
    
    # Example 4: First-Time Borrower (Thin Credit File)
    print("\n" + "="*80)
    print("EXAMPLE 4: FIRST-TIME BORROWER - THIN CREDIT FILE")
    print("="*80)
    
    applicant4 = Applicant(
        name="Emily Rodriguez",
        annual_income=55000,
        monthly_income=4583,
        credit_score=680,
        existing_monthly_debts=150,  # Just one credit card
        employment_type="W-2 Employee",
        employment_years=1.5,
        requested_loan_amount=18000,
        loan_type=LoanType.AUTO,
        down_payment=2000,
        additional_info={
            "credit_history_months": 18,
            "age": 24,
            "first_major_loan": True
        }
    )
    
    assessment4 = analyzer.analyze_applicant(applicant4)
    analyzer.print_assessment(assessment4)
    
    # Example 5: Good Applicant (Low Risk)
    print("\n" + "="*80)
    print("EXAMPLE 5: IDEAL APPLICANT - LOW RISK")
    print("="*80)
    
    applicant5 = Applicant(
        name="Jennifer Smith",
        annual_income=120000,
        monthly_income=10000,
        credit_score=780,
        existing_monthly_debts=1500,
        employment_type="W-2 Employee",
        employment_years=8,
        requested_loan_amount=300000,
        loan_type=LoanType.MORTGAGE,
        down_payment=75000,
        additional_info={
            "occupation": "Senior Software Engineer",
            "savings": 150000,
            "investments": 200000
        }
    )
    
    assessment5 = analyzer.analyze_applicant(applicant5)
    analyzer.print_assessment(assessment5)
    
    # Example 6: Business Loan - Multiple Risk Factors
    print("\n" + "="*80)
    print("EXAMPLE 6: BUSINESS LOAN - MULTIPLE RISK FACTORS")
    print("="*80)
    
    applicant6 = Applicant(
        name="Robert Taylor",
        annual_income=85000,
        monthly_income=7083,
        credit_score=650,
        existing_monthly_debts=2100,
        employment_type="Business Owner",
        employment_years=2.5,
        requested_loan_amount=50000,
        loan_type=LoanType.BUSINESS,
        additional_info={
            "recent_credit_inquiries": 8,
            "new_credit_cards": 3,
            "denied_loan_applications": 1
        }
    )
    
    assessment6 = analyzer.analyze_applicant(applicant6)
    analyzer.print_assessment(assessment6)
    
    # Summary Statistics
    print("\n" + "="*80)
    print("SUMMARY OF RISK ASSESSMENTS")
    print("="*80)
    
    applicants = [
        ("Sarah Johnson", assessment1.risk_level.value, assessment1.approval_recommendation.split('-')[0].strip()),
        ("Michael Chen", assessment2.risk_level.value, assessment2.approval_recommendation.split('-')[0].strip()),
        ("David Martinez", assessment3.risk_level.value, assessment3.approval_recommendation.split('-')[0].strip()),
        ("Emily Rodriguez", assessment4.risk_level.value, assessment4.approval_recommendation.split('-')[0].strip()),
        ("Jennifer Smith", assessment5.risk_level.value, assessment5.approval_recommendation.split('-')[0].strip()),
        ("Robert Taylor", assessment6.risk_level.value, assessment6.approval_recommendation.split('-')[0].strip()),
    ]
    
    print(f"\n{'Applicant':<25} {'Risk Level':<20} {'Decision':<20}")
    print("-" * 65)
    for name, risk, decision in applicants:
        print(f"{name:<25} {risk:<20} {decision:<20}")
    
    print("\n" + "="*80)
    print("KEY TAKEAWAYS:")
    print("="*80)
    print("""
1. DTI Ratio is Critical: Keep total debt payments below 43% of monthly income
2. Credit Score Matters: Scores below 620 face significant challenges
3. Employment Stability: 2+ years in current role is preferred
4. Self-Employment Requires Extra Documentation: Expect more scrutiny
5. Thin Credit Files Need Building: Start with smaller loans or co-signers
6. Multiple Solutions Available: Rarely is an application completely hopeless
7. Risk-Based Pricing: Higher risk = higher interest rates
8. Collateral Reduces Risk: Secured loans easier to obtain with poor credit
9. Down Payments Matter: Larger down payments reduce lender risk
10. Documentation is Key: Thorough records improve approval chances
    """)


if __name__ == "__main__":
    main()
