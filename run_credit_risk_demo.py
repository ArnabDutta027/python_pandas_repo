#!/usr/bin/env python3
"""
Credit Risk Analysis Demo Runner

This script demonstrates real-world credit risk examples and solutions.
Run this to see comprehensive credit risk analysis in action.
"""

import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from credit_risk_examples import demonstrate_credit_risk_analysis
from advanced_credit_solutions import demonstrate_advanced_solutions

def main():
    """Run complete credit risk demonstration"""
    
    print("🏦 COMPREHENSIVE CREDIT RISK ANALYSIS DEMONSTRATION")
    print("=" * 80)
    print()
    print("This demonstration covers:")
    print("1. Real-world credit risk scenarios")
    print("2. Risk assessment methodologies") 
    print("3. Mitigation strategies")
    print("4. Advanced ML-based solutions")
    print("5. Portfolio risk management")
    print()
    
    try:
        # Run basic credit risk analysis
        print("PART 1: BASIC CREDIT RISK ANALYSIS")
        print("=" * 50)
        demonstrate_credit_risk_analysis()
        
        print("\n" + "=" * 80)
        print()
        
        # Run advanced solutions
        print("PART 2: ADVANCED CREDIT RISK SOLUTIONS")
        print("=" * 50)
        demonstrate_advanced_solutions()
        
        print("\n" + "=" * 80)
        print("✅ DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)