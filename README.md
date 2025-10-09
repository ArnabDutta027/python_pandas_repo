# Credit Risk Prediction with Decision Trees

A comprehensive Python implementation for predicting credit risk of loan applicants using Decision Tree algorithms. This project includes data generation, preprocessing, model training, evaluation, and visualization components.

## 🎯 Project Overview

This system helps financial institutions assess the credit risk of loan applicants by analyzing various factors such as:
- Personal information (age, income, employment)
- Credit history and score
- Loan details
- Financial ratios (debt-to-income)

## 📊 Features

- **Synthetic Data Generation**: Creates realistic loan applicant datasets
- **Advanced Preprocessing**: Handles categorical encoding and data preparation
- **Hyperparameter Optimization**: Uses GridSearchCV for optimal model parameters
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Decision Tree Visualization**: Visual representation of decision rules
- **Single Applicant Prediction**: Easy-to-use prediction interface
- **Feature Importance Analysis**: Identifies key risk factors

## 🚀 Quick Start

### Prerequisites

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### Running the Complete Analysis

```bash
python3 credit_risk_predictor.py
```

### Running the Demo

```bash
python3 demo.py
```

### Generating New Dataset

```bash
python3 generate_loan_data.py
```

## 📁 Project Structure

```
├── credit_risk_predictor.py    # Main prediction system
├── generate_loan_data.py       # Dataset generation script
├── demo.py                     # Simple demo with test cases
├── loan_data.csv              # Generated dataset
├── requirements.txt           # Python dependencies
├── credit_risk_analysis.png   # Analysis visualizations
└── decision_tree_visualization.png  # Tree structure
```

## 🔍 Dataset Features

The synthetic dataset includes 11 features:

### Numerical Features:
- **age**: Applicant's age (18-80)
- **annual_income**: Annual income in USD (20k-200k)
- **employment_length**: Years of employment (0-40)
- **loan_amount**: Requested loan amount (1k-100k)
- **credit_score**: Credit score (300-850)
- **debt_to_income**: Debt-to-income ratio (0-1)
- **num_credit_lines**: Number of credit lines (0-20)
- **credit_history_years**: Years of credit history (0-50)

### Categorical Features:
- **education**: Education level (High School, Bachelor, Master, PhD, Associate)
- **home_ownership**: Housing status (Rent, Own, Mortgage, Other)
- **loan_purpose**: Purpose of loan (debt_consolidation, home_improvement, etc.)

### Target Variable:
- **credit_risk**: 0 = Low Risk, 1 = High Risk

## 🎯 Model Performance

The optimized Decision Tree achieves:
- **Accuracy**: ~84%
- **F1-Score**: ~79%
- **ROC-AUC**: ~91%
- **Precision**: ~84%
- **Recall**: ~75%

## 📈 Key Insights

Based on feature importance analysis:

1. **Debt-to-Income Ratio** (67.4%): Most critical factor
2. **Employment Length** (9.7%): Job stability indicator
3. **Credit Score** (8.8%): Traditional creditworthiness measure
4. **Annual Income** (7.6%): Financial capacity
5. **Credit History Years** (2.2%): Experience with credit

## 🔧 Model Configuration

The optimized Decision Tree uses:
- **Criterion**: Entropy
- **Max Depth**: 5
- **Min Samples Split**: 2
- **Min Samples Leaf**: 1
- **Max Features**: None (all features)

## 📊 Visualizations

The system generates:
1. **Feature Importance Plot**: Shows which factors matter most
2. **Confusion Matrix**: Model prediction accuracy breakdown
3. **ROC Curve**: Performance across different thresholds
4. **Distribution Plots**: Risk patterns by key features
5. **Decision Tree Structure**: Visual representation of decision rules

## 🎮 Usage Examples

### Basic Prediction

```python
from credit_risk_predictor import CreditRiskPredictor

# Initialize predictor
predictor = CreditRiskPredictor()
predictor.load_data()
predictor.preprocess_data()
predictor.train_decision_tree()

# Predict for new applicant
applicant = {
    'age': 35,
    'annual_income': 75000,
    'employment_length': 8,
    'loan_amount': 25000,
    'credit_score': 720,
    'debt_to_income': 0.35,
    'num_credit_lines': 5,
    'credit_history_years': 12,
    'education': 'Bachelor',
    'home_ownership': 'Own',
    'loan_purpose': 'debt_consolidation'
}

prediction, probability = predictor.predict_single_applicant(applicant)
```

### Custom Dataset

```python
from generate_loan_data import generate_loan_dataset

# Generate custom dataset
custom_data = generate_loan_dataset(n_samples=2000, random_state=123)
custom_data.to_csv('custom_loan_data.csv', index=False)
```

## 🎯 Decision Rules

The model learns interpretable rules like:
- If debt_to_income ≤ 0.60 AND employment_length ≤ 10.5 → Analyze credit_score
- If debt_to_income > 0.60 AND annual_income ≤ 80,316 → High Risk likely
- If credit_score > 728 AND loan_amount > 10,397 → Low Risk likely

## 🔮 Future Enhancements

Potential improvements:
- **Ensemble Methods**: Random Forest, Gradient Boosting
- **Deep Learning**: Neural networks for complex patterns
- **Real-time API**: Web service for instant predictions
- **Explainable AI**: SHAP values for prediction explanations
- **A/B Testing**: Model performance comparison framework

## 📝 Requirements

- Python 3.7+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙋‍♂️ Support

For questions or issues:
1. Check the code comments for detailed explanations
2. Review the decision tree rules for model logic
3. Examine feature importance for key risk factors
4. Use the demo script for testing different scenarios

---

**Note**: This is a demonstration project using synthetic data. For production use, ensure compliance with financial regulations and use real, validated datasets.