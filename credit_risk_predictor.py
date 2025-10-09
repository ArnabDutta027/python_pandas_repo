"""
Credit Risk Prediction using Decision Tree Algorithm
This script predicts whether a loan applicant is a credit risk using a Decision Tree Classifier.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data(n_samples=1000):
    """Create a sample credit risk dataset for demonstration"""
    
    # Generate synthetic data
    data = {
        'Age': np.random.randint(18, 70, n_samples),
        'Income': np.random.randint(20000, 150000, n_samples),
        'Loan_Amount': np.random.randint(5000, 50000, n_samples),
        'Credit_Score': np.random.randint(300, 850, n_samples),
        'Employment_Years': np.random.randint(0, 40, n_samples),
        'Existing_Loans': np.random.randint(0, 5, n_samples),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples),
        'Marital_Status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples),
        'Home_Ownership': np.random.choice(['Rent', 'Own', 'Mortgage'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on certain conditions (simulating credit risk)
    # High risk if: low credit score, high loan amount relative to income, or many existing loans
    df['Credit_Risk'] = 'Low Risk'
    
    conditions = (
        (df['Credit_Score'] < 600) |
        (df['Loan_Amount'] / df['Income'] > 0.3) |
        (df['Existing_Loans'] >= 3) |
        (df['Employment_Years'] < 2)
    )
    
    df.loc[conditions, 'Credit_Risk'] = 'High Risk'
    
    return df


def preprocess_data(df):
    """Preprocess the data for model training"""
    
    # Create a copy to avoid modifying original data
    df_processed = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    categorical_columns = ['Education', 'Marital_Status', 'Home_Ownership']
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        label_encoders[col] = le
    
    # Encode target variable
    le_target = LabelEncoder()
    df_processed['Credit_Risk'] = le_target.fit_transform(df_processed['Credit_Risk'])
    label_encoders['Credit_Risk'] = le_target
    
    return df_processed, label_encoders


def train_decision_tree(X_train, y_train, max_depth=5):
    """Train a Decision Tree Classifier"""
    
    # Create and train the model
    dt_classifier = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    dt_classifier.fit(X_train, y_train)
    
    return dt_classifier


def evaluate_model(model, X_test, y_test, label_encoder):
    """Evaluate the model and print metrics"""
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    print("=" * 50)
    print("MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"\nAccuracy Score: {accuracy_score(y_test, y_pred):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    target_names = label_encoder.classes_
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    return y_pred, cm


def visualize_results(model, feature_names, class_names, cm, output_file='decision_tree_visualization.png'):
    """Visualize the decision tree and confusion matrix"""
    
    # Create a figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(20, 16))
    
    # Plot decision tree
    plot_tree(
        model,
        filled=True,
        feature_names=feature_names,
        class_names=class_names,
        rounded=True,
        ax=axes[0],
        fontsize=10
    )
    axes[0].set_title('Decision Tree for Credit Risk Prediction', fontsize=16, fontweight='bold')
    
    # Plot confusion matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1]
    )
    axes[1].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Actual', fontsize=12)
    axes[1].set_xlabel('Predicted', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_file}")
    plt.close()


def predict_new_applicant(model, label_encoders, applicant_data):
    """Predict credit risk for a new loan applicant"""
    
    # Create a DataFrame for the new applicant
    df_applicant = pd.DataFrame([applicant_data])
    
    # Encode categorical variables
    categorical_columns = ['Education', 'Marital_Status', 'Home_Ownership']
    for col in categorical_columns:
        df_applicant[col] = label_encoders[col].transform(df_applicant[col])
    
    # Make prediction
    prediction = model.predict(df_applicant)
    risk_label = label_encoders['Credit_Risk'].inverse_transform(prediction)[0]
    
    # Get prediction probability
    probabilities = model.predict_proba(df_applicant)[0]
    
    print("\n" + "=" * 50)
    print("NEW APPLICANT PREDICTION")
    print("=" * 50)
    print(f"\nApplicant Details:")
    for key, value in applicant_data.items():
        print(f"  {key}: {value}")
    
    print(f"\nPredicted Credit Risk: {risk_label}")
    print(f"Confidence: {max(probabilities) * 100:.2f}%")
    print(f"Risk Probabilities: High Risk={probabilities[0]*100:.2f}%, Low Risk={probabilities[1]*100:.2f}%")
    
    return risk_label, probabilities


def main():
    """Main function to run the credit risk prediction pipeline"""
    
    print("Credit Risk Prediction using Decision Tree")
    print("=" * 50)
    
    # Step 1: Create sample data
    print("\n1. Creating sample credit risk dataset...")
    df = create_sample_data(n_samples=1000)
    print(f"   Dataset created with {len(df)} samples")
    print(f"   Features: {list(df.columns[:-1])}")
    print(f"\n   Data preview:")
    print(df.head())
    
    # Save dataset to CSV
    df.to_csv('credit_risk_dataset.csv', index=False)
    print("\n   Dataset saved to: credit_risk_dataset.csv")
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    df_processed, label_encoders = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Credit_Risk', axis=1)
    y = df_processed['Credit_Risk']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Step 3: Train the model
    print("\n3. Training Decision Tree Classifier...")
    model = train_decision_tree(X_train, y_train, max_depth=5)
    print("   Model training complete!")
    
    # Print feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n   Feature Importance:")
    for idx, row in feature_importance.iterrows():
        print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    # Step 4: Evaluate the model
    print("\n4. Evaluating model...")
    y_pred, cm = evaluate_model(model, X_test, y_test, label_encoders['Credit_Risk'])
    
    # Step 5: Visualize results
    print("\n5. Creating visualizations...")
    visualize_results(
        model,
        feature_names=list(X.columns),
        class_names=label_encoders['Credit_Risk'].classes_,
        cm=cm
    )
    
    # Step 6: Make predictions for new applicants
    print("\n6. Testing with new loan applicants...")
    
    # Example 1: Low risk applicant
    low_risk_applicant = {
        'Age': 35,
        'Income': 80000,
        'Loan_Amount': 15000,
        'Credit_Score': 750,
        'Employment_Years': 10,
        'Existing_Loans': 1,
        'Education': 'Bachelor',
        'Marital_Status': 'Married',
        'Home_Ownership': 'Own'
    }
    
    predict_new_applicant(model, label_encoders, low_risk_applicant)
    
    # Example 2: High risk applicant
    high_risk_applicant = {
        'Age': 25,
        'Income': 30000,
        'Loan_Amount': 40000,
        'Credit_Score': 550,
        'Employment_Years': 1,
        'Existing_Loans': 3,
        'Education': 'High School',
        'Marital_Status': 'Single',
        'Home_Ownership': 'Rent'
    }
    
    predict_new_applicant(model, label_encoders, high_risk_applicant)
    
    print("\n" + "=" * 50)
    print("Credit Risk Prediction Complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
