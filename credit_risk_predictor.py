"""
Credit Risk Prediction using Decision Tree Algorithm
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score, roc_curve)
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class CreditRiskPredictor:
    """
    A comprehensive credit risk prediction system using Decision Trees
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = None
        self.target_names = ['Low Risk', 'High Risk']
        
    def load_data(self, filepath='loan_data.csv'):
        """Load the loan dataset"""
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully! Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"File {filepath} not found. Please ensure the data file exists.")
            return None
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic info
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nFirst 5 rows:")
        print(self.data.head())
        
        print("\nStatistical Summary:")
        print(self.data.describe())
        
        print("\nTarget Variable Distribution:")
        target_counts = self.data['credit_risk'].value_counts()
        print(target_counts)
        print(f"Low Risk: {target_counts[0]} ({target_counts[0]/len(self.data)*100:.1f}%)")
        print(f"High Risk: {target_counts[1]} ({target_counts[1]/len(self.data)*100:.1f}%)")
        
        # Check for missing values
        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found!")
        else:
            print(missing[missing > 0])
        
        # Correlation analysis for numerical features
        print("\nCorrelation with Target Variable:")
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        correlations = self.data[numerical_cols].corr()['credit_risk'].sort_values(key=abs, ascending=False)
        print(correlations[1:])  # Exclude self-correlation
        
    def preprocess_data(self):
        """Preprocess the data for machine learning"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Separate features and target
        X = self.data.drop('credit_risk', axis=1)
        y = self.data['credit_risk']
        
        # Handle categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        print(f"\nCategorical columns: {list(categorical_cols)}")
        
        X_processed = X.copy()
        
        # Label encode categorical variables
        for col in categorical_cols:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
            print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
        
        self.feature_names = X_processed.columns.tolist()
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_decision_tree(self, optimize_hyperparameters=True):
        """Train the decision tree model"""
        print("\n" + "="*50)
        print("DECISION TREE TRAINING")
        print("="*50)
        
        if optimize_hyperparameters:
            print("Performing hyperparameter optimization...")
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'criterion': ['gini', 'entropy'],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Grid search with cross-validation
            dt = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(
                dt, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        else:
            # Train with default parameters
            self.model = DecisionTreeClassifier(
                max_depth=7, min_samples_split=10, min_samples_leaf=5,
                criterion='gini', random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5, scoring='f1')
        print(f"Cross-validation F1 scores: {cv_scores}")
        print(f"Mean CV F1 score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self.model
    
    def evaluate_model(self):
        """Evaluate the trained model"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Predictions
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        y_test_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Training metrics
        print("TRAINING SET PERFORMANCE:")
        print(f"Accuracy: {accuracy_score(self.y_train, y_train_pred):.4f}")
        print(f"Precision: {precision_score(self.y_train, y_train_pred):.4f}")
        print(f"Recall: {recall_score(self.y_train, y_train_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_train, y_train_pred):.4f}")
        
        # Test metrics
        print("\nTEST SET PERFORMANCE:")
        print(f"Accuracy: {accuracy_score(self.y_test, y_test_pred):.4f}")
        print(f"Precision: {precision_score(self.y_test, y_test_pred):.4f}")
        print(f"Recall: {recall_score(self.y_test, y_test_pred):.4f}")
        print(f"F1-Score: {f1_score(self.y_test, y_test_pred):.4f}")
        print(f"ROC-AUC: {roc_auc_score(self.y_test, y_test_proba):.4f}")
        
        # Detailed classification report
        print("\nDETAILED CLASSIFICATION REPORT:")
        print(classification_report(self.y_test, y_test_pred, 
                                  target_names=self.target_names))
        
        # Confusion Matrix
        cm = confusion_matrix(self.y_test, y_test_pred)
        print("\nCONFUSION MATRIX:")
        print(cm)
        
        return {
            'accuracy': accuracy_score(self.y_test, y_test_pred),
            'precision': precision_score(self.y_test, y_test_pred),
            'recall': recall_score(self.y_test, y_test_pred),
            'f1': f1_score(self.y_test, y_test_pred),
            'roc_auc': roc_auc_score(self.y_test, y_test_proba)
        }
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        # Get feature importances
        importances = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Feature Importance Ranking:")
        for i, row in feature_importance_df.iterrows():
            print(f"{row['feature']}: {row['importance']:.4f}")
        
        return feature_importance_df
    
    def visualize_results(self):
        """Create visualizations for the results"""
        print("\n" + "="*50)
        print("CREATING VISUALIZATIONS")
        print("="*50)
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Feature Importance Plot
        plt.subplot(2, 3, 1)
        importance_df = self.feature_importance_analysis()
        plt.barh(importance_df['feature'], importance_df['importance'])
        plt.title('Feature Importance in Decision Tree')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()
        
        # 2. Confusion Matrix Heatmap
        plt.subplot(2, 3, 2)
        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.target_names,
                   yticklabels=self.target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        # 3. ROC Curve
        plt.subplot(2, 3, 3)
        y_proba = self.model.predict_proba(self.X_test)[:, 1]
        fpr, tpr, _ = roc_curve(self.y_test, y_proba)
        auc_score = roc_auc_score(self.y_test, y_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        
        # 4. Target Distribution
        plt.subplot(2, 3, 4)
        target_counts = self.data['credit_risk'].value_counts()
        plt.pie(target_counts.values, labels=self.target_names, autopct='%1.1f%%')
        plt.title('Credit Risk Distribution')
        
        # 5. Credit Score vs Risk
        plt.subplot(2, 3, 5)
        plt.boxplot([self.data[self.data['credit_risk']==0]['credit_score'],
                    self.data[self.data['credit_risk']==1]['credit_score']],
                   labels=self.target_names)
        plt.title('Credit Score Distribution by Risk')
        plt.ylabel('Credit Score')
        
        # 6. Income vs Risk
        plt.subplot(2, 3, 6)
        plt.boxplot([self.data[self.data['credit_risk']==0]['annual_income'],
                    self.data[self.data['credit_risk']==1]['annual_income']],
                   labels=self.target_names)
        plt.title('Annual Income Distribution by Risk')
        plt.ylabel('Annual Income')
        
        plt.tight_layout()
        plt.savefig('credit_risk_analysis.png', dpi=300, bbox_inches='tight')
        print("Visualizations saved as 'credit_risk_analysis.png'")
        
        # Decision Tree Visualization (simplified)
        plt.figure(figsize=(20, 12))
        plot_tree(self.model, max_depth=3, 
                 feature_names=self.feature_names,
                 class_names=self.target_names,
                 filled=True, fontsize=10)
        plt.title('Decision Tree (First 3 Levels)')
        plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
        print("Decision tree visualization saved as 'decision_tree_visualization.png'")
        
    def predict_single_applicant(self, applicant_data):
        """Predict credit risk for a single applicant"""
        print("\n" + "="*50)
        print("SINGLE APPLICANT PREDICTION")
        print("="*50)
        
        # Convert to DataFrame
        if isinstance(applicant_data, dict):
            applicant_df = pd.DataFrame([applicant_data])
        else:
            applicant_df = applicant_data.copy()
        
        # Encode categorical variables
        for col in applicant_df.select_dtypes(include=['object']).columns:
            if col in self.label_encoders:
                applicant_df[col] = self.label_encoders[col].transform(applicant_df[col])
        
        # Make prediction
        prediction = self.model.predict(applicant_df)[0]
        probability = self.model.predict_proba(applicant_df)[0]
        
        print(f"Applicant Data:")
        for col, val in applicant_data.items():
            print(f"  {col}: {val}")
        
        print(f"\nPrediction: {self.target_names[prediction]}")
        print(f"Probability of Low Risk: {probability[0]:.3f}")
        print(f"Probability of High Risk: {probability[1]:.3f}")
        
        return prediction, probability
    
    def get_decision_rules(self):
        """Extract decision rules from the tree"""
        print("\n" + "="*50)
        print("DECISION TREE RULES")
        print("="*50)
        
        tree_rules = export_text(self.model, 
                                feature_names=self.feature_names,
                                class_names=self.target_names)
        print(tree_rules)
        
        return tree_rules

def main():
    """Main function to run the complete credit risk prediction pipeline"""
    print("="*60)
    print("CREDIT RISK PREDICTION USING DECISION TREES")
    print("="*60)
    
    # Initialize the predictor
    predictor = CreditRiskPredictor()
    
    # Load and explore data
    data = predictor.load_data()
    if data is None:
        return
    
    predictor.explore_data()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train model
    predictor.train_decision_tree(optimize_hyperparameters=True)
    
    # Evaluate model
    metrics = predictor.evaluate_model()
    
    # Feature importance
    predictor.feature_importance_analysis()
    
    # Create visualizations
    predictor.visualize_results()
    
    # Show decision rules
    predictor.get_decision_rules()
    
    # Example prediction for a new applicant
    print("\n" + "="*50)
    print("EXAMPLE PREDICTION")
    print("="*50)
    
    sample_applicant = {
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
    
    predictor.predict_single_applicant(sample_applicant)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"Final Model Performance:")
    print(f"  Accuracy: {metrics['accuracy']:.3f}")
    print(f"  F1-Score: {metrics['f1']:.3f}")
    print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")

if __name__ == "__main__":
    main()