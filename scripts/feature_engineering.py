"""
Feature Engineering Module
Creates new features and performs feature selection
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class FeatureEngineer:
    def __init__(self, output_dir='../visualizations'):
        """
        Initialize Feature Engineering class
        
        Args:
            output_dir: Directory to save feature importance visualizations
        """
        self.output_dir = output_dir
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def create_interaction_features(self, X_train, X_val, X_test):
        """
        Create interaction features
        """
        print("\n" + "="*50)
        print("CREATING INTERACTION FEATURES")
        print("="*50)
        
        X_train_new = X_train.copy()
        X_val_new = X_val.copy()
        X_test_new = X_test.copy()
        
        # Feature 1: Spend per Purchase
        print("\n1. Creating 'Spend_per_Purchase' feature...")
        print("   Formula: Total_Spend / Num_of_Purchases")
        X_train_new['Spend_per_Purchase'] = X_train_new['Total_Spend'] / (X_train_new['Num_of_Purchases'] + 1)
        X_val_new['Spend_per_Purchase'] = X_val_new['Total_Spend'] / (X_val_new['Num_of_Purchases'] + 1)
        X_test_new['Spend_per_Purchase'] = X_test_new['Total_Spend'] / (X_test_new['Num_of_Purchases'] + 1)
        print("   Feature created")
        
        # Feature 2: Return Rate
        print("\n2. Creating 'Return_Rate' feature...")
        print("   Formula: Num_of_Returns / Num_of_Purchases")
        X_train_new['Return_Rate'] = X_train_new['Num_of_Returns'] / (X_train_new['Num_of_Purchases'] + 1)
        X_val_new['Return_Rate'] = X_val_new['Num_of_Returns'] / (X_val_new['Num_of_Purchases'] + 1)
        X_test_new['Return_Rate'] = X_test_new['Num_of_Returns'] / (X_test_new['Num_of_Purchases'] + 1)
        print("   Feature created")
        
        # Feature 3: Support Contact Rate
        print("\n3. Creating 'Support_per_Year' feature...")
        print("   Formula: Num_of_Support_Contacts / Years_as_Customer")
        X_train_new['Support_per_Year'] = X_train_new['Num_of_Support_Contacts'] / (X_train_new['Years_as_Customer'] + 1)
        X_val_new['Support_per_Year'] = X_val_new['Num_of_Support_Contacts'] / (X_val_new['Years_as_Customer'] + 1)
        X_test_new['Support_per_Year'] = X_test_new['Num_of_Support_Contacts'] / (X_test_new['Years_as_Customer'] + 1)
        print("   Feature created")
        
        # Feature 4: Purchase Frequency
        print("\n4. Creating 'Purchase_Frequency' feature...")
        print("   Formula: Num_of_Purchases / Years_as_Customer")
        X_train_new['Purchase_Frequency'] = X_train_new['Num_of_Purchases'] / (X_train_new['Years_as_Customer'] + 1)
        X_val_new['Purchase_Frequency'] = X_val_new['Num_of_Purchases'] / (X_val_new['Years_as_Customer'] + 1)
        X_test_new['Purchase_Frequency'] = X_test_new['Num_of_Purchases'] / (X_test_new['Years_as_Customer'] + 1)
        print("   Feature created")
        
        # Feature 5: Income to Spend Ratio
        print("\n5. Creating 'Income_to_Spend_Ratio' feature...")
        print("   Formula: Total_Spend / Annual_Income")
        X_train_new['Income_to_Spend_Ratio'] = X_train_new['Total_Spend'] / (X_train_new['Annual_Income'] + 1)
        X_val_new['Income_to_Spend_Ratio'] = X_val_new['Total_Spend'] / (X_val_new['Annual_Income'] + 1)
        X_test_new['Income_to_Spend_Ratio'] = X_test_new['Total_Spend'] / (X_test_new['Annual_Income'] + 1)
        print("   Feature created")
        
        # Feature 6: Recency Score (inverse of days since last purchase)
        print("\n6. Creating 'Recency_Score' feature...")
        print("   Formula: 1 / (Last_Purchase_Days_Ago + 1)")
        X_train_new['Recency_Score'] = 1 / (X_train_new['Last_Purchase_Days_Ago'] + 1)
        X_val_new['Recency_Score'] = 1 / (X_val_new['Last_Purchase_Days_Ago'] + 1)
        X_test_new['Recency_Score'] = 1 / (X_test_new['Last_Purchase_Days_Ago'] + 1)
        print("   Feature created")
        
        print(f"\nFeature engineering completed!")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  New features added: 6")
        print(f"  Total features: {X_train_new.shape[1]}")
        
        return X_train_new, X_val_new, X_test_new
    
    def feature_importance_analysis(self, X_train, y_train, method='f_classif', k=20):
        """
        Perform feature importance analysis
        
        Args:
            X_train: Training features
            y_train: Training target
            method: 'f_classif' or 'mutual_info'
            k: Number of top features to select
        """
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)
        
        if method == 'f_classif':
            print("\nUsing ANOVA F-statistic for feature selection...")
            selector = SelectKBest(score_func=f_classif, k='all')
        else:
            print("\nUsing Mutual Information for feature selection...")
            selector = SelectKBest(score_func=mutual_info_classif, k='all')
        
        selector.fit(X_train, y_train)
        
        # Get feature scores
        feature_scores = pd.DataFrame({
            'Feature': X_train.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        print(f"\nTop {min(k, len(feature_scores))} Features by Importance:")
        print(feature_scores.head(k))
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_scores.head(k)
        plt.barh(range(len(top_features)), top_features['Score'], color='skyblue')
        plt.yticks(range(len(top_features)), top_features['Feature'])
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title(f'Top {k} Feature Importance ({method})', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/09_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nFeature importance visualization saved: {self.output_dir}/09_feature_importance.png")
        plt.close()
        
        return feature_scores
    
    def select_top_features(self, X_train, X_val, X_test, feature_scores, k=20):
        """
        Select top k features
        """
        print("\n" + "="*50)
        print("FEATURE SELECTION")
        print("="*50)
        
        top_features = feature_scores.head(k)['Feature'].tolist()
        
        print(f"\nSelecting top {k} features:")
        print(top_features)
        
        X_train_selected = X_train[top_features]
        X_val_selected = X_val[top_features]
        X_test_selected = X_test[top_features]
        
        print(f"\nFeature selection completed!")
        print(f"  Original features: {X_train.shape[1]}")
        print(f"  Selected features: {X_train_selected.shape[1]}")
        print(f"  Reduction: {((X_train.shape[1] - X_train_selected.shape[1]) / X_train.shape[1] * 100):.1f}%")
        
        return X_train_selected, X_val_selected, X_test_selected, top_features
    
    def run_feature_engineering(self, X_train, X_val, X_test, y_train, apply_selection=False, k=20):
        """
        Run complete feature engineering pipeline
        
        Args:
            apply_selection: Whether to apply feature selection
            k: Number of features to select if apply_selection=True
        """
        print("\n" + "#"*50)
        print("STARTING FEATURE ENGINEERING")
        print("#"*50 + "\n")
        
        # Create interaction features
        X_train_eng, X_val_eng, X_test_eng = self.create_interaction_features(
            X_train, X_val, X_test
        )
        
        # Analyze feature importance
        feature_scores = self.feature_importance_analysis(X_train_eng, y_train, method='f_classif', k=k)
        
        # Optional: Select top features
        if apply_selection:
            X_train_final, X_val_final, X_test_final, selected_features = self.select_top_features(
                X_train_eng, X_val_eng, X_test_eng, feature_scores, k=k
            )
        else:
            print("\nSkipping feature selection - using all engineered features")
            X_train_final = X_train_eng
            X_val_final = X_val_eng
            X_test_final = X_test_eng
            selected_features = X_train_eng.columns.tolist()
        
        print("\n" + "#"*50)
        print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
        print("#"*50 + "\n")
        
        return X_train_final, X_val_final, X_test_final, selected_features


if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_val = pd.read_csv('../data/processed/X_val.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')['Target_Churn']
    
    # Run feature engineering
    engineer = FeatureEngineer(output_dir='../visualizations')
    X_train_eng, X_val_eng, X_test_eng, features = engineer.run_feature_engineering(
        X_train, X_val, X_test, y_train, apply_selection=False
    )
    
    print("\nFeature engineering module executed successfully!")
    print(f"Final shapes:")
    print(f"  X_train: {X_train_eng.shape}")
    print(f"  X_val: {X_val_eng.shape}")
    print(f"  X_test: {X_test_eng.shape}")
