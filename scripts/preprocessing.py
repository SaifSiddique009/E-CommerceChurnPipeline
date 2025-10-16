"""
Data Preprocessing Module
Handles data cleaning, encoding, scaling, and splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
from pathlib import Path

class ChurnPreprocessor:
    def __init__(self, data_path, output_dir='../models'):
        """
        Initialize Preprocessor class
        
        Args:
            data_path: Path to the dataset CSV file
            output_dir: Directory to save preprocessing artifacts
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        self.scaler = StandardScaler()
        self.encoded_columns = []
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset for preprocessing...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded! Shape: {self.df.shape}")
        return self.df
    
    def drop_unnecessary_columns(self):
        """Drop columns that are not useful for modeling"""
        print("\n" + "="*50)
        print("DROPPING UNNECESSARY COLUMNS")
        print("="*50)
        
        columns_to_drop = ['Customer_ID']
        self.df = self.df.drop(columns=columns_to_drop, errors='ignore')
        print(f"Dropped columns: {columns_to_drop}")
        print(f"Remaining columns: {self.df.shape[1]}")
        
        return self.df
    
    def encode_categorical_features(self):
        """
        Encode categorical features
        - One-Hot Encoding for Gender and Promotion_Response
        - Binary encoding for Email_Opt_In and Target_Churn
        """
        print("\n" + "="*50)
        print("ENCODING CATEGORICAL FEATURES")
        print("="*50)
        
        # One-Hot Encoding for Gender
        print("\nApplying One-Hot Encoding for 'Gender'...")
        print("Reasoning: Gender has 3 categories (Male, Female, Other) with no ordinal relationship")
        gender_dummies = pd.get_dummies(self.df['Gender'], prefix='Gender', drop_first=False)
        self.df = pd.concat([self.df, gender_dummies], axis=1)
        self.df = self.df.drop('Gender', axis=1)
        self.encoded_columns.extend(gender_dummies.columns.tolist())
        print(f"Created columns: {gender_dummies.columns.tolist()}")
        
        # One-Hot Encoding for Promotion_Response
        print("\nApplying One-Hot Encoding for 'Promotion_Response'...")
        print("Reasoning: Promotion_Response has 3 categories (Responded, Ignored, Unsubscribed) with no ordinal relationship")
        promo_dummies = pd.get_dummies(self.df['Promotion_Response'], prefix='Promo', drop_first=False)
        self.df = pd.concat([self.df, promo_dummies], axis=1)
        self.df = self.df.drop('Promotion_Response', axis=1)
        self.encoded_columns.extend(promo_dummies.columns.tolist())
        print(f"Created columns: {promo_dummies.columns.tolist()}")
        
        # Binary encoding for Email_Opt_In
        print("\nConverting 'Email_Opt_In' to binary (0/1)...")
        self.df['Email_Opt_In'] = self.df['Email_Opt_In'].astype(int)
        print(f"Email_Opt_In converted to: {self.df['Email_Opt_In'].unique()}")
        
        # Binary encoding for Target_Churn
        print("\nConverting 'Target_Churn' to binary (0/1)...")
        self.df['Target_Churn'] = self.df['Target_Churn'].astype(int)
        print(f"Target_Churn converted to: {self.df['Target_Churn'].unique()}")
        
        print(f"\nEncoding completed! New shape: {self.df.shape}")
        
        return self.df
    
    def split_features_target(self):
        """Split features and target variable"""
        print("\n" + "="*50)
        print("SPLITTING FEATURES AND TARGET")
        print("="*50)
        
        X = self.df.drop('Target_Churn', axis=1)
        y = self.df['Target_Churn']
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nFeature columns: {X.columns.tolist()}")
        
        return X, y
    
    def scale_numerical_features(self, X_train, X_val, X_test):
        """Scale numerical features using StandardScaler"""
        print("\n" + "="*50)
        print("SCALING NUMERICAL FEATURES")
        print("="*50)
        
        # Identify numerical columns (exclude one-hot encoded columns)
        numerical_cols = [col for col in X_train.columns if col not in self.encoded_columns]
        
        print(f"\nNumerical columns to scale: {numerical_cols}")
        print(f"Total: {len(numerical_cols)} columns")
        
        # Fit scaler on training data only
        X_train_scaled = X_train.copy()
        X_val_scaled = X_val.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_val_scaled[numerical_cols] = self.scaler.transform(X_val[numerical_cols])
        X_test_scaled[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print("Scaling completed using StandardScaler")
        print("Scaler fitted on training data and applied to validation and test sets")
        
        # Save scaler
        scaler_path = f'{self.output_dir}/scaler.pkl'
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler saved: {scaler_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def train_val_test_split(self, X, y, test_size=0.15, val_size=0.15, random_state=42):
        """Split data into train, validation, and test sets"""
        print("\n" + "="*50)
        print("TRAIN-VALIDATION-TEST SPLIT")
        print("="*50)
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state, stratify=y_temp
        )
        
        print(f"\nData split completed:")
        print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
        print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
        print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")
        
        # Check class distribution
        print("\nClass distribution in each set:")
        print(f"  Training:   Churn={y_train.sum()}, No Churn={len(y_train)-y_train.sum()}")
        print(f"  Validation: Churn={y_val.sum()}, No Churn={len(y_val)-y_val.sum()}")
        print(f"  Test:       Churn={y_test.sum()}, No Churn={len(y_test)-y_test.sum()}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """Save processed data"""
        print("\n" + "="*50)
        print("SAVING PROCESSED DATA")
        print("="*50)
        
        # Create data directory
        processed_dir = '../data/processed'
        Path(processed_dir).mkdir(parents=True, exist_ok=True)
        
        # Save as CSV
        pd.DataFrame(X_train).to_csv(f'{processed_dir}/X_train.csv', index=False)
        pd.DataFrame(X_val).to_csv(f'{processed_dir}/X_val.csv', index=False)
        pd.DataFrame(X_test).to_csv(f'{processed_dir}/X_test.csv', index=False)
        pd.DataFrame(y_train, columns=['Target_Churn']).to_csv(f'{processed_dir}/y_train.csv', index=False)
        pd.DataFrame(y_val, columns=['Target_Churn']).to_csv(f'{processed_dir}/y_val.csv', index=False)
        pd.DataFrame(y_test, columns=['Target_Churn']).to_csv(f'{processed_dir}/y_test.csv', index=False)
        
        print(f"Processed data saved in: {processed_dir}/")
        
        return processed_dir
    
    def run_full_preprocessing(self):
        """Run complete preprocessing pipeline"""
        print("\n" + "#"*50)
        print("STARTING DATA PREPROCESSING")
        print("#"*50 + "\n")
        
        # Load and clean data
        self.load_data()
        self.drop_unnecessary_columns()
        
        # Encode features
        self.encode_categorical_features()
        
        # Split features and target
        X, y = self.split_features_target()
        
        # Split into train, val, test
        X_train, X_val, X_test, y_train, y_val, y_test = self.train_val_test_split(X, y)
        
        # Scale features
        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_numerical_features(
            X_train, X_val, X_test
        )
        
        # Save processed data
        self.save_processed_data(
            X_train_scaled, X_val_scaled, X_test_scaled,
            y_train, y_val, y_test
        )
        
        print("\n" + "#"*50)
        print("PREPROCESSING COMPLETED SUCCESSFULLY!")
        print("#"*50 + "\n")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test


if __name__ == "__main__":
    # Set paths
    data_path = '../data/dataset.csv'
    output_dir = '../models'
    
    # Run preprocessing
    preprocessor = ChurnPreprocessor(data_path, output_dir)
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_full_preprocessing()
    
    print("\nPreprocessing module executed successfully!")
    print(f"Final shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  X_test: {X_test.shape}")
