"""
Main Execution Script
Runs the complete ML pipeline for customer churn prediction
"""

import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from eda import ChurnEDA
from preprocessing import ChurnPreprocessor
from feature_engineering import FeatureEngineer
from modeling import ChurnModeling
from evaluation import ModelEvaluator

def main():
    """
    Execute complete ML pipeline
    """
    print("\n" + "#"*50)
    print("CUSTOMER CHURN PREDICTION ML PIPELINE")
    print("#"*50 + "\n")
    
    # Step 1: EDA
    print("\nRunning Exploratory Data Analysis...")
    eda = ChurnEDA(data_path='../data/dataset.csv', output_dir='../visualizations')
    df = eda.run_full_eda()
    
    # Step 2: Preprocessing
    print("\nRunning Data Preprocessing...")
    preprocessor = ChurnPreprocessor(data_path='../data/dataset.csv', output_dir='../models')
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.run_full_preprocessing()
    
    # Step 3: Feature Engineering
    print("\nRunning Feature Engineering...")
    engineer = FeatureEngineer(output_dir='../visualizations')
    X_train_eng, X_val_eng, X_test_eng, features = engineer.run_feature_engineering(
        X_train, X_val, X_test, y_train, apply_selection=False
    )
    
    # Step 4: Model Training
    print("\nTraining Machine Learning Models...")
    modeling = ChurnModeling(output_dir='../models')
    
    # Train baseline models
    models, results = modeling.train_all_models(
        X_train_eng, y_train, X_val_eng, y_val
    )
    
    # Train ANN
    print("\nTraining Advanced Model (ANN)...")
    ann_model, history = modeling.train_neural_network(
        X_train_eng, y_train, X_val_eng, y_val, epochs=50
    )
    
    # Save engineered test data for evaluation
    
    pd.DataFrame(X_test_eng).to_csv('../data/processed/X_test.csv', index=False)
    print("\nUpdated test data with engineered features")
    
    # Step 5: Model Evaluation
    print("\nEvaluating Models...")
    evaluator = ModelEvaluator(models_dir='../models', output_dir='../visualizations')
    metrics = evaluator.run_full_evaluation(X_test_eng, y_test)
    
    # Check if accuracy target is met
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    best_accuracy = max([m['Accuracy'] for m in metrics.values()])
    print(f"\nBest Model Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    
    print("\n" + "#"*50)
    print("PIPELINE EXECUTION COMPLETED!")
    print("#"*50 + "\n")
    
    print("\nAll visualizations saved in: ../visualizations/")
    print("All models saved in: ../models/")
    print("All reports saved in: ../reports/\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
