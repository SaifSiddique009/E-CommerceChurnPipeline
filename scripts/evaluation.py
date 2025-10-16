"""
Model Evaluation Module
Evaluates and compares model performance
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from tensorflow import keras
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, models_dir='../models', output_dir='../visualizations'):
        """
        Initialize Evaluator class
        
        Args:
            models_dir: Directory containing trained models
            output_dir: Directory to save evaluation visualizations
        """
        self.models_dir = models_dir
        self.output_dir = output_dir
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_models(self):
        """Load all trained models"""
        print("\n" + "="*50)
        print("LOADING TRAINED MODELS")
        print("="*50)
        
        model_files = {
            'Logistic Regression': 'logistic_regression.pkl',
            'Random Forest': 'random_forest.pkl',
            'XGBoost': 'xgboost.pkl',
            'LightGBM': 'lightgbm.pkl'
        }
        
        for name, filename in model_files.items():
            filepath = f'{self.models_dir}/{filename}'
            if Path(filepath).exists():
                self.models[name] = joblib.load(filepath)
                print(f"Loaded: {name}")
            else:
                print(f"Not found: {name}")
        
        # Load ANN separately
        ann_path = f'{self.models_dir}/ann_model.keras'
        if Path(ann_path).exists():
            self.models['ANN'] = keras.models.load_model(ann_path)
            print(f"Loaded: ANN")
        else:
            print(f"Not found: ANN")
        
        print(f"\nTotal models loaded: {len(self.models)}")
        return self.models
    
    def generate_predictions(self, X_test):
        """Generate predictions for all models"""
        print("\n" + "="*50)
        print("GENERATING PREDICTIONS")
        print("="*50)
        
        for name, model in self.models.items():
            if name == 'ANN':
                # Neural network predictions
                y_pred_proba = model.predict(X_test, verbose=0)
                self.predictions[name] = {
                    'y_pred': (y_pred_proba > 0.5).astype(int).flatten(),
                    'y_pred_proba': y_pred_proba.flatten()
                }
            else:
                # Sklearn models
                self.predictions[name] = {
                    'y_pred': model.predict(X_test),
                    'y_pred_proba': model.predict_proba(X_test)[:, 1]
                }
            print(f"Generated predictions for: {name}")
        
        return self.predictions
    
    def calculate_metrics(self, y_test):
        """Calculate evaluation metrics for all models"""
        print("\n" + "="*50)
        print("CALCULATING EVALUATION METRICS")
        print("="*50)
        
        for name in self.predictions.keys():
            y_pred = self.predictions[name]['y_pred']
            y_pred_proba = self.predictions[name]['y_pred_proba']
            
            self.metrics[name] = {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred, zero_division=0),
                'Recall': recall_score(y_test, y_pred, zero_division=0),
                'F1-Score': f1_score(y_test, y_pred, zero_division=0),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
            }
            
            print(f"\n{name}:")
            for metric, value in self.metrics[name].items():
                print(f"  {metric}: {value:.4f}")
        
        return self.metrics
    
    def create_metrics_comparison(self):
        """Create comparison table and visualization"""
        print("\n" + "="*50)
        print("CREATING METRICS COMPARISON")
        print("="*50)
        
        # Create DataFrame
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df = metrics_df.sort_values('Accuracy', ascending=False)
        
        print("\nModel Performance Comparison:")
        print(metrics_df)
        
        # Save to CSV
        metrics_df.to_csv(f'{self.output_dir}/../reports/model_comparison.csv')
        print(f"\nComparison saved: {self.output_dir}/../reports/model_comparison.csv")
        
        # Visualize comparison
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Bar plot
        metrics_df.plot(kind='bar', ax=axes[0], width=0.8)
        axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Models', fontsize=12)
        axes[0].set_ylabel('Score', fontsize=12)
        axes[0].legend(loc='lower right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        
        # Heatmap
        sns.heatmap(metrics_df, annot=True, fmt='.3f', cmap='YlGnBu', ax=axes[1], 
                    cbar_kws={'label': 'Score'})
        axes[1].set_title('Model Metrics Heatmap', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Metrics', fontsize=12)
        axes[1].set_ylabel('Models', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/10_model_comparison.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {self.output_dir}/10_model_comparison.png")
        plt.close()
        
        return metrics_df
    
    def plot_confusion_matrices(self, y_test):
        """Plot confusion matrices for all models"""
        print("\n" + "="*50)
        print("GENERATING CONFUSION MATRICES")
        print("="*50)
        
        n_models = len(self.predictions)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (name, preds) in enumerate(self.predictions.items()):
            cm = confusion_matrix(y_test, preds['y_pred'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Predicted', fontsize=11)
            axes[idx].set_ylabel('Actual', fontsize=11)
            
            # Add accuracy to title
            acc = self.metrics[name]['Accuracy']
            axes[idx].text(0.5, -0.15, f'Accuracy: {acc:.3f}', 
                          transform=axes[idx].transAxes, ha='center', fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_models, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/11_confusion_matrices.png', dpi=300, bbox_inches='tight')
        print(f"Confusion matrices saved: {self.output_dir}/11_confusion_matrices.png")
        plt.close()
        
        return None
    
    def plot_roc_curves(self, y_test):
        """Plot ROC curves for all models"""
        print("\n" + "="*50)
        print("GENERATING ROC CURVES")
        print("="*50)
        
        plt.figure(figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
        
        for idx, (name, preds) in enumerate(self.predictions.items()):
            fpr, tpr, _ = roc_curve(y_test, preds['y_pred_proba'])
            auc = self.metrics[name]['ROC-AUC']
            
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})',
                    linewidth=2, color=colors[idx % len(colors)])
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/12_roc_curves.png', dpi=300, bbox_inches='tight')
        print(f"ROC curves saved: {self.output_dir}/12_roc_curves.png")
        plt.close()
        
        return None
    
    def generate_classification_reports(self, y_test):
        """Generate detailed classification reports"""
        print("\n" + "="*50)
        print("GENERATING CLASSIFICATION REPORTS")
        print("="*50)
        
        reports_dir = f'{self.output_dir}/../reports'
        Path(reports_dir).mkdir(parents=True, exist_ok=True)
        
        for name, preds in self.predictions.items():
            report = classification_report(y_test, preds['y_pred'],
                                          target_names=['Not Churned', 'Churned'],
                                          digits=4)
            
            print(f"\n{name} Classification Report:")
            print(report)
            
            # Save report
            with open(f'{reports_dir}/{name.replace(" ", "_").lower()}_report.txt', 'w') as f:
                f.write(f"{name} Classification Report\n")
                f.write("="*60 + "\n\n")
                f.write(report)
            
        print(f"\nClassification reports saved in: {reports_dir}/")
        
        return None
    
    def identify_best_model(self):
        """Identify the best performing model"""
        print("\n" + "="*50)
        print("IDENTIFYING BEST MODEL")
        print("="*50)
        
        # Best by accuracy
        best_acc = max(self.metrics.items(), key=lambda x: x[1]['Accuracy'])
        print(f"\nBest Model by Accuracy: {best_acc[0]}")
        print(f"   Accuracy: {best_acc[1]['Accuracy']:.4f}")
        
        # Best by F1-Score
        best_f1 = max(self.metrics.items(), key=lambda x: x[1]['F1-Score'])
        print(f"\nBest Model by F1-Score: {best_f1[0]}")
        print(f"   F1-Score: {best_f1[1]['F1-Score']:.4f}")
        
        # Best by ROC-AUC
        best_auc = max(self.metrics.items(), key=lambda x: x[1]['ROC-AUC'])
        print(f"\nBest Model by ROC-AUC: {best_auc[0]}")
        print(f"   ROC-AUC: {best_auc[1]['ROC-AUC']:.4f}")
        
        return best_acc[0], best_f1[0], best_auc[0]
    
    def run_full_evaluation(self, X_test, y_test):
        """Run complete evaluation pipeline"""
        print("\n" + "#"*50)
        print("STARTING MODEL EVALUATION")
        print("#"*50 + "\n")
        
        # Load models
        self.load_models()
        
        # Generate predictions
        self.generate_predictions(X_test)
        
        # Calculate metrics
        self.calculate_metrics(y_test)
        
        # Create visualizations
        self.create_metrics_comparison()
        self.plot_confusion_matrices(y_test)
        self.plot_roc_curves(y_test)
        
        # Generate reports
        self.generate_classification_reports(y_test)
        
        # Identify best model
        self.identify_best_model()
        
        print("\n" + "#"*50)
        print("MODEL EVALUATION COMPLETED SUCCESSFULLY!")
        print("#"*50 + "\n")
        
        return self.metrics


if __name__ == "__main__":
    # Load test data
    print("Loading test data...")
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_test = pd.read_csv('../data/processed/y_test.csv')['Target_Churn']
    
    # Run evaluation
    evaluator = ModelEvaluator(models_dir='../models', output_dir='../visualizations')
    metrics = evaluator.run_full_evaluation(X_test, y_test)
    
    print("\nEvaluation module executed successfully!")
