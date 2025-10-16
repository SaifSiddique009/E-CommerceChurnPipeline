"""
Modeling Module
Trains baseline and advanced models for churn prediction
"""

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from tensorflow import keras
from tensorflow.keras import layers, callbacks
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ChurnModeling:
    def __init__(self, output_dir='../models'):
        """
        Initialize Modeling class
        
        Args:
            output_dir: Directory to save trained models
        """
        self.output_dir = output_dir
        self.models = {}
        self.results = {}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def train_logistic_regression(self, X_train, y_train, X_val, y_val):
        """Train Logistic Regression model"""
        print("\n" + "="*50)
        print("TRAINING LOGISTIC REGRESSION")
        print("="*50)
        
        print("\nTraining Logistic Regression...")
        model = LogisticRegression(max_iter=100, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  Training completed!")
        print(f"  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        self.models['logistic_regression'] = model
        self.results['logistic_regression'] = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        # Save model
        joblib.dump(model, f'{self.output_dir}/logistic_regression.pkl')
        print(f"  Model saved: {self.output_dir}/logistic_regression.pkl")
        
        return model
    
    def train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model"""
        print("\n" + "="*50)
        print("TRAINING RANDOM FOREST")
        print("="*50)
        
        print("\nTraining Random Forest...")
        model = RandomForestClassifier(
            n_estimators=3,
            max_depth=5,
            min_samples_split=3,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  Training completed!")
        print(f"  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        # Save model
        joblib.dump(model, f'{self.output_dir}/random_forest.pkl')
        print(f"  Model saved: {self.output_dir}/random_forest.pkl")
        
        return model
    
    def train_xgboost(self, X_train, y_train, X_val, y_val):
        """Train XGBoost model"""
        print("\n" + "="*50)
        print("TRAINING XGBOOST")
        print("="*50)
        
        print("\nTraining XGBoost...")
        model = XGBClassifier(
            n_estimators=2,
            max_depth=3,
            learning_rate=0.2,
            random_state=42,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  Training completed!")
        print(f"  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        self.models['xgboost'] = model
        self.results['xgboost'] = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        # Save model
        joblib.dump(model, f'{self.output_dir}/xgboost.pkl')
        print(f"  Model saved: {self.output_dir}/xgboost.pkl")
        
        return model
    
    def train_lightgbm(self, X_train, y_train, X_val, y_val):
        """Train LightGBM model"""
        print("\n" + "="*50)
        print("TRAINING LIGHTGBM")
        print("="*50)
        
        print("\nTraining LightGBM...")
        model = LGBMClassifier(
            n_estimators=2,
            max_depth=3,
            learning_rate=0.2,
            random_state=42,
            verbose=-1
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        train_score = model.score(X_train, y_train)
        val_score = model.score(X_val, y_val)
        
        print(f"  Training completed!")
        print(f"  Training Accuracy: {train_score:.4f}")
        print(f"  Validation Accuracy: {val_score:.4f}")
        
        self.models['lightgbm'] = model
        self.results['lightgbm'] = {
            'train_accuracy': train_score,
            'val_accuracy': val_score
        }
        
        # Save model
        joblib.dump(model, f'{self.output_dir}/lightgbm.pkl')
        print(f"  Model saved: {self.output_dir}/lightgbm.pkl")
        
        return model
    
    def train_neural_network(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train Artificial Neural Network using TensorFlow/Keras"""
        print("\n" + "="*50)
        print("TRAINING ARTIFICIAL NEURAL NETWORK (ANN)")
        print("="*50)
        
        print("\nBuilding ANN architecture...")
        
        # Build model
        model = keras.Sequential([
            layers.Input(shape=(X_train.shape[1],)),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        print("\nModel Architecture:")
        model.summary()
        
        print(f"\nTraining ANN for {epochs} epochs...")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=0.00001
        )
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stop, reduce_lr],
            verbose=1
        )
        
        # Evaluate
        train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
        val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
        
        print(f"\n  Training completed!")
        print(f"  Training Accuracy: {train_acc:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f}")
        
        self.models['ann'] = model
        self.results['ann'] = {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
            'history': history.history
        }
        
        # Save model
        model.save(f'{self.output_dir}/ann_model.keras')
        print(f"  Model saved: {self.output_dir}/ann_model.keras")
        
        return model, history
    
    def train_all_models(self, X_train, y_train, X_val, y_val):
        """Train all baseline models"""
        print("\n" + "#"*50)
        print("TRAINING BASELINE MODELS")
        print("#"*50 + "\n")
        
        # Train all models
        self.train_logistic_regression(X_train, y_train, X_val, y_val)
        self.train_random_forest(X_train, y_train, X_val, y_val)
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)
        
        # Summary
        print("\n" + "="*50)
        print("BASELINE MODELS SUMMARY")
        print("="*50)
        
        results_df = pd.DataFrame(self.results).T
        print("\n", results_df)
        
        # Save results
        results_df.to_csv(f'{self.output_dir}/baseline_results.csv')
        print(f"\n  Results saved: {self.output_dir}/baseline_results.csv")
        
        print("\n" + "#"*50)
        print("BASELINE MODELS TRAINING COMPLETED!")
        print("#"*50 + "\n")
        
        return self.models, self.results


if __name__ == "__main__":
    # Load processed data
    print("Loading processed data...")
    X_train = pd.read_csv('../data/processed/X_train.csv')
    X_val = pd.read_csv('../data/processed/X_val.csv')
    X_test = pd.read_csv('../data/processed/X_test.csv')
    y_train = pd.read_csv('../data/processed/y_train.csv')['Target_Churn']
    y_val = pd.read_csv('../data/processed/y_val.csv')['Target_Churn']
    y_test = pd.read_csv('../data/processed/y_test.csv')['Target_Churn']
    
    # Train models
    modeling = ChurnModeling(output_dir='../models')
    models, results = modeling.train_all_models(X_train, y_train, X_val, y_val)
    
    # Train ANN
    print("\nTraining Advanced Model (ANN)...")
    modeling.train_neural_network(X_train, y_train, X_val, y_val)
    
    print("\nModeling module executed successfully!")
