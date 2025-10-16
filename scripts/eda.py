"""
Exploratory Data Analysis (EDA) Module
Performs comprehensive data exploration and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style for better visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class ChurnEDA:
    def __init__(self, data_path, output_dir='../visualizations'):
        """
        Initialize EDA class
        
        Args:
            data_path: Path to the dataset CSV file
            output_dir: Directory to save visualization outputs
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.df = None
        
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the dataset"""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def basic_info(self):
        """Display basic information about the dataset"""
        print("\n" + "="*50)
        print("BASIC DATASET INFORMATION")
        print("="*50)
        
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Rows: {self.df.shape[0]}")
        print(f"Number of Columns: {self.df.shape[1]}")
        
        print("\nColumn Names and Data Types:")
        print(self.df.dtypes)
        
        print("\nFirst 5 Rows:")
        print(self.df.head())
        
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nStatistical Summary:")
        print(self.df.describe())
        
        return self.df.info()
    
    def check_missing_values(self):
        """Check for missing values"""
        print("\n" + "="*50)
        print("MISSING VALUES ANALYSIS")
        print("="*50)
        
        missing = self.df.isnull().sum()
        missing_percent = (missing / len(self.df)) * 100
        
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing_Count': missing.values,
            'Percentage': missing_percent.values
        })
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        
        if len(missing_df) == 0:
            print("No missing values found in the dataset!")
        else:
            print("Missing Values Summary:")
            print(missing_df)
        
        return missing_df
    
    def check_duplicates(self):
        """Check for duplicate rows"""
        print("\n" + "="*50)
        print("DUPLICATE ROWS ANALYSIS")
        print("="*50)
        
        duplicates = self.df.duplicated().sum()
        print(f"Number of duplicate rows: {duplicates}")
        
        if duplicates > 0:
            print("\nDuplicate rows:")
            print(self.df[self.df.duplicated()])
        else:
            print("No duplicate rows found!")
        
        return duplicates
    
    def analyze_target_variable(self):
        """Analyze the target variable distribution"""
        print("\n" + "="*50)
        print("TARGET VARIABLE ANALYSIS")
        print("="*50)
        
        target_counts = self.df['Target_Churn'].value_counts()
        target_percent = self.df['Target_Churn'].value_counts(normalize=True) * 100
        
        print("\nTarget Variable Distribution:")
        print(f"Churned (True): {target_counts.get(True, 0)} ({target_percent.get(True, 0):.2f}%)")
        print(f"Not Churned (False): {target_counts.get(False, 0)} ({target_percent.get(False, 0):.2f}%)")
        
        # Visualize target distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        sns.countplot(data=self.df, x='Target_Churn', ax=axes[0], palette='Set2')
        axes[0].set_title('Target Variable Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Churn Status', fontsize=12)
        axes[0].set_ylabel('Count', fontsize=12)
        
        # Add value labels on bars
        for container in axes[0].containers:
            axes[0].bar_label(container, fmt='%d')
        
        # Pie chart
        colors = sns.color_palette('Set2', 2)
        axes[1].pie(target_counts.values, labels=['Not Churned', 'Churned'], autopct='%1.1f%%',
                   startangle=90, colors=colors, textprops={'fontsize': 12})
        axes[1].set_title('Target Variable Distribution (%)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/01_target_distribution.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {self.output_dir}/01_target_distribution.png")
        plt.close()
        
        return target_counts
    
    def analyze_numerical_features(self):
        """Analyze numerical features"""
        print("\n" + "="*50)
        print("NUMERICAL FEATURES ANALYSIS")
        print("="*50)
        
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['Customer_ID', 'Target_Churn']]
        
        print(f"\nNumerical Features: {numerical_cols}")
        print(f"Total: {len(numerical_cols)} features")
        
        # Distribution plots
        n_cols = 3
        n_rows = (len(numerical_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            sns.histplot(data=self.df, x=col, kde=True, ax=axes[idx], color='steelblue')
            axes[idx].set_title(f'Distribution of {col}', fontsize=11, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=10)
            axes[idx].set_ylabel('Frequency', fontsize=10)
        
        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/02_numerical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {self.output_dir}/02_numerical_distributions.png")
        plt.close()
        
        # Box plots for outlier detection
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 4))
        axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes
        
        for idx, col in enumerate(numerical_cols):
            sns.boxplot(data=self.df, y=col, ax=axes[idx], color='lightcoral')
            axes[idx].set_title(f'Box Plot of {col}', fontsize=11, fontweight='bold')
            axes[idx].set_ylabel(col, fontsize=10)
        
        # Hide empty subplots
        for idx in range(len(numerical_cols), len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/03_numerical_boxplots.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {self.output_dir}/03_numerical_boxplots.png")
        plt.close()
        
        return numerical_cols
    
    def analyze_categorical_features(self):
        """Analyze categorical features"""
        print("\n" + "="*50)
        print("CATEGORICAL FEATURES ANALYSIS")
        print("="*50)
        
        categorical_cols = ['Gender', 'Promotion_Response', 'Email_Opt_In']
        
        print(f"\nCategorical Features: {categorical_cols}")
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, col in enumerate(categorical_cols):
            value_counts = self.df[col].value_counts()
            print(f"\n{col} Distribution:")
            print(value_counts)
            
            sns.countplot(data=self.df, x=col, ax=axes[idx], palette='viridis')
            axes[idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=11)
            axes[idx].set_ylabel('Count', fontsize=11)
            axes[idx].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for container in axes[idx].containers:
                axes[idx].bar_label(container, fmt='%d')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/04_categorical_distributions.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {self.output_dir}/04_categorical_distributions.png")
        plt.close()
        
        return categorical_cols
    
    def correlation_analysis(self):
        """Perform correlation analysis"""
        print("\n" + "="*50)
        print("CORRELATION ANALYSIS")
        print("="*50)
        
        # Convert Target_Churn to numerical for correlation
        df_corr = self.df.copy()
        df_corr['Target_Churn'] = df_corr['Target_Churn'].astype(int)
        
        # Select numerical columns (including converted Target_Churn)
        numerical_cols = df_corr.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df_corr[numerical_cols].corr()
        
        # Correlation with target
        target_corr = corr_matrix['Target_Churn'].sort_values(ascending=False)
        print("\nCorrelation with Target_Churn:")
        print(target_corr)
        
        # Visualize correlation matrix
        plt.figure(figsize=(14, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/05_correlation_matrix.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {self.output_dir}/05_correlation_matrix.png")
        plt.close()
        
        # Top correlated features with target
        top_features = target_corr[1:11]  # Exclude target itself
        plt.figure(figsize=(12, 6))
        top_features.plot(kind='barh', color='teal')
        plt.title('Top 10 Features Correlated with Churn', fontsize=14, fontweight='bold')
        plt.xlabel('Correlation Coefficient', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/06_top_correlations.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {self.output_dir}/06_top_correlations.png")
        plt.close()
        
        return corr_matrix
    
    def bivariate_analysis(self):
        """Analyze relationships between features and target"""
        print("\n" + "="*50)
        print("BIVARIATE ANALYSIS: FEATURES vs TARGET")
        print("="*50)
        
        # Numerical features vs Target
        numerical_cols = ['Age', 'Annual_Income', 'Total_Spend', 'Years_as_Customer',
                         'Num_of_Purchases', 'Average_Transaction_Amount', 'Num_of_Returns',
                         'Num_of_Support_Contacts', 'Satisfaction_Score', 'Last_Purchase_Days_Ago']
        
        # Select key features for visualization
        key_features = ['Last_Purchase_Days_Ago', 'Satisfaction_Score', 'Num_of_Support_Contacts',
                       'Total_Spend', 'Years_as_Customer', 'Num_of_Returns']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, col in enumerate(key_features):
            sns.boxplot(data=self.df, x='Target_Churn', y=col, ax=axes[idx], palette='Set1')
            axes[idx].set_title(f'{col} vs Churn', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel('Churned', fontsize=11)
            axes[idx].set_ylabel(col, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/07_bivariate_numerical.png', dpi=300, bbox_inches='tight')
        print(f"\nVisualization saved: {self.output_dir}/07_bivariate_numerical.png")
        plt.close()
        
        # Categorical features vs Target
        categorical_cols = ['Gender', 'Promotion_Response', 'Email_Opt_In']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, col in enumerate(categorical_cols):
            ct = pd.crosstab(self.df[col], self.df['Target_Churn'], normalize='index') * 100
            ct.plot(kind='bar', ax=axes[idx], stacked=False, color=['#2ecc71', '#e74c3c'])
            axes[idx].set_title(f'{col} vs Churn Rate', fontsize=12, fontweight='bold')
            axes[idx].set_xlabel(col, fontsize=11)
            axes[idx].set_ylabel('Percentage (%)', fontsize=11)
            axes[idx].legend(['Not Churned', 'Churned'], loc='upper right')
            axes[idx].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/08_bivariate_categorical.png', dpi=300, bbox_inches='tight')
        print(f"Visualization saved: {self.output_dir}/08_bivariate_categorical.png")
        plt.close()
        
        return None
    
    def generate_summary_report(self):
        """Generate a summary report of EDA findings"""
        print("\n" + "="*50)
        print("EDA SUMMARY REPORT")
        print("="*50)
        
        report = []
        report.append("EXPLORATORY DATA ANALYSIS SUMMARY")
        report.append("="*40)
        
        # Dataset info
        report.append(f"\n1. DATASET INFORMATION")
        report.append(f"   - Total Rows: {self.df.shape[0]}")
        report.append(f"   - Total Columns: {self.df.shape[1]}")
        report.append(f"   - Missing Values: {self.df.isnull().sum().sum()}")
        report.append(f"   - Duplicate Rows: {self.df.duplicated().sum()}")
        
        # Target variable
        target_counts = self.df['Target_Churn'].value_counts()
        churn_rate = (target_counts.get(True, 0) / len(self.df)) * 100
        report.append(f"\n2. TARGET VARIABLE")
        report.append(f"   - Churn Rate: {churn_rate:.2f}%")
        report.append(f"   - Churned Customers: {target_counts.get(True, 0)}")
        report.append(f"   - Retained Customers: {target_counts.get(False, 0)}")
        
        # Features
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols if col not in ['Customer_ID', 'Target_Churn']]
        report.append(f"\n3. FEATURES")
        report.append(f"   - Numerical Features: {len(numerical_cols)}")
        report.append(f"   - Categorical Features: 3 (Gender, Promotion_Response, Email_Opt_In)")
        
        # Key insights
        df_corr = self.df.copy()
        df_corr['Target_Churn'] = df_corr['Target_Churn'].astype(int)
        target_corr = df_corr[numerical_cols + ['Target_Churn']].corr()['Target_Churn'].sort_values(ascending=False)
        
        report.append(f"\n4. KEY INSIGHTS")
        report.append(f"   Top 3 Features Positively Correlated with Churn:")
        for i, (feature, corr) in enumerate(list(target_corr[1:4].items()), 1):
            report.append(f"      {i}. {feature}: {corr:.3f}")
        
        report.append(f"\n   Top 3 Features Negatively Correlated with Churn:")
        for i, (feature, corr) in enumerate(list(target_corr[-3:].items()), 1):
            report.append(f"      {i}. {feature}: {corr:.3f}")
        
        report_text = "\n".join(report)
        print(report_text)
        
        # Save report
        with open(f'{self.output_dir}/EDA_SUMMARY_REPORT.txt', 'w') as f:
            f.write(report_text)
        print(f"\nSummary report saved: {self.output_dir}/EDA_SUMMARY_REPORT.txt")
        
        return report_text
    
    def run_full_eda(self):
        """Run complete EDA pipeline"""
        print("\n" + "#"*50)
        print("STARTING EXPLORATORY DATA ANALYSIS")
        print("#"*50 + "\n")
        
        # Load data
        self.load_data()
        
        # Basic information
        self.basic_info()
        
        # Check data quality
        self.check_missing_values()
        self.check_duplicates()
        
        # Analyze variables
        self.analyze_target_variable()
        self.analyze_numerical_features()
        self.analyze_categorical_features()
        
        # Advanced analysis
        self.correlation_analysis()
        self.bivariate_analysis()
        
        # Generate summary
        self.generate_summary_report()
        
        print("\n" + "#"*50)
        print("EDA COMPLETED SUCCESSFULLY!")
        print("#"*50 + "\n")
        
        return self.df


if __name__ == "__main__":
    # Set paths
    data_path = '../data/dataset.csv'
    output_dir = '../visualizations'
    
    # Run EDA
    eda = ChurnEDA(data_path, output_dir)
    df = eda.run_full_eda()
    
    print("\nEDA module executed successfully!")
    print(f"All visualizations saved in: {output_dir}/")
