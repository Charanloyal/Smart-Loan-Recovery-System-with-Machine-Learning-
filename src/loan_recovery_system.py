"""
Smart Loan Recovery System with Machine Learning
Main analysis script that implements the complete loan recovery system
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
from datetime import datetime

# Machine Learning imports
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Local imports
from visualizations import (plot_recovery_status_distribution, plot_income_vs_loan_recovery,
                          plot_correlation_heatmap, plot_borrower_segments_interactive,
                          plot_risk_score_distribution, plot_feature_importance,
                          plot_model_performance, create_dashboard_plots)

warnings.filterwarnings('ignore')

class SmartLoanRecoverySystem:
    """
    Smart Loan Recovery System using Machine Learning
    """
    
    def __init__(self, data_path=None):
        """
        Initialize the Loan Recovery System
        
        Args:
            data_path (str): Path to the loan recovery dataset
        """
        self.df = None
        self.scaler = StandardScaler()
        self.kmeans_model = None
        self.rf_model = None
        self.feature_names = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """Load the loan recovery dataset"""
        try:
            self.df = pd.read_csv(data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
        except FileNotFoundError:
            print(f"File not found: {data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        if self.df is None:
            print("Please load data first.")
            return
        
        print("=== DATASET OVERVIEW ===")
        print(f"Dataset shape: {self.df.shape}")
        print(f"\\nDataset info:")
        print(self.df.info())
        
        print(f"\\n=== MISSING VALUES ===")
        missing_values = self.df.isnull().sum()
        print(missing_values[missing_values > 0])
        
        print(f"\\n=== RECOVERY STATUS DISTRIBUTION ===")
        print(self.df['Recovery_Status'].value_counts())
        print(f"\\nRecovery Status Percentages:")
        print(self.df['Recovery_Status'].value_counts(normalize=True) * 100)
        
        print(f"\\n=== BASIC STATISTICS ===")
        print(self.df.describe())
        
        # Create visualizations
        print("\\nCreating visualizations...")
        plot_recovery_status_distribution(self.df)
        plot_income_vs_loan_recovery(self.df)
        plot_correlation_heatmap(self.df)
    
    def perform_clustering(self, n_clusters=4):
        """
        Perform K-Means clustering to segment borrowers
        
        Args:
            n_clusters (int): Number of clusters
        """
        if self.df is None:
            print("Please load data first.")
            return
        
        print(f"\\n=== BORROWER SEGMENTATION (K-MEANS CLUSTERING) ===")
        
        # Select features for clustering
        features = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure', 
                   'Interest_Rate', 'Collateral_Value', 'Outstanding_Loan_Amount',
                   'Monthly_EMI', 'Num_Missed_Payments', 'Days_Past_Due']
        
        # Scale features
        X = self.df[features]
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform K-means clustering
        self.kmeans_model = KMeans(n_clusters=n_clusters, random_state=42)
        self.df['Borrower_Segment'] = self.kmeans_model.fit_predict(X_scaled)
        
        print(f"Created {n_clusters} borrower segments")
        print("\\nSegment distribution:")
        print(self.df['Borrower_Segment'].value_counts().sort_index())
        
        # Analyze segments
        print("\\n=== SEGMENT ANALYSIS ===")
        segment_analysis = self.df.groupby('Borrower_Segment').agg({
            'Monthly_Income': 'mean',
            'Loan_Amount': 'mean',
            'Outstanding_Loan_Amount': 'mean',
            'Num_Missed_Payments': 'mean',
            'Days_Past_Due': 'mean',
            'Recovery_Status': lambda x: x.value_counts().to_dict()
        }).round(2)
        
        print(segment_analysis)
        
        # Name the segments based on their characteristics
        self.df['Segment_Name'] = self.df['Borrower_Segment'].map({
            0: 'Moderate Income, High Loan Burden',
            1: 'High Income, Low Default Risk',
            2: 'Moderate Income, Medium Risk',
            3: 'High Loan, Higher Default Risk'
        })
        
        print("\\nSegment names assigned:")
        for i in range(n_clusters):
            segment_name = self.df[self.df['Borrower_Segment'] == i]['Segment_Name'].iloc[0]
            count = len(self.df[self.df['Borrower_Segment'] == i])
            print(f"Segment {i}: {segment_name} ({count} borrowers)")
        
        # Visualize segments
        plot_borrower_segments_interactive(self.df)
    
    def build_risk_prediction_model(self):
        """
        Build a machine learning model to predict high-risk borrowers
        """
        if self.df is None or 'Borrower_Segment' not in self.df.columns:
            print("Please load data and perform clustering first.")
            return
        
        print(f"\\n=== BUILDING RISK PREDICTION MODEL ===")
        
        # Create high risk flag
        self.df['High_Risk_Flag'] = self.df['Segment_Name'].apply(
            lambda x: 1 if x in ['High Loan, Higher Default Risk', 
                               'Moderate Income, High Loan Burden'] else 0
        )
        
        print(f"High-risk borrowers: {self.df['High_Risk_Flag'].sum()}")
        print(f"Low-risk borrowers: {len(self.df) - self.df['High_Risk_Flag'].sum()}")
        
        # Select features for the model
        self.feature_names = ['Age', 'Monthly_Income', 'Loan_Amount', 'Loan_Tenure',
                             'Interest_Rate', 'Collateral_Value', 'Outstanding_Loan_Amount',
                             'Monthly_EMI', 'Num_Missed_Payments', 'Days_Past_Due']
        
        X = self.df[self.feature_names]
        y = self.df['High_Risk_Flag']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train the model
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.rf_model.predict(X_test)
        y_prob = self.rf_model.predict_proba(X_test)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\\nModel Accuracy: {accuracy:.3f}")
        
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Add risk scores to the entire dataset
        self.df['Risk_Score'] = self.rf_model.predict_proba(X)[:, 1]
        
        print(f"\\nRisk Score Distribution:")
        print(f"Mean Risk Score: {self.df['Risk_Score'].mean():.3f}")
        print(f"Std Risk Score: {self.df['Risk_Score'].std():.3f}")
        
        # Visualizations
        plot_feature_importance(self.rf_model, self.feature_names)
        plot_model_performance(y_test, y_pred, y_prob)
        plot_risk_score_distribution(self.df)
        
        return self.rf_model
    
    def assign_recovery_strategies(self):
        """
        Assign recovery strategies based on risk scores
        """
        if self.df is None or 'Risk_Score' not in self.df.columns:
            print("Please build the risk prediction model first.")
            return
        
        print(f"\\n=== ASSIGNING RECOVERY STRATEGIES ===")
        
        def assign_recovery_strategy(risk_score):
            if risk_score > 0.75:
                return "Immediate legal notices & aggressive recovery attempts"
            elif 0.50 <= risk_score <= 0.75:
                return "Settlement offers & repayment plans"
            else:
                return "Automated reminders & monitoring"
        
        self.df['Recovery_Strategy'] = self.df['Risk_Score'].apply(assign_recovery_strategy)
        
        print("Recovery Strategy Distribution:")
        strategy_counts = self.df['Recovery_Strategy'].value_counts()
        for strategy, count in strategy_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"{strategy}: {count} borrowers ({percentage:.1f}%)")
        
        # Analyze strategy effectiveness by segment
        print("\\n=== STRATEGY BY SEGMENT ===")
        strategy_segment = pd.crosstab(self.df['Segment_Name'], 
                                     self.df['Recovery_Strategy'], 
                                     normalize='index') * 100
        print(strategy_segment.round(1))
        
        return self.df[['Borrower_ID', 'Risk_Score', 'Recovery_Strategy', 'Segment_Name']].head(10)
    
    def generate_insights(self):
        """
        Generate key insights from the analysis
        """
        if self.df is None:
            print("Please load and analyze data first.")
            return
        
        print(f"\\n=== KEY INSIGHTS ===")
        
        # Recovery insights
        recovery_stats = self.df.groupby('Recovery_Status').agg({
            'Monthly_Income': 'mean',
            'Loan_Amount': 'mean',
            'Num_Missed_Payments': 'mean',
            'Days_Past_Due': 'mean'
        }).round(2)
        
        print("1. Recovery Status Analysis:")
        print(recovery_stats)
        
        # Risk insights
        if 'Risk_Score' in self.df.columns:
            high_risk = self.df[self.df['Risk_Score'] > 0.75]
            print(f"\\n2. High-Risk Borrowers ({len(high_risk)} borrowers):")
            print(f"   Average Outstanding Amount: ${high_risk['Outstanding_Loan_Amount'].mean():.2f}")
            print(f"   Average Days Past Due: {high_risk['Days_Past_Due'].mean():.0f} days")
            print(f"   Average Missed Payments: {high_risk['Num_Missed_Payments'].mean():.1f}")
        
        # Income insights
        income_recovery = self.df.groupby(pd.qcut(self.df['Monthly_Income'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']))['Recovery_Status'].apply(lambda x: (x == 'Full Recovery').mean() * 100).round(1)
        print("\\n3. Recovery Rate by Income Level:")
        print(income_recovery)
        
        # Segment insights
        if 'Segment_Name' in self.df.columns:
            segment_recovery = self.df.groupby('Segment_Name')['Recovery_Status'].apply(lambda x: (x == 'Full Recovery').mean() * 100).round(1)
            print("\\n4. Recovery Rate by Borrower Segment:")
            print(segment_recovery)
    
    def run_complete_analysis(self, data_path=None):
        """
        Run the complete loan recovery analysis pipeline
        
        Args:
            data_path (str): Path to the dataset
        """
        print("=== SMART LOAN RECOVERY SYSTEM ANALYSIS ===")
        print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        if data_path:
            self.load_data(data_path)
        
        if self.df is None:
            print("No data available. Please provide a valid data path.")
            return
        
        # Step 1: Explore data
        self.explore_data()
        
        # Step 2: Perform clustering
        self.perform_clustering()
        
        # Step 3: Build risk prediction model
        self.build_risk_prediction_model()
        
        # Step 4: Assign recovery strategies
        sample_strategies = self.assign_recovery_strategies()
        
        # Step 5: Generate insights
        self.generate_insights()
        
        # Step 6: Save results
        self.save_results()
        
        print(f"\\n=== ANALYSIS COMPLETED ===")
        print(f"Results saved to ../outputs/ directory")
        print("\\nSample Recovery Strategies:")
        print(sample_strategies)
    
    def save_results(self):
        """Save analysis results to files"""
        try:
            # Save enhanced dataset
            output_path = "../outputs/loan_recovery_analysis_results.csv"
            self.df.to_csv(output_path, index=False)
            print(f"\\nResults saved to: {output_path}")
            
            # Save summary report
            summary_path = "../outputs/analysis_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("SMART LOAN RECOVERY SYSTEM - ANALYSIS SUMMARY\\n")
                f.write("=" * 50 + "\\n\\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
                f.write(f"Dataset Size: {self.df.shape}\\n\\n")
                
                # Recovery status distribution
                f.write("RECOVERY STATUS DISTRIBUTION:\\n")
                recovery_dist = self.df['Recovery_Status'].value_counts()
                for status, count in recovery_dist.items():
                    pct = (count / len(self.df)) * 100
                    f.write(f"{status}: {count} ({pct:.1f}%)\\n")
                
                # Segment distribution
                if 'Segment_Name' in self.df.columns:
                    f.write("\\nBORROWER SEGMENTS:\\n")
                    segment_dist = self.df['Segment_Name'].value_counts()
                    for segment, count in segment_dist.items():
                        pct = (count / len(self.df)) * 100
                        f.write(f"{segment}: {count} ({pct:.1f}%)\\n")
                
                # Recovery strategies
                if 'Recovery_Strategy' in self.df.columns:
                    f.write("\\nRECOVERY STRATEGIES:\\n")
                    strategy_dist = self.df['Recovery_Strategy'].value_counts()
                    for strategy, count in strategy_dist.items():
                        pct = (count / len(self.df)) * 100
                        f.write(f"{strategy}: {count} ({pct:.1f}%)\\n")
            
            print(f"Summary report saved to: {summary_path}")
            
        except Exception as e:
            print(f"Error saving results: {e}")

# Example usage and main execution
if __name__ == "__main__":
    # Initialize the system
    recovery_system = SmartLoanRecoverySystem()
    
    # Check if data file exists, if not, generate sample data
    data_path = "../data/loan_recovery_data.csv"
    
    try:
        # Try to load existing data
        if not recovery_system.load_data(data_path):
            print("Generating sample data...")
            from generate_sample_data import generate_loan_data
            
            # Generate and save sample data
            sample_df = generate_loan_data(1000)
            sample_df.to_csv(data_path, index=False)
            print(f"Sample data saved to {data_path}")
            
            # Load the generated data
            recovery_system.load_data(data_path)
    
    except ImportError:
        print("Could not import sample data generator. Please ensure data file exists.")
        exit(1)
    
    # Run complete analysis
    recovery_system.run_complete_analysis()
    
    print("\\n" + "="*60)
    print("SMART LOAN RECOVERY SYSTEM ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)
