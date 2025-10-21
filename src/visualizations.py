"""
Visualization module for Smart Loan Recovery System
Contains functions for generating various plots and charts for data analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_recovery_status_distribution(df, save_path=None):
    """Plot the distribution of recovery status"""
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    # Count plot
    recovery_counts = df['Recovery_Status'].value_counts()
    ax[0].pie(recovery_counts.values, labels=recovery_counts.index, autopct='%1.1f%%')
    ax[0].set_title('Recovery Status Distribution', fontsize=14, fontweight='bold')
    
    # Bar plot
    sns.countplot(data=df, x='Recovery_Status', ax=ax[1])
    ax[1].set_title('Recovery Status Counts', fontsize=14, fontweight='bold')
    ax[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_income_vs_loan_recovery(df, save_path=None):
    """Plot relationship between income and loan recovery"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Scatter plot
    colors = {'Full Recovery': 'green', 'Partial Recovery': 'orange', 'Write-off': 'red'}
    for status in df['Recovery_Status'].unique():
        subset = df[df['Recovery_Status'] == status]
        axes[0].scatter(subset['Monthly_Income'], subset['Loan_Amount'], 
                       c=colors.get(status, 'blue'), alpha=0.6, label=status)
    
    axes[0].set_xlabel('Monthly Income ($)')
    axes[0].set_ylabel('Loan Amount ($)')
    axes[0].set_title('Income vs Loan Amount by Recovery Status')
    axes[0].legend()
    
    # Box plot
    sns.boxplot(data=df, x='Recovery_Status', y='Monthly_Income', ax=axes[1])
    axes[1].set_title('Monthly Income Distribution by Recovery Status')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_heatmap(df, save_path=None):
    """Plot correlation heatmap of numerical features"""
    # Select numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_borrower_segments_interactive(df, save_path=None):
    """Create interactive plot of borrower segments"""
    if 'Borrower_Segment' not in df.columns:
        print("Borrower segments not found. Run clustering first.")
        return
    
    fig = px.scatter(df, x='Monthly_Income', y='Loan_Amount',
                     color='Borrower_Segment', 
                     hover_data=['Age', 'Outstanding_Loan_Amount', 'Recovery_Status'],
                     title='Borrower Segments Analysis',
                     labels={'Monthly_Income': 'Monthly Income ($)', 
                            'Loan_Amount': 'Loan Amount ($)'})
    
    fig.update_layout(
        template='plotly_white',
        title_font_size=16,
        legend_title='Borrower Segment'
    )
    
    if save_path:
        fig.write_html(save_path)
    
    fig.show()

def plot_risk_score_distribution(df, save_path=None):
    """Plot risk score distribution"""
    if 'Risk_Score' not in df.columns:
        print("Risk scores not found. Run the model first.")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Risk score histogram
    axes[0, 0].hist(df['Risk_Score'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel('Risk Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Risk Score Distribution')
    
    # Risk score by recovery status
    for status in df['Recovery_Status'].unique():
        subset = df[df['Recovery_Status'] == status]
        axes[0, 1].hist(subset['Risk_Score'], alpha=0.5, label=status, bins=20)
    axes[0, 1].set_xlabel('Risk Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Risk Score by Recovery Status')
    axes[0, 1].legend()
    
    # Box plot of risk scores by segment
    if 'Segment_Name' in df.columns:
        sns.boxplot(data=df, x='Segment_Name', y='Risk_Score', ax=axes[1, 0])
        axes[1, 0].set_title('Risk Score by Borrower Segment')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Recovery strategy distribution
    if 'Recovery_Strategy' in df.columns:
        strategy_counts = df['Recovery_Strategy'].value_counts()
        axes[1, 1].pie(strategy_counts.values, labels=strategy_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Recovery Strategy Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_feature_importance(model, feature_names, save_path=None):
    """Plot feature importance from the trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 8))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Feature Importance for Loan Recovery Prediction')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("Model doesn't have feature_importances_ attribute")

def plot_model_performance(y_true, y_pred, y_prob=None, save_path=None):
    """Plot model performance metrics"""
    from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
    from sklearn.preprocessing import label_binarize
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Confusion Matrix')
    axes[0, 0].set_xlabel('Predicted')
    axes[0, 0].set_ylabel('Actual')
    
    # Classification Report (as text)
    report = classification_report(y_true, y_pred)
    axes[0, 1].text(0.1, 0.5, report, fontsize=10, family='monospace')
    axes[0, 1].set_title('Classification Report')
    axes[0, 1].axis('off')
    
    # ROC Curve (if probabilities provided)
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        axes[1, 0].plot(fpr, tpr, color='darkorange', lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.2f})')
        axes[1, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('False Positive Rate')
        axes[1, 0].set_ylabel('True Positive Rate')
        axes[1, 0].set_title('ROC Curve')
        axes[1, 0].legend(loc="lower right")
    
    # Prediction Distribution
    pred_counts = pd.Series(y_pred).value_counts()
    axes[1, 1].pie(pred_counts.values, labels=pred_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Prediction Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def create_dashboard_plots(df, save_dir=None):
    """Create all dashboard plots"""
    plots_created = []
    
    # Recovery status distribution
    save_path = f"{save_dir}/recovery_status_distribution.png" if save_dir else None
    plot_recovery_status_distribution(df, save_path)
    if save_path:
        plots_created.append(save_path)
    
    # Income vs loan recovery
    save_path = f"{save_dir}/income_vs_loan_recovery.png" if save_dir else None
    plot_income_vs_loan_recovery(df, save_path)
    if save_path:
        plots_created.append(save_path)
    
    # Correlation heatmap
    save_path = f"{save_dir}/correlation_heatmap.png" if save_dir else None
    plot_correlation_heatmap(df, save_path)
    if save_path:
        plots_created.append(save_path)
    
    # Interactive borrower segments
    if 'Borrower_Segment' in df.columns:
        save_path = f"{save_dir}/borrower_segments.html" if save_dir else None
        plot_borrower_segments_interactive(df, save_path)
        if save_path:
            plots_created.append(save_path)
    
    # Risk score distribution
    if 'Risk_Score' in df.columns:
        save_path = f"{save_dir}/risk_score_distribution.png" if save_dir else None
        plot_risk_score_distribution(df, save_path)
        if save_path:
            plots_created.append(save_path)
    
    print(f"Created {len(plots_created)} plots")
    return plots_created
