"""
Generate sample loan recovery data for the Smart Loan Recovery System
This script creates synthetic data with realistic patterns for demonstration purposes.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_loan_data(n_samples=1000):
    """
    Generate synthetic loan recovery dataset
    
    Args:
        n_samples (int): Number of samples to generate
        
    Returns:
        pd.DataFrame: Generated loan dataset
    """
    np.random.seed(42)
    random.seed(42)
    
    # Generate demographic information
    ages = np.random.normal(35, 10, n_samples).astype(int)
    ages = np.clip(ages, 18, 70)
    
    # Employment types with realistic distribution
    employment_types = np.random.choice(
        ['Salaried', 'Self-employed', 'Business', 'Unemployed', 'Retired'],
        size=n_samples,
        p=[0.5, 0.25, 0.15, 0.05, 0.05]
    )
    
    # Monthly income based on employment type and age
    base_income = []
    for i, emp_type in enumerate(employment_types):
        if emp_type == 'Salaried':
            income = np.random.normal(5000, 2000)
        elif emp_type == 'Self-employed':
            income = np.random.normal(4000, 2500)
        elif emp_type == 'Business':
            income = np.random.normal(7000, 3000)
        elif emp_type == 'Unemployed':
            income = np.random.normal(1000, 500)
        else:  # Retired
            income = np.random.normal(2000, 800)
        
        # Age factor
        if ages[i] > 45:
            income *= 1.2
        elif ages[i] < 25:
            income *= 0.8
            
        base_income.append(max(500, income))
    
    monthly_income = np.array(base_income)
    
    # Number of dependents
    num_dependents = np.random.poisson(1.5, n_samples)
    num_dependents = np.clip(num_dependents, 0, 6)
    
    # Loan details
    loan_amounts = []
    loan_tenures = []
    interest_rates = []
    
    for income in monthly_income:
        # Loan amount based on income (typically 40-60x monthly income)
        loan_multiplier = np.random.uniform(30, 80)
        loan_amount = income * loan_multiplier
        loan_amounts.append(loan_amount)
        
        # Loan tenure (6 months to 10 years)
        tenure = np.random.choice([12, 24, 36, 48, 60, 84, 120], p=[0.1, 0.2, 0.25, 0.2, 0.15, 0.08, 0.02])
        loan_tenures.append(tenure)
        
        # Interest rate based on loan amount and tenure
        if loan_amount > 100000:
            rate = np.random.uniform(8, 15)
        elif loan_amount > 50000:
            rate = np.random.uniform(10, 18)
        else:
            rate = np.random.uniform(12, 22)
        interest_rates.append(rate)
    
    loan_amounts = np.array(loan_amounts)
    loan_tenures = np.array(loan_tenures)
    interest_rates = np.array(interest_rates)
    
    # Calculate monthly EMI
    monthly_rates = interest_rates / (12 * 100)
    monthly_emi = (loan_amounts * monthly_rates * (1 + monthly_rates)**loan_tenures) / \
                  ((1 + monthly_rates)**loan_tenures - 1)
    
    # Collateral value (0-150% of loan amount)
    collateral_values = loan_amounts * np.random.uniform(0, 1.5, n_samples)
    
    # Outstanding loan amount (20-100% of original)
    outstanding_ratios = np.random.beta(2, 2, n_samples)  # Beta distribution for realistic spread
    outstanding_loan_amounts = loan_amounts * outstanding_ratios
    
    # Payment behavior
    num_missed_payments = np.random.poisson(2, n_samples)
    num_missed_payments = np.clip(num_missed_payments, 0, 12)
    
    days_past_due = []
    for missed in num_missed_payments:
        if missed == 0:
            days = 0
        elif missed <= 2:
            days = np.random.uniform(0, 60)
        elif missed <= 5:
            days = np.random.uniform(30, 180)
        else:
            days = np.random.uniform(90, 365)
        days_past_due.append(days)
    
    days_past_due = np.array(days_past_due)
    
    # Collection efforts
    collection_methods = np.random.choice(
        ['Phone', 'Email', 'SMS', 'Field Visit', 'Legal Notice'],
        size=n_samples,
        p=[0.3, 0.25, 0.2, 0.15, 0.1]
    )
    
    recovery_attempts = np.random.poisson(3, n_samples)
    recovery_attempts = np.clip(recovery_attempts, 1, 15)
    
    legal_actions = np.random.choice([0, 1], size=n_samples, p=[0.8, 0.2])
    
    # Recovery status based on risk factors
    recovery_status = []
    for i in range(n_samples):
        # Calculate risk score based on multiple factors
        income_ratio = monthly_emi[i] / monthly_income[i]
        collateral_ratio = collateral_values[i] / loan_amounts[i]
        payment_history_score = num_missed_payments[i] / 12
        
        risk_score = (income_ratio * 0.4 + 
                     payment_history_score * 0.3 + 
                     (1 - collateral_ratio) * 0.2 + 
                     (days_past_due[i] / 365) * 0.1)
        
        if risk_score > 0.7:
            status = np.random.choice(['Write-off', 'Partial Recovery'], p=[0.7, 0.3])
        elif risk_score > 0.4:
            status = np.random.choice(['Full Recovery', 'Partial Recovery'], p=[0.3, 0.7])
        else:
            status = np.random.choice(['Full Recovery', 'Partial Recovery'], p=[0.8, 0.2])
        
        recovery_status.append(status)
    
    # Create DataFrame
    data = {
        'Borrower_ID': [f'BOR_{i+1:05d}' for i in range(n_samples)],
        'Age': ages,
        'Employment_Type': employment_types,
        'Monthly_Income': monthly_income.round(2),
        'Num_Dependents': num_dependents,
        'Loan_Amount': loan_amounts.round(2),
        'Loan_Tenure': loan_tenures,
        'Interest_Rate': interest_rates.round(2),
        'Collateral_Value': collateral_values.round(2),
        'Outstanding_Loan_Amount': outstanding_loan_amounts.round(2),
        'Monthly_EMI': monthly_emi.round(2),
        'Num_Missed_Payments': num_missed_payments,
        'Days_Past_Due': days_past_due.round(0).astype(int),
        'Collection_Method': collection_methods,
        'Num_Recovery_Attempts': recovery_attempts,
        'Legal_Action_Taken': legal_actions,
        'Recovery_Status': recovery_status
    }
    
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    # Generate sample data
    print("Generating sample loan recovery data...")
    df = generate_loan_data(1000)
    
    # Save to CSV
    output_path = "../data/loan_recovery_data.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Sample data generated and saved to {output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nDataset preview:")
    print(df.head())
    
    print("\nDataset info:")
    print(df.info())
