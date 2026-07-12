import os
import numpy as np
import pandas as pd

def generate_loan_dataset(output_path: str, num_samples: int = 10000, seed: int = 42):
    """
    Generates a synthetic, highly realistic loan dataset for credit risk modeling.
    Contains both classification (is_default) and regression (recovery_rate) targets.
    """
    np.random.seed(seed)
    
    # 1. Generate features
    credit_score = np.random.normal(670, 75, num_samples).astype(int)
    credit_score = np.clip(credit_score, 300, 850)
    
    annual_income = np.random.exponential(65000, num_samples) + 20000
    annual_income = np.clip(annual_income, 15000, 300000)
    
    # Loan amount is moderately correlated with annual income
    loan_amount = (annual_income * 0.3 * np.random.uniform(0.5, 1.5, num_samples))
    loan_amount = np.clip(loan_amount, 1000, 40000).astype(int)
    
    # Interest rate is strongly correlated with credit score (higher score = lower rate)
    base_rate = 35.0 - (credit_score - 300) * 0.05
    interest_rate = base_rate + np.random.normal(0, 2.5, num_samples)
    interest_rate = np.clip(interest_rate, 5.0, 36.0)
    
    # Debt to income ratio (DTI)
    debt_to_income = np.random.beta(2, 5, num_samples) * 0.7
    debt_to_income = np.clip(debt_to_income, 0.02, 0.65)
    
    employment_length = np.random.randint(0, 11, num_samples)
    
    home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN'], 
        size=num_samples, 
        p=[0.4, 0.5, 0.1]
    )
    
    loan_purpose = np.random.choice(
        ['debt_consolidation', 'credit_card', 'home_improvement', 'major_purchase', 'small_business'],
        size=num_samples,
        p=[0.5, 0.25, 0.12, 0.08, 0.05]
    )
    
    # 2. Generate Targets using mathematical logic
    # Default risk increases with high DTI, low credit score, high interest rate, and high loan amount relative to income
    log_odds = (
        2.2
        - (credit_score - 600) * 0.012
        + (debt_to_income * 2.8)
        - np.log(annual_income / 10000) * 0.6
        + (interest_rate / 10.0) * 0.5
        + (loan_amount / annual_income) * 1.5
        - (employment_length * 0.05)
    )
    
    # Add random noise to make the classification non-deterministic
    noise = np.random.normal(0, 0.5, num_samples)
    prob_default = 1 / (1 + np.exp(-(log_odds + noise)))
    
    is_default = (np.random.uniform(0, 1, num_samples) < prob_default).astype(int)
    
    # Recovery rate is only applicable when default occurs
    # Recovery rate decreases with high loan amount and low credit score, increases with home ownership (collateral)
    base_recovery = 0.55 - (loan_amount / 100000) + (employment_length * 0.02)
    # Adjust for home ownership (MORTGAGE/OWN indicates assets available for recovery)
    asset_modifier = np.where(home_ownership == 'OWN', 0.15, np.where(home_ownership == 'MORTGAGE', 0.05, 0.0))
    recovery_rate_raw = base_recovery + asset_modifier + np.random.normal(0, 0.1, num_samples)
    recovery_rate = np.clip(recovery_rate_raw, 0.0, 1.0)
    
    # 3. Create DataFrame and Save
    df = pd.DataFrame({
        'credit_score': credit_score,
        'annual_income': np.round(annual_income, 2),
        'loan_amount': loan_amount,
        'interest_rate': np.round(interest_rate, 2),
        'debt_to_income': np.round(debt_to_income, 4),
        'employment_length': employment_length,
        'home_ownership': home_ownership,
        'loan_purpose': loan_purpose,
        'is_default': is_default,
        'recovery_rate': np.round(recovery_rate, 4)
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {num_samples} loan records and saved to {output_path}")

if __name__ == "__main__":
    generate_loan_dataset("ml/loan_data.csv")
