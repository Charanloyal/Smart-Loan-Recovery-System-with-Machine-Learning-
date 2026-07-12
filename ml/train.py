import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor

def train_models(data_path: str, artifacts_dir: str):
    """
    Implements a strict, industry-standard ML pipeline:
    1. Loads dataset and handles preprocessing correctly to avoid data leakage.
    2. Trains and compares Logistic Regression, Random Forest, and XGBoost models for Default Classification.
    3. Trains and compares Linear Regression, Random Forest, and XGBoost models for Recovery Rate Regression (on defaults).
    4. Serializes the best performing models, preprocessors, and evaluation metrics.
    """
    print("Loading dataset...")
    df = pd.read_csv(data_path)
    
    # Define features
    num_cols = ['credit_score', 'annual_income', 'loan_amount', 'interest_rate', 'debt_to_income', 'employment_length']
    cat_cols = ['home_ownership', 'loan_purpose']
    
    X = df[num_cols + cat_cols]
    y_class = df['is_default']
    y_reg = df['recovery_rate']
    
    # ----------------------------------------------------
    # PHASE 1: Default Classification Model (All records)
    # ----------------------------------------------------
    print("\n--- Training Default Classification Models ---")
    
    # Strict split BEFORE fitting any preprocessing
    X_train, X_test, y_train_class, y_test_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    
    # Fit preprocessing pipeline on training data ONLY
    preprocessor_class = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_preprocessed = preprocessor_class.fit_transform(X_train)
    X_test_preprocessed = preprocessor_class.transform(X_test)
    
    # Model 1: Logistic Regression Baseline
    lr_clf = LogisticRegression(random_state=42, max_iter=1000)
    lr_clf.fit(X_train_preprocessed, y_train_class)
    
    # Model 2: Random Forest
    rf_clf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=8)
    rf_clf.fit(X_train_preprocessed, y_train_class)
    
    # Model 3: XGBoost (State of the art)
    xgb_clf = XGBClassifier(random_state=42, n_estimators=100, max_depth=5, learning_rate=0.05, eval_metric='logloss')
    xgb_clf.fit(X_train_preprocessed, y_train_class)
    
    # Evaluate classifiers
    classifiers = {
        'Logistic Regression': lr_clf,
        'Random Forest': rf_clf,
        'XGBoost': xgb_clf
    }
    
    clf_metrics = {}
    best_clf_name = None
    best_clf_f1 = -1
    
    for name, model in classifiers.items():
        preds = model.predict(X_test_preprocessed)
        probs = model.predict_proba(X_test_preprocessed)[:, 1]
        
        acc = accuracy_score(y_test_class, preds)
        prec = precision_score(y_test_class, preds)
        rec = recall_score(y_test_class, preds)
        f1 = f1_score(y_test_class, preds)
        auc = roc_auc_score(y_test_class, probs)
        
        print(f"{name} -> Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC-AUC: {auc:.4f}")
        
        clf_metrics[name] = {
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1_score': round(f1, 4),
            'roc_auc': round(auc, 4)
        }
        
        if f1 > best_clf_f1:
            best_clf_f1 = f1
            best_clf_name = name
            
    print(f"Winner Classification Model: {best_clf_name} (F1: {best_clf_f1:.4f})")
    
    # ----------------------------------------------------
    # PHASE 2: Recovery Rate Regression Model (Defaults Only)
    # ----------------------------------------------------
    print("\n--- Training Recovery Rate Regression Models ---")
    
    # Filter dataset for defaults only
    df_default = df[df['is_default'] == 1].copy()
    X_reg = df_default[num_cols + cat_cols]
    y_reg_filtered = df_default['recovery_rate']
    
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg_filtered, test_size=0.2, random_state=42
    )
    
    preprocessor_reg = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_reg_preprocessed = preprocessor_reg.fit_transform(X_train_reg)
    X_test_reg_preprocessed = preprocessor_reg.transform(X_test_reg)
    
    # Model 1: Linear Regression Baseline
    lr_reg = LinearRegression()
    lr_reg.fit(X_train_reg_preprocessed, y_train_reg)
    
    # Model 2: Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=6)
    rf_reg.fit(X_train_reg_preprocessed, y_train_reg)
    
    # Model 3: XGBoost Regressor
    xgb_reg = XGBRegressor(random_state=42, n_estimators=100, max_depth=4, learning_rate=0.05)
    xgb_reg.fit(X_train_reg_preprocessed, y_train_reg)
    
    regressors = {
        'Linear Regression': lr_reg,
        'Random Forest': rf_reg,
        'XGBoost': xgb_reg
    }
    
    reg_metrics = {}
    best_reg_name = None
    best_reg_r2 = -9999
    
    for name, model in regressors.items():
        preds = model.predict(X_test_reg_preprocessed)
        
        r2 = r2_score(y_test_reg, preds)
        mae = mean_absolute_error(y_test_reg, preds)
        rmse = np.sqrt(mean_squared_error(y_test_reg, preds))
        
        print(f"{name} -> R2: {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")
        
        reg_metrics[name] = {
            'r2_score': round(r2, 4),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4)
        }
        
        if r2 > best_reg_r2:
            best_reg_r2 = r2
            best_reg_name = name
            
    print(f"Winner Regression Model: {best_reg_name} (R2: {best_reg_r2:.4f})")
    
    # ----------------------------------------------------
    # PHASE 3: Save Artifacts
    # ----------------------------------------------------
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save winner models and their corresponding preprocessors
    best_clf_model = classifiers[best_clf_name]
    best_reg_model = regressors[best_reg_name]
    
    joblib.dump(preprocessor_class, os.path.join(artifacts_dir, "preprocessor_class.joblib"))
    joblib.dump(best_clf_model, os.path.join(artifacts_dir, "classifier.joblib"))
    
    joblib.dump(preprocessor_reg, os.path.join(artifacts_dir, "preprocessor_reg.joblib"))
    joblib.dump(best_reg_model, os.path.join(artifacts_dir, "regressor.joblib"))
    
    # Save metadata metric files for dashboard visualizer
    metrics_summary = {
        'classification': clf_metrics,
        'regression': reg_metrics,
        'best_classification': best_clf_name,
        'best_regression': best_reg_name
    }
    
    with open(os.path.join(artifacts_dir, "metrics.json"), "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"\nAll artifacts successfully saved to {artifacts_dir}")

if __name__ == "__main__":
    train_models("ml/loan_data.csv", "ml/artifacts")
