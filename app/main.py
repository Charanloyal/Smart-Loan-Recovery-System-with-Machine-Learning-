import os
import json
from contextlib import asynccontextmanager
from typing import Dict, Any, List
import pandas as pd
import joblib
from fastapi import FastAPI, Request, status, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# Global model artifacts placeholders
preprocessor_class = None
classifier_model = None
preprocessor_reg = None
regressor_model = None
model_metrics = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads all machine learning artifacts on startup."""
    global preprocessor_class, classifier_model, preprocessor_reg, regressor_model, model_metrics
    artifacts_dir = "ml/artifacts"
    
    # Path validation
    metrics_path = os.path.join(artifacts_dir, "metrics.json")
    clf_prep_path = os.path.join(artifacts_dir, "preprocessor_class.joblib")
    clf_model_path = os.path.join(artifacts_dir, "classifier.joblib")
    reg_prep_path = os.path.join(artifacts_dir, "preprocessor_reg.joblib")
    reg_model_path = os.path.join(artifacts_dir, "regressor.joblib")
    
    # Verify files exist (if not, we raise error to prompt training execution)
    missing = [p for p in [metrics_path, clf_prep_path, clf_model_path, reg_prep_path, reg_model_path] if not os.path.exists(p)]
    if missing:
        print(f"CRITICAL: Serialized ML artifacts missing: {missing}. Training must run first.")
    else:
        # Load artifacts using joblib
        preprocessor_class = joblib.load(clf_prep_path)
        classifier_model = joblib.load(clf_model_path)
        preprocessor_reg = joblib.load(reg_prep_path)
        regressor_model = joblib.load(reg_model_path)
        
        with open(metrics_path, "r") as f:
            model_metrics = json.load(f)
        print("Successfully loaded all ML models and metrics.")
        
    yield
    print("Application shutdown complete.")

app = FastAPI(
    title="Smart Loan Recovery API",
    description="A Fintech predictive credit risk and recovery analytics engine served by FastAPI and scikit-learn/XGBoost.",
    version="1.0.0",
    lifespan=lifespan
)

# Mount static and template directories
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")

# Pydantic input models
class LoanApplication(BaseModel):
    credit_score: int = Field(..., ge=300, le=850, description="Applicant Credit Score (FICO)")
    annual_income: float = Field(..., ge=10000, description="Applicant Annual Income in USD")
    loan_amount: float = Field(..., ge=500, le=50000, description="Requested Loan Amount")
    interest_rate: float = Field(..., ge=1.0, le=45.0, description="Assigned Loan Interest Rate (%)")
    debt_to_income: float = Field(..., ge=0.0, le=1.0, description="Debt to Income Ratio (DTI)")
    employment_length: int = Field(..., ge=0, le=40, description="Length of employment in years")
    home_ownership: str = Field(..., description="Home ownership status (RENT, MORTGAGE, OWN)")
    loan_purpose: str = Field(..., description="Purpose of the loan")

    model_config = {
        "json_schema_extra": {
            "example": {
                "credit_score": 680,
                "annual_income": 65000.0,
                "loan_amount": 15000.0,
                "interest_rate": 14.5,
                "debt_to_income": 0.28,
                "employment_length": 5,
                "home_ownership": "MORTGAGE",
                "loan_purpose": "debt_consolidation"
            }
        }
    }

# Serving the Dashboard HTML UI
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"model_loaded": classifier_model is not None}
    )

# Predictive evaluation endpoint
@app.post("/api/v1/predict")
async def evaluate_loan(application: LoanApplication):
    """
    Receives applicant details, runs ML pipelines:
    1. Preprocesses features using classification scaler/encoders.
    2. Runs XGBoost Classifier to output Default Probability.
    3. If Default Probability > 0.0, calculates the expected recovery rate using Regression.
    4. Computes risk tiers and recommended collection actions.
    """
    global preprocessor_class, classifier_model, preprocessor_reg, regressor_model
    
    if not classifier_model or not preprocessor_class:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, 
            detail="Machine learning models are not loaded. Please run model training first."
        )
        
    try:
        # Convert input into a single-row Pandas DataFrame matching features
        input_data = pd.DataFrame([{
            'credit_score': application.credit_score,
            'annual_income': application.annual_income,
            'loan_amount': application.loan_amount,
            'interest_rate': application.interest_rate,
            'debt_to_income': application.debt_to_income,
            'employment_length': application.employment_length,
            'home_ownership': application.home_ownership.upper(),
            'loan_purpose': application.loan_purpose.lower()
        }])
        
        # 1. Classification prediction
        X_clf_preprocessed = preprocessor_class.transform(input_data)
        prob_default = float(classifier_model.predict_proba(X_clf_preprocessed)[0, 1])
        
        # Determine risk tier and recommendation
        if prob_default < 0.15:
            risk_tier = "Low"
            recommendation = "Auto-Approve"
        elif prob_default < 0.45:
            risk_tier = "Medium"
            recommendation = "Manual Review"
        elif prob_default < 0.75:
            risk_tier = "High"
            recommendation = "Flagged Default / Collateral Hold"
        else:
            risk_tier = "Critical"
            recommendation = "Reject Application"
            
        # 2. Regression prediction (expected recovery rate)
        # Even if default is low, we predict the hypothetical recovery rate if default *were* to occur
        X_reg_preprocessed = preprocessor_reg.transform(input_data)
        predicted_recovery = float(regressor_model.predict(X_reg_preprocessed)[0])
        predicted_recovery = max(0.0, min(1.0, predicted_recovery)) # Clip boundaries
        
        # Compute expected recovery amount
        expected_recovery_amount = round(application.loan_amount * predicted_recovery, 2)
        
        return {
            "default_probability": round(prob_default, 4),
            "risk_tier": risk_tier,
            "recommended_action": recommendation,
            "predicted_recovery_rate": round(predicted_recovery, 4),
            "expected_recovery_amount": expected_recovery_amount,
            "status": "success"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline failure: {str(e)}"
        )

# Endpoint to fetch model metrics for Chart.js rendering
@app.get("/api/v1/metrics")
async def get_metrics():
    """Returns classification and regression metrics summary."""
    global model_metrics
    if not model_metrics:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, 
            detail="Model metrics not found. Run model training first."
        )
    return JSONResponse(content=model_metrics)
