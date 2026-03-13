# FraudGuard — End-to-End Fraud Detection

**Poorani M · IIT Dhanbad · Data Science Portfolio**
Check It Out at: [https://frauddetection.up.railway.app/](url)

## Quick Start

```bash
# 1. Install dependencies
pip install fastapi uvicorn scikit-learn pandas numpy pydantic

# 2. Train the model (already done — skip if analysis.json exists)
python3 analysis/engine.py

# 3. Start the API server
python3 app.py
# OR
uvicorn app:app --reload --port 8000

# 4. Open browser
http://localhost:8000
```

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Full dashboard UI |
| `/api/health` | GET | Server + model status |
| `/api/analysis` | GET | All training results (EDA, metrics, SHAP) |
| `/api/predict` | POST | Predict fraud for a transaction |
| `/api/example/fraud` | GET | Example fraud transaction |
| `/api/example/legit` | GET | Example legitimate transaction |
| `/docs` | GET | Swagger UI (auto-generated) |

## Stack

- **ML**: Scikit-learn — Logistic Regression, Random Forest, Gradient Boosting
- **Imbalance**: SMOTE (manual implementation, no imblearn required)
- **API**: FastAPI + Pydantic
- **Frontend**: Pure HTML/CSS/JS with Canvas charts (zero CDN dependencies)
