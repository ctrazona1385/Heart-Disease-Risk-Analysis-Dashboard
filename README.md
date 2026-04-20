# Heart Disease Risk Prediction Dashboard

A machine learning-powered web application that predicts heart disease risk and explains contributing factors using SHAP. Deployed as a containerized application on Microsoft Azure Container Apps.

---

## Live Deployment

**URL:** https://heart-disease-dashboard.gentlemoss-1b6edc24.eastus.azurecontainerapps.io

---

## What This Project Does

Users enter their health metrics (blood pressure, cholesterol, heart rate, etc.) and receive:
- A **risk score** (0–100%) with a color-coded gauge
- **Clinical threshold indicators** comparing their values against medical guidelines
- A **SHAP explainability chart** showing which factors are driving their risk — filtered to only actionable factors the patient can actually change
- **Personalized recommendations** scaled to the severity of each metric

---

## Machine Learning Models

### Dataset
- **Source:** UCI Heart Disease Dataset (Cleveland subset)
- **Size:** 297 patients after cleaning
- **Target:** Binary — presence or absence of heart disease
- **Features:** 21 features after one-hot encoding (age, cholesterol, blood pressure, chest pain type, thalassemia, ST depression, and more)

### Models Trained
| Model | Base AUC | Tuned AUC |
|---|---|---|
| Logistic Regression | 0.952 | Tuned via RandomizedSearchCV |
| Random Forest | 0.940 | Tuned via RandomizedSearchCV |
| Gradient Boosting | 0.909 | Baseline only |

### Hyperparameter Tuning
Both Logistic Regression and Random Forest were tuned using `RandomizedSearchCV` with 5-fold cross-validation scored on ROC-AUC:

**Random Forest parameters searched:**
- `n_estimators`: [50, 100, 200, 300, 500]
- `max_depth`: [None, 5, 10, 15, 20]
- `min_samples_split`: [2, 5, 10]
- `max_features`: ['sqrt', 'log2', 0.5, 0.7]

**Logistic Regression parameters searched:**
- `C`: log-uniform distribution [0.001, 100]
- `solver`: ['lbfgs', 'liblinear']
- `penalty`: ['l2']

### Model Selection
The best model by ROC-AUC is saved as `model.pkl` for predictions. Random Forest is always used for SHAP explainability (`shap_model.pkl`) since `TreeExplainer` requires a tree-based model.

### Explainability (SHAP)
SHAP (SHapley Additive exPlanations) values are computed using `TreeExplainer` on the tuned Random Forest. The dashboard filters SHAP output to show only **actionable features** — factors the patient can influence:
- Cholesterol
- Resting Blood Pressure
- Max Heart Rate
- ST Depression
- ST Slope

Non-actionable clinical factors (vessel count, thalassemia type, ECG results, sex) are still used by the model but are not shown to the patient.

---

## Project Files

```
├── heart_ml.ipynb        # Full ML pipeline: EDA, training, tuning, SHAP, export
├── dashboard.py          # Streamlit web application
├── heart_disease_uci.csv # Raw dataset (not included in container)
├── model.pkl             # Trained prediction model (Logistic Regression)
├── shap_model.pkl        # SHAP explainability model (Random Forest)
├── scaler.pkl            # StandardScaler for numeric features
├── feature_cols.pkl      # Feature column names for OHE alignment
├── Dockerfile            # Container image definition
├── requirements.txt      # Python dependencies (pinned versions)
├── .dockerignore         # Files excluded from the Docker build
├── deploy_azure.sh       # Azure Container Apps deployment script
└── locustfile.py         # Locust load testing configuration
```

---

## Containerization

The application is packaged as a Docker container, making it portable and reproducible across any environment.

### Why Containerization?
- **Reproducibility:** The exact Python version (3.14), all dependencies, and model artifacts are baked into the image — it runs identically on any machine or cloud provider
- **Isolation:** No conflicts with system Python or other projects
- **Portability:** The same image runs locally, on Azure, or any other cloud provider
- **Scalability:** Azure Container Apps can automatically spin up multiple instances under load

### What the Container Includes
The Dockerfile explicitly copies only what the app needs:
```
dashboard.py  model.pkl  shap_model.pkl  scaler.pkl  feature_cols.pkl
```
Everything else (notebook, raw data, venv, images, documents) is excluded via `.dockerignore`, keeping the image lean.

### Building Locally
```bash
docker build -t heart-dashboard .
docker run -p 8501:8501 heart-dashboard
# Open http://localhost:8501
```

---

## Cloud Deployment (Microsoft Azure)

### Why This is a Cloud Project
- **Managed infrastructure:** No server to maintain — Azure handles OS updates, networking, and hardware
- **Auto-scaling:** Container Apps automatically scales from 0 to 3 replicas based on traffic
- **Scale-to-zero:** The app shuts down when idle and cold-starts on demand, minimizing cost
- **Global availability:** Deployed to Azure's East US region, accessible from anywhere via HTTPS
- **Managed HTTPS:** Azure provides TLS termination automatically — no certificate management needed
- **Container Registry:** Docker image is stored in Azure Container Registry (ACR), versioned and ready for redeployment

### Azure Services Used
| Service | Purpose |
|---|---|
| Azure Container Registry (ACR) | Stores and versions the Docker image |
| Azure Container Apps | Runs the containerized dashboard |
| Azure Container Apps Environment | Manages networking and shared infrastructure |

### Deployment Architecture
```
Local Build
    │
    ▼
Docker Image
    │
    ▼
Azure Container Registry (heartdashboardcyron)
    │
    ▼
Azure Container Apps Environment (heart-disease-env)
    │
    ▼
Azure Container App (heart-disease-dashboard)
    │
    ▼
Public HTTPS URL → Users
```

### Deploying
```bash
# Build and push image
docker build -t heart-dashboard .
az acr login --name heartdashboardcyron
docker tag heart-dashboard heartdashboardcyron.azurecr.io/heart-dashboard:latest
docker push heartdashboardcyron.azurecr.io/heart-dashboard:latest

# Deploy to Azure Container Apps
bash deploy_azure.sh
```

### Scaling Controls
```bash
# Turn off (zero cost)
az containerapp update --name heart-disease-dashboard --resource-group heart-disease-rg --min-replicas 0 --max-replicas 0

# Turn on (scale-to-zero)
az containerapp update --name heart-disease-dashboard --resource-group heart-disease-rg --min-replicas 0 --max-replicas 3

# Always on (no cold start)
az containerapp update --name heart-disease-dashboard --resource-group heart-disease-rg --min-replicas 1
```

---

## Load Testing

[Locust](https://locust.io) is used to validate performance of the deployed container under simulated traffic.

```bash
# Install
pip install locust

# Run against deployed Azure app
locust -f locustfile.py

# Run against local dashboard (for comparison)
locust -f locustfile.py --host http://localhost:8501
```

Open `http://localhost:8089` to control the test and view live metrics (requests/sec, response times, failure rate).

---

## Running Locally

```bash
# Clone and set up environment
python -m venv venv
source venv/Scripts/activate  # Windows
pip install -r requirements.txt

# Run the notebook first to generate model artifacts
# Open heart_ml.ipynb and run all cells

# Launch dashboard
streamlit run dashboard.py
```

---

## Dependencies

Key packages (pinned in `requirements.txt`):
- `streamlit==1.56.0` — Web dashboard framework
- `scikit-learn==1.8.0` — Machine learning models
- `shap==0.51.0` — Model explainability
- `plotly==6.6.0` — Interactive visualizations
- `pandas==3.0.2` / `numpy==2.4.4` — Data processing
- `scipy==1.17.1` — Hyperparameter search distributions
