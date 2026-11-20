# FinGuard – Financial Fraud Detection & Analytics System

FinGuard is a Machine Learning-based financial fraud detection and analytics application built using Python, Scikit-learn, SQLAlchemy, and Streamlit.

The system analyzes financial transaction details, predicts potentially fraudulent transactions using a trained Random Forest model, and stores transaction results in a SQLite database.

## Features

- Financial transaction analysis
- Fraud prediction using a trained Random Forest model
- Fraud probability calculation
- Configurable prediction threshold
- Transaction storage using SQLite and SQLAlchemy
- Duplicate transaction prevention
- Interactive Streamlit dashboard
- Fraud analytics and data visualization
- Model performance information

## Technology Stack

| Category | Technology |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| Dashboard | Streamlit |
| Database | SQLite |
| ORM | SQLAlchemy |
| Model Storage | Joblib |
| Configuration | YAML |

## Project Structure

```text
Fraud_Detection/
│
├── config/
│   └── config.yaml
│
├── dashboard/
│   ├── app.py
│   └── style.css
│
├── database/
│   ├── connection.py
│   └── models.py
│
├── models/
│   └── fraud_detection_model.joblib
│
├── src/
│   ├── data_processing/
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   │
│   ├── models/
│   │   ├── train_model.py
│   │   └── evaluate_model.py
│   │
│   └── services/
│       ├── prediction_service.py
│       └── transaction_service.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd Fraud_Detection
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Dataset and Model Setup

The required raw and processed datasets and trained model are included in this private repository for deployment. Keep the repository private unless you have permission to publish those artifacts.

Place your authorized dataset files in:

- `data/raw/`
- `data/processed/`

The application uses the trained model file:

`models/fraud_detection_model.joblib`

Use the existing trained model artifact. The project does not require retraining to perform predictions.

## Run the Application

From the project root directory:

```bash
python -m streamlit run dashboard/app.py
```

The FinGuard dashboard will open in your browser.

## Docker

The Docker image contains the application, saved model, and the two CSV files required by the dashboard. Review those artifacts before building or pushing the image; do not publish a registry image that contains data you are not authorized to distribute.

Build and start FinGuard with Docker Compose:

```bash
docker compose up --build -d
```

Open `http://localhost:8501`. The Compose setup stores SQLite data in the named `finguard_database` volume. Stop the container with `docker compose down`; keep the volume to retain transactions.

## Kubernetes

Build the image with the tag referenced in `k8s/finguard.yaml`, make it available to the cluster, then apply the manifest:

```bash
docker build -t finguard:1.0.3 .
kubectl apply -f k8s/finguard.yaml
kubectl rollout status deployment/finguard -n finguard
kubectl port-forward service/finguard 8502:8501 -n finguard
```

Open `http://localhost:8502`. For a remote cluster, push the image to an approved registry and update the manifest's image reference. The deployment intentionally uses one replica because SQLite is stored on a persistent volume. Use a client/server database and a migration plan before scaling beyond one writer. The manifest does not configure public ingress or authentication; add the protections required by your environment before exposing the service.

## Deployment Links

- Final project report: [Financial Fraud Detection Report (PDF)](documentation/financial-fraud-detection-doc.pdf)
- Streamlit Cloud: [FinGuard app](https://finguard-fraud-detection.streamlit.app)
- Docker local app: http://localhost:8501
- Kubernetes local app: http://localhost:8502

The Docker and Kubernetes addresses are local to the machine running the services. The Streamlit app is private while its Community Cloud sharing setting is private.

## Machine Learning Model

- Algorithm: Random Forest Classifier
- Number of estimators: 200 (training configuration)
- Test size: 20% (training configuration)
- Random state: 42 (training configuration)
- Prediction threshold: 0.395 (currently saved model configuration)

The model returns a fraud probability and classifies transactions as Fraudulent or Genuine using the saved threshold.

## Database

FinGuard uses SQLite with SQLAlchemy to store transaction information and prediction results.

The database is created locally and is not included in the public repository.

## Important Notes

- This project is intended for educational and demonstration purposes.
- Predictions are model-based estimates and should not be treated as confirmed financial fraud.
- Do not upload private customer information, transaction datasets, database files, or credentials to a public repository.
- The existing trained model and prediction workflow should be preserved.

## Author

**Shekhar Shelke**

GitHub: https://github.com/shekharshelke171-eng

Portfolio: https://portfolio-y7l1.vercel.app/

LinkedIn: https://www.linkedin.com/in/shekhar-shelke-6936263a6/
