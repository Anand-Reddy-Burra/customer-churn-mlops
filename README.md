# Customer Churn Prediction – End-to-End MLOps

## Project Overview

This project implements an end-to-end Machine Learning and MLOps pipeline for predicting customer churn in a telecommunications company.

The system takes customer information such as tenure, contract type, internet service, monthly charges, payment method, and service usage, and predicts whether the customer is likely to churn.

The project covers the complete workflow:

**Dataset → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → MLflow → DVC → FastAPI → Streamlit → Pytest → Docker → GitHub Actions → AWS EC2**

---

## Business Problem

Telecommunication companies lose revenue when customers discontinue their services.

The objective of this project is to build a machine learning system that can identify customers who are likely to churn so that the business can take preventive retention actions.

### Business Questions

- Which customers are likely to churn?
- What is the probability of customer churn?
- Which customer characteristics are associated with churn?
- Which machine learning model provides the best churn prediction performance?

---

## Dataset

**Dataset:** IBM Telco Customer Churn Dataset

The dataset contains approximately 7,000 customer records and includes customer demographics, services, contract information, charges, and churn status.

### Target Variable

`Churn`

Values:

- `Yes`
- `No`

### Dataset Size

```text
7043 rows
21 columns
```

---

## Project Architecture

```text
                    Customer Churn Dataset
                              |
                              v
                    Data Preprocessing
                              |
                              v
                    Feature Engineering
                              |
                              v
                      Model Training
                              |
                +-------------+-------------+
                |             |             |
                v             v             v
          Logistic       Decision Tree   Random Forest
         Regression
                |             |             |
                +-------------+-------------+
                              |
                              v
                       Model Evaluation
                              |
                              v
                    Best Model Selection
                              |
                    +---------+---------+
                    |                   |
                    v                   v
                 MLflow                DVC
                    |                   |
                    +---------+---------+
                              |
                              v
                       Trained Model
                              |
                              v
                         FastAPI
                              |
                              v
                        Streamlit
                              |
                              v
                           Docker
                              |
                              v
                       AWS EC2 Deployment
```

---

## Project Structure

```text
customer-churn-mlops/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data/
│   │   └── preprocess.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
│   │
│   └── utils/
│       └── logger.py
│
├── api/
│   └── main.py
│
├── app/
│   └── streamlit_app.py
│
├── tests/
│   ├── test_preprocessing.py
│   └── test_api.py
│
├── artifacts/
│   └── model/
│
├── logs/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── requirements-docker.txt
├── requirements-frontend.txt
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── .gitignore
└── README.md
```

---

## Data Preprocessing

The preprocessing pipeline performs the following operations:

- Loads the raw customer churn dataset
- Handles missing values
- Converts `TotalCharges` to numeric
- Removes `customerID`
- Performs train/test splitting
- Uses stratification for the target variable
- Encodes categorical features
- Scales numerical features
- Uses median imputation for numerical variables
- Uses most-frequent imputation for categorical variables
- Uses Scikit-learn pipelines to prevent data leakage
- Saves processed datasets

### Preprocessing Pipeline

```text
Raw Dataset
     |
     v
Missing Value Handling
     |
     v
Data Type Conversion
     |
     v
Remove Customer ID
     |
     v
Train/Test Split
     |
     v
Numerical Scaling
     |
     v
Categorical Encoding
     |
     v
Processed Dataset
```

---

## Feature Engineering

Additional features were created to improve the information available to the machine learning models.

### Engineered Features

#### Tenure Group

Groups customers based on their tenure:

```text
0-12
13-24
25-48
49-72
```

#### Monthly Charge Category

Customers are categorized into:

```text
Low
Medium
High
```

#### Average Monthly Spend

Calculated using:

```text
TotalCharges / Tenure
```

#### Service Count

Counts the number of subscribed services for each customer.

### Final Engineered Dataset

```text
Original columns: 21
Engineered columns: 4
Final columns: 25
```

---

## Machine Learning Models

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

---

## Model Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7977 | 0.6478 | 0.5214 | 0.5778 | 0.8421 |
| Decision Tree | 0.7984 | 0.6347 | 0.5668 | 0.5989 | 0.8310 |
| Random Forest | 0.7921 | 0.6392 | 0.4973 | 0.5594 | 0.8374 |

### Best Model

The **Decision Tree** was selected as the final model because it achieved the highest F1 score.

```text
Best Model: Decision Tree
F1 Score: 0.5989
Accuracy: 0.7984
Precision: 0.6347
Recall: 0.5668
ROC-AUC: 0.8310
```

The trained model is saved as:

```text
artifacts/model/model.joblib
```

---

## MLflow

MLflow is used to track machine learning experiments and model performance.

The following information is tracked:

- Model parameters
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Trained models

The project logs separate MLflow runs for:

- Logistic Regression
- Decision Tree
- Random Forest

This makes it easier to compare experiments and select the best-performing model.

---

## DVC

DVC is used for data and pipeline versioning.

The project contains two major pipeline stages:

```text
+------------+
| preprocess |
+------------+
      |
      v
+-------+
| train |
+-------+
```

### DVC Commands

Check pipeline status:

```bash
dvc status
```

Display pipeline:

```bash
dvc dag
```

Reproduce pipeline:

```bash
dvc repro
```

DVC helps make the preprocessing and training workflow reproducible.

---

## FastAPI

FastAPI provides the machine learning prediction API.

### API Endpoints

#### Root

```text
GET /
```

Returns information about the API.

#### Health Check

```text
GET /health
```

Checks whether the model is successfully loaded.

Example response:

```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### Model Information

```text
GET /model-info
```

Returns information about the selected machine learning model.

#### Prediction

```text
POST /predict
```

Accepts customer information and returns the predicted churn result and probability.

Example response:

```json
{
  "churn_prediction": "Yes",
  "churn_probability": 0.75
}
```

### Run FastAPI Locally

```bash
uvicorn api.main:app --reload
```

Swagger documentation:

```text
http://localhost:8000/docs
```

---

## Streamlit Dashboard

A Streamlit dashboard provides an interactive interface for customer churn prediction.

The dashboard allows users to enter:

- Gender
- Senior Citizen status
- Partner
- Dependents
- Tenure
- Phone Service
- Multiple Lines
- Internet Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Contract
- Paperless Billing
- Payment Method
- Monthly Charges
- Total Charges

The dashboard sends the customer information to the FastAPI backend and displays:

- Churn Prediction
- Churn Probability

### Live AWS Dashboard

**http://3.106.209.236:8501**

---

## Testing

Pytest is used to test the application.

The tests cover:

- Dataset loading
- Data cleaning
- Train/test preparation
- Preprocessing pipeline
- API root endpoint
- API health endpoint
- Model information endpoint
- Prediction endpoint
- Invalid API input validation

### Run Tests

```bash
python -m pytest -v
```

---

## Docker

The application uses separate Docker containers for the backend and frontend.

### Backend

```text
FastAPI
Port: 8000
```

### Frontend

```text
Streamlit
Port: 8501
```

### Start Application

```bash
docker compose up -d --build
```

### Check Containers

```bash
docker compose ps
```

### Stop Application

```bash
docker compose down
```

### Local URLs

Streamlit:

```text
http://localhost:8501
```

FastAPI:

```text
http://localhost:8000
```

Swagger:

```text
http://localhost:8000/docs
```

---

## GitHub Actions CI

GitHub Actions is used for continuous integration.

The CI workflow performs the following steps:

1. Checkout the repository
2. Set up Python 3.11
3. Install dependencies
4. Verify the dataset
5. Run preprocessing
6. Train machine learning models
7. Run Pytest
8. Build the backend Docker image
9. Build the frontend Docker image

This ensures that changes pushed to the repository are automatically tested.

---

## AWS Deployment

The application is deployed on an AWS EC2 instance using Docker Compose.

### AWS Architecture

```text
                 AWS EC2
                    |
              Docker Compose
                    |
          +---------+---------+
          |                   |
          v                   v
     FastAPI API         Streamlit
      Port 8000          Port 8501
          |                   |
          +---------+---------+
                    |
                    v
              Churn Model
```

### Deployment Commands

Clone the repository:

```bash
git clone https://github.com/Anand-Reddy-Burra/customer-churn-mlops.git
cd customer-churn-mlops
```

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install required dependencies:

```bash
pip install -r requirements-docker.txt
```

Build and start the application:

```bash
docker compose up -d --build
```

Check running containers:

```bash
docker compose ps
```

Check API health:

```bash
curl http://localhost:8000/health
```

---

## Live Deployment

### Streamlit Dashboard

**http://3.106.209.236:8501**

### FastAPI

The FastAPI service runs internally on port `8000`.

### Health Check

```text
http://localhost:8000/health
```

---

## GitHub Repository

**Repository:**

https://github.com/Anand-Reddy-Burra/customer-churn-mlops

---

## Key Project Outcomes

The project successfully demonstrates an end-to-end MLOps workflow for customer churn prediction.

### Implemented Components

- Data preprocessing
- Feature engineering
- Exploratory data analysis
- Machine learning model training
- Model evaluation
- MLflow experiment tracking
- DVC pipeline management
- FastAPI prediction API
- Streamlit dashboard
- Pytest automated testing
- Docker containerization
- Docker Compose
- GitHub Actions CI
- AWS EC2 deployment

### Final Model

```text
Decision Tree
F1 Score: 0.5989
```

### Deployment Status

```text
AWS EC2 Deployment: Completed
Docker Compose: Running
FastAPI: Running
Streamlit: Running
Dashboard: Publicly Accessible
```

---

## How to Run the Complete Project

### Step 1 — Clone Repository

```bash
git clone https://github.com/Anand-Reddy-Burra/customer-churn-mlops.git
cd customer-churn-mlops
```

### Step 2 — Create Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Run Preprocessing

```bash
python -m src.data.preprocess
```

### Step 5 — Train Models

```bash
python -m src.models.train
```

### Step 6 — Run Tests

```bash
python -m pytest -v
```

### Step 7 — Start Docker Application

```bash
docker compose up -d --build
```

### Step 8 — Open Dashboard

```text
http://localhost:8501
```

---

## Author

**Anand Reddy Burra**

**B.Tech – Computer Science and Engineering**  
**Vellore Institute of Technology (VIT-AP)**

### GitHub

https://github.com/Anand-Reddy-Burra

### LinkedIn

https://www.linkedin.com/in/anandreddyburra

### Portfolio

https://anandreddyburra.vercel.app/

---

## License

This project was developed as an academic/internship MLOps project for learning and demonstration purposes.
