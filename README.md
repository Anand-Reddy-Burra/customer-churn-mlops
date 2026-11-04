# Customer Churn Prediction – End-to-End MLOps

## Project Overview

This project implements an end-to-end Machine Learning and MLOps pipeline for predicting customer churn in a telecommunications company.

The system takes customer information such as tenure, contract type, internet service, monthly charges, payment method, and service usage, and predicts whether the customer is likely to churn.

The project covers the complete workflow:

Dataset → Data Preprocessing → Feature Engineering → Model Training → Model Evaluation → MLflow → DVC → FastAPI → Streamlit → Pytest → Docker → GitHub Actions → AWS EC2

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

Dataset: IBM Telco Customer Churn Dataset

The dataset contains approximately 7,000 customer records and includes customer demographics, services, contract information, charges, and churn status.

Target variable:

`Churn`

Values:

- Yes
- No

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
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── predict.py
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
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── Dockerfile.frontend
├── docker-compose.yml
├── dvc.yaml
├── dvc.lock
├── params.yaml
├── requirements.txt
└── README.md