# 📊 HR Analytics & Churn Intelligence

An interactive HR Analytics dashboard built with **Python, Streamlit, Pandas, Plotly and Machine Learning** to analyze workforce trends, employee attrition and predict employee churn risk.

## 🚀 Live Demo

👉 [Open Live Dashboard](https://hr-analytics-churn-intelligence.streamlit.app/)

---

## 🎯 Project Overview

HR Analytics & Churn Intelligence is an end-to-end data analytics and machine learning application designed to help HR teams understand employee behavior and identify potential churn risks.

The dashboard combines:

- Workforce analytics
- Employee attrition analysis
- Department-level insights
- Employee churn prediction
- Risk classification
- Management recommendations
- Individual employee exploration

---

## 📌 Key Features

### 🏠 Executive Dashboard
- Total employee overview
- Employees who left and stayed
- Overall attrition rate
- Average salary
- Employee satisfaction
- Department-wise workforce distribution
- Department-wise attrition analysis
- Key HR insights
- Management action recommendations

### 🤖 Churn Prediction
Predict employee churn probability using employee-level information such as:

- Satisfaction
- Evaluation
- Number of projects
- Average monthly hours
- Years at company
- Work accident
- Promotion
- Department
- Salary

The application classifies employees into:

- 🔴 High Risk
- 🟠 Medium Risk
- 🟢 Low Risk

### 📊 HR Analytics
Provides detailed workforce and employee analytics to identify important HR patterns.

### 🎯 Risk Center
Helps HR teams focus on employees with higher predicted churn risk.

### 💡 Executive Insights
Provides data-driven recommendations based on:
- Attrition trends
- Employee satisfaction
- Workload
- Department performance
- ML churn risk

### 👤 Employee Explorer
Allows exploration of individual employee information and related HR indicators.

---

## 📈 Key Findings

Based on the current dataset:

| Metric | Value |
|---|---:|
| Total Employees | 14,999 |
| Employees Left | 3,571 |
| Attrition Rate | 23.81% |
| Average Satisfaction | 6.13/10 |
| Average Salary | ₹62,743 |
| Average Monthly Hours | 201 |

### Department Insights

- Highest Attrition Department: **HR — 29.09%**
- Lowest Attrition Department: **Management — 14.44%**
- High Workload (>250 hours/month) Attrition: **38.79%**
- Low Satisfaction (<4/10) Attrition: **53.72%**

---

## 🧠 Machine Learning

The project uses a trained machine learning model to estimate employee churn probability.

The prediction pipeline includes:

1. Employee data input
2. Feature preprocessing
3. Categorical encoding
4. Feature alignment with the trained model
5. Churn probability prediction
6. Risk-level classification
7. HR recommendations

---

## 🛠️ Tech Stack

### Programming
- Python

### Data Analytics
- Pandas
- NumPy

### Visualization
- Plotly

### Machine Learning
- Scikit-learn
- Joblib

### Application
- Streamlit

---

## 📂 Project Structure

```text
HR-Analytics-Churn-Intelligence/
│
├── app.py
├── Employee_HR.csv
├── employee_churn_model.pkl
├── requirements.txt
├── README.md
└── .gitignore
