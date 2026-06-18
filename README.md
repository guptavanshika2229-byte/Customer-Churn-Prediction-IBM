# 📊 IBM Telco Customer Churn Prediction

An end-to-end data science project that predicts customer churn 
for a telecom company using machine learning, with a live 
interactive web app and a business intelligence dashboard.

🔗 **Live App:** [ibm-churn-predictor.streamlit.app](https://ibm-churn-predictor.streamlit.app)

---

## 📌 Problem Statement

Customer churn is one of the biggest challenges in the telecom 
industry. Losing a customer is far more expensive than retaining 
one. This project builds a machine learning model that predicts 
which customers are likely to churn, enabling the retention team 
to take proactive action.

---

## 🗂️ Project Structure

customer-churn-prediction/

├── 02_eda.ipynb                  # Exploratory Data Analysis
├── 03_feature_engineering.ipynb  # Preprocessing & Feature Engineering
├── 04_modeling.ipynb             # Model Training & Evaluation
├── streamlit_app.py              # Live Web App
├── churn_model.pkl               # Trained Model
├── scaler.pkl                    # Feature Scaler
├── model_columns.pkl             # Model Feature Columns
└── requirements.txt              # Dependencies

---

## 🔄 Project Pipeline

**Data Source → SQL → EDA → Feature Engineering → ML Modeling → Power BI Dashboard → Streamlit Deployment**

---

## 📊 Dataset

- **Source:** IBM Telco Customer Churn Dataset (Kaggle)
- **Size:** 7,043 customers × 33 features
- **Target:** Churn Label (Yes/No)
- **Features:** Demographics, account info, services subscribed, 
  contract type, payment method

---

## 🛠️ Tools & Technologies

| Area | Tools |
|------|-------|
| Data Storage | SQLite, SQL |
| Data Analysis | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn, Power BI |
| Machine Learning | Scikit-learn, XGBoost |
| Statistics | SciPy (t-test, chi-square) |
| Deployment | Streamlit, Streamlit Community Cloud |
| Version Control | Git, GitHub |

---

## 🔍 Key Findings from EDA

- Overall churn rate is **26.5%**
- Customers on **month-to-month contracts** churn at ~42% vs 
  only ~3% for two-year contracts
- **Fiber optic** internet service customers have the highest 
  churn rate (~41%)
- **Electronic check** payment method has significantly higher 
  churn than automatic payment methods
- A t-test confirmed tenure is significantly lower for churned 
  customers (t = -31.58, p < 0.001)
- Chi-square test confirmed contract type significantly affects 
  churn (p < 0.001)

---

## 🤖 ML Models Compared

| Model | Accuracy | Recall (Churn) | ROC-AUC |
|-------|----------|----------------|---------|
| Logistic Regression | 74% | **78%** | **0.848** |
| Random Forest | 79% | 49% | 0.839 |
| XGBoost | 78% | 64% | 0.826 |

**Winner: Logistic Regression** — highest ROC-AUC and recall 
for the churn class. For a retention use case, catching 
churners (recall) matters more than raw accuracy.

---

## 🔑 Top Factors Driving Churn

**Increasing churn risk:**
- High total charges
- Fiber optic internet service
- Electronic check payment
- Month-to-month contract
- Paperless billing

**Decreasing churn risk:**
- Long tenure
- Two-year contract
- Having dependents
- Online security subscription
- Tech support subscription

---

## 🚀 How to Run Locally

1. Clone the repository:
```bash
git clone https://github.com/guptavanshika2229-byte/Customer-Churn-Prediction-IBM.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run streamlit_app.py
```

---

## 📱 Live Demo

🔗 [ibm-churn-predictor.streamlit.app](https://ibm-churn-predictor.streamlit.app)

Enter any customer's details and get an instant churn 
probability score with risk level (Low/Medium/High).

---

## 👤 Author

**Vanshika Gupta**  
Aspiring Data Scientist  
📧 guptavanshika2229@gmail.com  
🔗 [GitHub](https://github.com/guptavanshika2229-byte)

---

## 📄 License

This project is open source and available under the 
[MIT License](LICENSE).
