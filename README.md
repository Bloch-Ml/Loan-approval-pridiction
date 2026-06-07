# 🏦 Loan Approval Prediction System

> **Student:** Muhammad Saeed | **Reg No:** 2023-uam-2308  
> **Department:** BSIT 6th-D (2023–27) | **University:** UAM  
> **Dataset:** [Kaggle – muhammadsaeed786/loan-approval-prediction](https://www.kaggle.com/datasets/muhammadsaeed786/loan-approval-prediction)

---

## 📌 About

A machine learning web application that predicts whether a loan application will be **Approved** or **Rejected** using real banking data. Three models are trained and compared:

| Model | Accuracy | F1-Score |
|---|---|---|
| 🌲 Random Forest | **99.85%** | **0.9988** |
| 🌳 Decision Tree | 99.56% | 0.9964 |
| 📈 Logistic Regression | 90.34% | 0.9218 |

---

## 🚀 Run Locally

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-prediction.git
cd loan-approval-prediction
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Run the app
```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📁 Project Structure

```
loan-approval-prediction/
│
├── app.py                        # Main Streamlit application
├── loan_approval_prediction.py   # ML training script (Jupyter)
├── requirements.txt              # Python dependencies
├── README.md                     # This file
│
├── loan_approval_train.csv       # Training dataset (3,415 records)
└── loan_approval_test.csv        # Test dataset (854 records)
```

---

## 🔧 Features

- **5 App Pages:**
  - 🏠 Home & Overview — project summary and key metrics
  - 🔍 Predict Loan — enter applicant details and get instant prediction
  - 📊 Model Performance — accuracy, ROC curves, confusion matrices
  - 📈 Data Analysis — EDA charts and correlation heatmap
  - 📋 Dataset Explorer — browse raw data

- **16 Features Used** (11 original + 5 engineered):
  - CIBIL Score, Annual Income, Loan Amount, Loan Term
  - Residential, Commercial, Luxury & Bank Assets
  - Engineered: `total_assets`, `asset_to_loan`, `income_to_loan`, `loan_per_year`, `cibil_band`

---

## 📊 Dataset

- **Source:** Kaggle — [muhammadsaeed786/loan-approval-prediction](https://www.kaggle.com/datasets/muhammadsaeed786/loan-approval-prediction)
- **Training:** 3,415 records | **Test:** 854 records
- **Target:** `loan_status` → Approved / Rejected
- **Missing Values:** None

---

## 🛠️ Tech Stack

- Python 3.x
- Streamlit
- scikit-learn
- pandas, NumPy
- matplotlib, seaborn
