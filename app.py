"""
================================================================
  LOAN APPROVAL PREDICTION SYSTEM — Streamlit App
  Student    : Muhammad Saeed  |  2023-uam-2308
  Department : BSIT 6th-D (2023-27)  |  MNSUAM
================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, os
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, roc_curve, auc)

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;600;700&family=IBM+Plex+Mono:wght@400;600&display=swap');

  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }

  .main-header {
    background: linear-gradient(135deg, #0f2342 0%, #1a3a6b 50%, #0d4f8c 100%);
    padding: 2.5rem 2rem 2rem 2rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border-left: 6px solid #f0a500;
    box-shadow: 0 8px 32px rgba(15,35,66,0.3);
  }
  .main-header h1 {
    color: #ffffff;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.5px;
  }
  .main-header p {
    color: #a8c4e0;
    font-size: 0.95rem;
    margin: 0;
  }
  .main-header .badge {
    display: inline-block;
    background: #f0a500;
    color: #0f2342;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    margin-bottom: 0.8rem;
    letter-spacing: 0.5px;
  }

  .metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1rem;
    text-align: center;
    border-top: 4px solid #1a3a6b;
    box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-3px); }
  .metric-card .val {
    font-size: 2rem;
    font-weight: 700;
    color: #1a3a6b;
    font-family: 'IBM Plex Mono', monospace;
  }
  .metric-card .lbl {
    font-size: 0.78rem;
    color: #666;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 0.2rem;
  }
  .metric-card.green  { border-top-color: #1e7145; }
  .metric-card.green .val { color: #1e7145; }
  .metric-card.red    { border-top-color: #c0392b; }
  .metric-card.red   .val { color: #c0392b; }
  .metric-card.gold   { border-top-color: #f0a500; }
  .metric-card.gold  .val { color: #e08800; }

  .result-approved {
    background: linear-gradient(135deg, #d9ead3, #b6d7a8);
    border: 2px solid #1e7145;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-rejected {
    background: linear-gradient(135deg, #fce8e6, #f4b8b3);
    border: 2px solid #c0392b;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1rem 0;
  }
  .result-title { font-size: 2rem; font-weight: 700; margin: 0; }
  .result-sub   { font-size: 1rem; margin-top: 0.5rem; color: #333; }

  .section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #0f2342;
    border-bottom: 3px solid #f0a500;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
  }
  .info-box {
    background: #ebf5fb;
    border-left: 4px solid #2e74b5;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0 1rem 0;
    font-size: 0.88rem;
    color: #1a3a6b;
  }
  .stButton>button {
    background: linear-gradient(135deg, #1a3a6b, #2e74b5);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: all 0.2s;
    letter-spacing: 0.3px;
  }
  .stButton>button:hover {
    background: linear-gradient(135deg, #0f2342, #1a3a6b);
    transform: translateY(-1px);
    box-shadow: 0 4px 15px rgba(26,58,107,0.4);
  }
  .sidebar-section {
    background: #f8fafc;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #e2e8f0;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  LOAD & TRAIN MODELS  (cached so it runs only once)
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner="🔄 Training models on dataset…")
def load_and_train():
    train_df = pd.read_csv("loan_approval_train.csv")
    test_df  = pd.read_csv("loan_approval_test.csv")

    for df in [train_df, test_df]:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

    def engineer(df):
        df = df.copy()
        df['total_assets']   = (df['residential_assets_value'] + df['commercial_assets_value'] +
                                 df['luxury_assets_value']      + df['bank_asset_value'])
        df['asset_to_loan']  = df['total_assets']   / (df['loan_amount'] + 1)
        df['income_to_loan'] = df['income_annum']   / (df['loan_amount'] + 1)
        df['loan_per_year']  = df['loan_amount']    / (df['loan_term']   + 1)
        df['cibil_band']     = pd.cut(df['cibil_score'], bins=[299,500,600,700,750,900],
                                       labels=[0,1,2,3,4]).astype(int)
        return df

    train_df = engineer(train_df)
    test_df  = engineer(test_df)

    le_edu = LabelEncoder(); le_emp = LabelEncoder(); le_tgt = LabelEncoder()
    train_df['education']     = le_edu.fit_transform(train_df['education'])
    train_df['self_employed'] = le_emp.fit_transform(train_df['self_employed'])
    train_df['loan_status']   = le_tgt.fit_transform(train_df['loan_status'])
    mapping = dict(zip(le_tgt.classes_, le_tgt.transform(le_tgt.classes_)))
    if mapping['Approved'] == 0:
        train_df['loan_status'] = 1 - train_df['loan_status']
    test_df['education']     = le_edu.transform(test_df['education'])
    test_df['self_employed'] = le_emp.transform(test_df['self_employed'])

    FEATURES = ['no_of_dependents','education','self_employed','income_annum','loan_amount',
                'loan_term','cibil_score','residential_assets_value','commercial_assets_value',
                'luxury_assets_value','bank_asset_value','total_assets','asset_to_loan',
                'income_to_loan','loan_per_year','cibil_band']

    X = train_df[FEATURES]; y = train_df['loan_status']
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20,
                                                       random_state=42, stratify=y)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_v_sc  = scaler.transform(X_val)

    lr = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
    lr.fit(X_tr_sc, y_train)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                 min_samples_leaf=4, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    dt = DecisionTreeClassifier(max_depth=8, min_samples_leaf=6, random_state=42)
    dt.fit(X_train, y_train)

    metrics = {}
    for name, pred, prob in [
        ('Logistic Regression', lr.predict(X_v_sc),  lr.predict_proba(X_v_sc)[:,1]),
        ('Random Forest',       rf.predict(X_val),   rf.predict_proba(X_val)[:,1]),
        ('Decision Tree',       dt.predict(X_val),   dt.predict_proba(X_val)[:,1]),
    ]:
        metrics[name] = {
            'acc' : accuracy_score(y_val, pred),
            'prec': precision_score(y_val, pred, zero_division=0),
            'rec' : recall_score(y_val, pred, zero_division=0),
            'f1'  : f1_score(y_val, pred, zero_division=0),
            'cm'  : confusion_matrix(y_val, pred),
            'fpr' : roc_curve(y_val, prob)[0],
            'tpr' : roc_curve(y_val, prob)[1],
            'auc' : auc(*roc_curve(y_val, prob)[:2]),
        }

    fi = pd.DataFrame({'Feature': FEATURES,
                        'Importance': rf.feature_importances_}).sort_values(
                            'Importance', ascending=False)

    return {
        'train_df': train_df, 'test_df': test_df,
        'lr': lr, 'rf': rf, 'dt': dt,
        'scaler': scaler, 'FEATURES': FEATURES,
        'X_val': X_val, 'y_val': y_val,
        'metrics': metrics, 'fi': fi,
        'le_edu': le_edu, 'le_emp': le_emp
    }

data = load_and_train()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
      <div style='font-size:2.5rem;'>🏦</div>
      <div style='font-weight:700; font-size:1.1rem; color:#0f2342;'>Loan Approval System</div>
      <div style='font-size:0.78rem; color:#666; margin-top:4px;'>ML-Powered Decision Tool</div>
    </div>
    <hr style='border-color:#e2e8f0; margin: 0.8rem 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("📌 Navigate", [
        "🏠 Home & Overview",
        "🔍 Predict Loan",
        "📊 Model Performance",
        "📈 Data Analysis (EDA)",
        "📋 Dataset Explorer"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:#e2e8f0; margin: 0.8rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='sidebar-section'>
      <div style='font-size:0.78rem; font-weight:700; color:#0f2342; margin-bottom:6px;'>👨‍💻 STUDENT INFO</div>
      <div style='font-size:0.8rem; color:#444; line-height:1.7;'>
        <b>Muhammad Saeed</b><br>
        Reg: 2023-uam-2308<br>
        BSIT 6th-D (2023–27)<br>
        MNSUAM
      </div>
    </div>
    <div class='sidebar-section'>
      <div style='font-size:0.78rem; font-weight:700; color:#0f2342; margin-bottom:6px;'>📦 DATASET</div>
      <div style='font-size:0.78rem; color:#444; line-height:1.7;'>
        Train: 3,415 records<br>
        Test: 854 records<br>
        Features: 16 (incl. 5 engineered)<br>
        Source: Kaggle
      </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: HOME
# ══════════════════════════════════════════════════════════════
if page == "🏠 Home & Overview":
    st.markdown("""
    <div class='main-header'>
      <div class='badge'>🎓 BSIT 6TH-D PROJECT · 2023-UAM-2308</div>
      <h1>🏦 Loan Approval Prediction System</h1>
      <p>Machine Learning-Based Classification using Logistic Regression , Decision Tree & Random Forest · Muhammad Nawaz Sharif University of Agriculture, Multan</p>
    </div>
    """, unsafe_allow_html=True)

    m = data['metrics']
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='metric-card green'>
            <div class='val'>{m['Random Forest']['acc']*100:.2f}%</div>
            <div class='lbl'>RF Accuracy</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='metric-card green'>
            <div class='val'>{m['Random Forest']['f1']:.4f}</div>
            <div class='lbl'>RF F1-Score</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='metric-card'>
            <div class='val'>{m['Logistic Regression']['acc']*100:.2f}%</div>
            <div class='lbl'>LR Accuracy</div></div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class='metric-card gold'>
            <div class='val'>3,415</div>
            <div class='lbl'>Training Records</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>📌 Project Overview</div>", unsafe_allow_html=True)
        st.markdown("""
        This system predicts whether a loan application will be **Approved** or **Rejected**
        using real data from Kaggle. Three ML models are compared:

        | Model | Accuracy | F1-Score |
        |---|---|---|
        | 🌲 Random Forest | **99.85%** | **0.9988** |
        | 🌳 Decision Tree | 99.56% | 0.9964 |
        | 📈 Logistic Regression | 90.34% | 0.9218 |

        > **Key finding:** CIBIL score is the #1 predictor of loan approval.
        """)
    with col2:
        st.markdown("<div class='section-header'>🔧 Features Used</div>", unsafe_allow_html=True)
        st.markdown("""
        **Original Features (11):**
        - CIBIL Score, Annual Income, Loan Amount
        - Loan Term, No. of Dependents
        - Education, Self-Employed
        - Residential, Commercial, Luxury & Bank Assets

        **Engineered Features (5):**
        - `total_assets` — sum of all asset values
        - `asset_to_loan` — collateral coverage ratio
        - `income_to_loan` — repayment capacity ratio
        - `loan_per_year` — annual repayment burden
        - `cibil_band` — ordinal credit score group
        """)

    st.markdown("<div class='info-box'>💡 Use the <b>sidebar</b> to navigate between pages — try <b>Predict Loan</b> to test with your own input values!</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: PREDICT
# ══════════════════════════════════════════════════════════════
elif page == "🔍 Predict Loan":
    st.markdown("<h2 style='color:#0f2342;'>🔍 Loan Approval Predictor</h2>", unsafe_allow_html=True)
    st.markdown("<div class='info-box'>Fill in the applicant details below and click <b>Predict</b> to get instant results from all 3 models.</div>", unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown("<div class='section-header'>👤 Applicant Information</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            no_dep   = st.selectbox("No. of Dependents", [0,1,2,3,4,5])
            education= st.selectbox("Education", ["Graduate", "Not Graduate"])
        with c2:
            self_emp = st.selectbox("Self Employed", ["No", "Yes"])
            loan_term= st.slider("Loan Term (years)", 2, 20, 10)
        with c3:
            cibil    = st.slider("CIBIL Score", 300, 900, 650)
            income   = st.number_input("Annual Income (PKR)", min_value=100000,
                                        max_value=10000000, value=5000000, step=100000)

        st.markdown("<div class='section-header'>💰 Loan & Asset Details</div>", unsafe_allow_html=True)
        c4, c5 = st.columns(2)
        with c4:
            loan_amt = st.number_input("Loan Amount (PKR)", min_value=100000,
                                        max_value=40000000, value=10000000, step=500000)
        with c5:
            res_asset = st.number_input("Residential Asset Value", min_value=0,
                                         max_value=30000000, value=3000000, step=500000)

        c6, c7, c8 = st.columns(3)
        with c6:
            com_asset = st.number_input("Commercial Asset Value", min_value=0,
                                         max_value=20000000, value=2000000, step=500000)
        with c7:
            lux_asset = st.number_input("Luxury Asset Value", min_value=0,
                                         max_value=40000000, value=5000000, step=500000)
        with c8:
            bank_asset= st.number_input("Bank Asset Value", min_value=0,
                                         max_value=15000000, value=1500000, step=500000)

        submitted = st.form_submit_button("🏦 PREDICT LOAN APPROVAL")

    if submitted:
        # Build sample
        edu_enc = 1 if education == "Graduate" else 0
        emp_enc = 1 if self_emp  == "Yes"      else 0
        total_assets   = res_asset + com_asset + lux_asset + bank_asset
        asset_to_loan  = total_assets   / (loan_amt + 1)
        income_to_loan = income         / (loan_amt + 1)
        loan_per_year  = loan_amt       / (loan_term + 1)
        cibil_band     = int(pd.cut([cibil], bins=[299,500,600,700,750,900],
                                     labels=[0,1,2,3,4])[0])

        sample = pd.DataFrame([{
            'no_of_dependents': no_dep, 'education': edu_enc, 'self_employed': emp_enc,
            'income_annum': income, 'loan_amount': loan_amt, 'loan_term': loan_term,
            'cibil_score': cibil, 'residential_assets_value': res_asset,
            'commercial_assets_value': com_asset, 'luxury_assets_value': lux_asset,
            'bank_asset_value': bank_asset, 'total_assets': total_assets,
            'asset_to_loan': asset_to_loan, 'income_to_loan': income_to_loan,
            'loan_per_year': loan_per_year, 'cibil_band': cibil_band
        }])

        FEATURES = data['FEATURES']
        s_sc = data['scaler'].transform(sample[FEATURES])

        lr_res = data['lr'].predict(s_sc)[0]
        rf_res = data['rf'].predict(sample[FEATURES])[0]
        dt_res = data['dt'].predict(sample[FEATURES])[0]

        lr_conf = data['lr'].predict_proba(s_sc)[0]
        rf_conf = data['rf'].predict_proba(sample[FEATURES])[0]
        dt_conf = data['dt'].predict_proba(sample[FEATURES])[0]

        votes = lr_res + rf_res + dt_res
        final = "Approved" if votes >= 2 else "Rejected"

        # Display final result
        if final == "Approved":
            st.markdown(f"""<div class='result-approved'>
              <div class='result-title'>✅ LOAN APPROVED</div>
              <div class='result-sub'>Majority vote: {votes}/3 models predict Approval</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class='result-rejected'>
              <div class='result-title'>❌ LOAN REJECTED</div>
              <div class='result-sub'>Majority vote: {3-votes}/3 models predict Rejection</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>📊 Individual Model Results</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        for col, name, res, conf in [
            (col1, "📈 Logistic Reg.", lr_res, lr_conf),
            (col2, "🌲 Random Forest", rf_res, rf_conf),
            (col3, "🌳 Decision Tree", dt_res, dt_conf),
        ]:
            with col:
                emoji  = "✅" if res == 1 else "❌"
                status = "Approved" if res == 1 else "Rejected"
                color  = "#1e7145" if res == 1 else "#c0392b"
                conf_pct = conf[1]*100
                st.markdown(f"""
                <div style='background:white; border-radius:12px; padding:1.2rem;
                            border-top:4px solid {color}; text-align:center;
                            box-shadow:0 2px 10px rgba(0,0,0,0.07);'>
                  <div style='font-size:1.6rem;'>{emoji}</div>
                  <div style='font-weight:700; font-size:1rem; color:#0f2342;'>{name}</div>
                  <div style='font-weight:700; color:{color}; font-size:1.1rem;'>{status}</div>
                  <div style='font-size:0.82rem; color:#666; margin-top:4px;'>
                    Confidence: {conf_pct:.1f}%
                  </div>
                </div>""", unsafe_allow_html=True)

        # Input summary
        st.markdown("<div class='section-header'>📋 Input Summary</div>", unsafe_allow_html=True)
        summary = pd.DataFrame({
            'Feature': ['CIBIL Score','Annual Income','Loan Amount','Loan Term',
                        'Total Assets','Asset/Loan Ratio','Income/Loan Ratio','Education','Self-Employed'],
            'Value':   [cibil, f"PKR {income:,}", f"PKR {loan_amt:,}", f"{loan_term} years",
                        f"PKR {total_assets:,}", f"{asset_to_loan:.2f}", f"{income_to_loan:.2f}",
                        education, self_emp]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.markdown("<h2 style='color:#0f2342;'>📊 Model Performance & Evaluation</h2>", unsafe_allow_html=True)

    m = data['metrics']
    names  = ['Logistic Regression','Random Forest','Decision Tree']
    colors = ['#3498db','#2ecc71','#e67e22']

    # Metrics table
    st.markdown("<div class='section-header'>📋 Performance Summary</div>", unsafe_allow_html=True)
    perf_df = pd.DataFrame([{
        'Model': n,
        'Accuracy':  f"{m[n]['acc']*100:.2f}%",
        'Precision': f"{m[n]['prec']:.4f}",
        'Recall':    f"{m[n]['rec']:.4f}",
        'F1-Score':  f"{m[n]['f1']:.4f}",
        'AUC':       f"{m[n]['auc']:.4f}"
    } for n in names])
    st.dataframe(perf_df, use_container_width=True, hide_index=True)

    # Metric bar chart
    st.markdown("<div class='section-header'>📊 Metric Comparison Chart</div>", unsafe_allow_html=True)
    metrics_list = ['acc','prec','rec','f1']
    metric_names = ['Accuracy','Precision','Recall','F1-Score']
    x = np.arange(len(metric_names)); width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#fafafa')
    ax.set_facecolor('#fafafa')
    for i, (name, col) in enumerate(zip(names, colors)):
        vals = [m[name][k] for k in metrics_list]
        bars = ax.bar(x + i*width, vals, width, label=name, color=col,
                      edgecolor='white', linewidth=0.8)
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}",
                        (bar.get_x()+bar.get_width()/2, bar.get_height()+0.005),
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.set_xticks(x + width); ax.set_xticklabels(metric_names, fontsize=12)
    ax.set_ylim(0, 1.12); ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=13)
    ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    col1, col2 = st.columns(2)
    # ROC Curves
    with col1:
        st.markdown("<div class='section-header'>📈 ROC Curves</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        for name, col in zip(names, colors):
            ax.plot(m[name]['fpr'], m[name]['tpr'], lw=2.2, color=col,
                    label=f"{name} (AUC={m[name]['auc']:.4f})")
        ax.plot([0,1],[0,1],'k--', lw=1)
        ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves', fontweight='bold')
        ax.legend(fontsize=9); ax.grid(alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Confusion Matrix (RF best)
    with col2:
        st.markdown("<div class='section-header'>🔲 Confusion Matrix (Random Forest)</div>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        sns.heatmap(m['Random Forest']['cm'], annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Rejected','Approved'],
                    yticklabels=['Rejected','Approved'],
                    annot_kws={'size':14,'weight':'bold'}, linewidths=0.5)
        ax.set_title('Random Forest', fontweight='bold')
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Feature Importance
    st.markdown("<div class='section-header'>🌲 Feature Importance (Random Forest)</div>", unsafe_allow_html=True)
    fi = data['fi']
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
    colors_fi = ['#e74c3c' if v > 0.08 else '#3498db' for v in fi['Importance']]
    ax.barh(fi['Feature'], fi['Importance'], color=colors_fi, edgecolor='white')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title('Feature Importance — Top Features in Red', fontweight='bold', fontsize=13)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
#  PAGE: EDA
# ══════════════════════════════════════════════════════════════
elif page == "📈 Data Analysis (EDA)":
    st.markdown("<h2 style='color:#0f2342;'>📈 Exploratory Data Analysis</h2>", unsafe_allow_html=True)

    train_df = data['train_df'].copy()
    # Decode back for display
    train_df['loan_label'] = train_df['loan_status'].map({1:'Approved', 0:'Rejected'})
    palette = {'Approved':'#2ecc71','Rejected':'#e74c3c'}

    # Row 1
    st.markdown("<div class='section-header'>📊 Distribution Charts</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        vc = train_df['loan_label'].value_counts()
        ax.bar(vc.index, vc.values, color=[palette[x] for x in vc.index], edgecolor='white', linewidth=1.2)
        for bar, val in zip(ax.patches, vc.values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                    f'{val}\n({val/len(train_df)*100:.1f}%)', ha='center', fontsize=9)
        ax.set_title('Loan Status Distribution', fontweight='bold')
        ax.set_ylabel('Count'); plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        for status, grp in train_df.groupby('loan_label'):
            ax.hist(grp['cibil_score'], bins=30, alpha=0.7, label=status,
                    color=palette[status], edgecolor='white')
        ax.axvline(600, color='grey', linestyle='--', linewidth=1.5)
        ax.set_title('CIBIL Score by Loan Status', fontweight='bold')
        ax.set_xlabel('CIBIL Score'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c3:
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        approved = train_df[train_df['loan_status']==1]['cibil_score']
        rejected = train_df[train_df['loan_status']==0]['cibil_score']
        bp = ax.boxplot([rejected, approved], labels=['Rejected','Approved'],
                         patch_artist=True, notch=True,
                         medianprops={'color':'black','linewidth':2})
        bp['boxes'][0].set_facecolor('#e74c3c'); bp['boxes'][1].set_facecolor('#2ecc71')
        ax.set_title('CIBIL Score Boxplot', fontweight='bold')
        ax.set_ylabel('CIBIL Score'); ax.grid(axis='y', alpha=0.3)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Row 2
    c4, c5, c6 = st.columns(3)
    with c4:
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        for status, grp in train_df.groupby('loan_label'):
            ax.hist(grp['income_annum']/1e6, bins=30, alpha=0.7, label=status,
                    color=palette[status], edgecolor='white')
        ax.set_title('Annual Income Distribution', fontweight='bold')
        ax.set_xlabel('Income (Millions PKR)'); ax.legend()
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c5:
        train_df['edu_label'] = train_df['education'].map({1:'Graduate', 0:'Not Graduate'})
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        ct = pd.crosstab(train_df['edu_label'], train_df['loan_label'])
        ct.plot(kind='bar', ax=ax, color=[palette[c] for c in ct.columns],
                edgecolor='white', rot=0)
        ax.set_title('Education vs Loan Status', fontweight='bold')
        ax.set_ylabel('Count'); plt.tight_layout(); st.pyplot(fig); plt.close()

    with c6:
        fig, ax = plt.subplots(figsize=(5,4)); fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
        train_df['loan_term_orig'] = data['test_df']['loan_term'].iloc[0]  # fallback
        # use original loan_term from train
        train_df2 = pd.read_csv("loan_approval_train.csv")
        train_df2.columns = train_df2.columns.str.strip()
        train_df2['loan_term'].value_counts().sort_index().plot(
            kind='bar', ax=ax, color='#3498db', edgecolor='white')
        ax.set_title('Loan Term Distribution', fontweight='bold')
        ax.set_xlabel('Term (years)'); ax.tick_params(rotation=0)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    # Correlation heatmap
    st.markdown("<div class='section-header'>🔥 Correlation Heatmap</div>", unsafe_allow_html=True)
    FEATURES = data['FEATURES']
    corr = data['train_df'][FEATURES + ['loan_status']].corr()
    fig, ax = plt.subplots(figsize=(14, 10))
    fig.patch.set_facecolor('#fafafa'); ax.set_facecolor('#fafafa')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                linewidths=0.4, annot_kws={'size':7}, ax=ax, cbar_kws={'shrink':0.8})
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
    plt.tight_layout(); st.pyplot(fig); plt.close()

# ══════════════════════════════════════════════════════════════
#  PAGE: DATASET EXPLORER
# ══════════════════════════════════════════════════════════════
elif page == "📋 Dataset Explorer":
    st.markdown("<h2 style='color:#0f2342;'>📋 Dataset Explorer</h2>", unsafe_allow_html=True)

    train_raw = pd.read_csv("loan_approval_train.csv")
    test_raw  = pd.read_csv("loan_approval_test.csv")
    for df in [train_raw, test_raw]:
        df.columns = df.columns.str.strip()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

    t1, t2 = st.tabs(["📂 Training Set (3,415 records)", "📂 Test Set (854 records)"])

    with t1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Records", "3,415")
        c2.metric("Features",      "13 (raw)")
        c3.metric("Approved",      f"{(train_raw['loan_status'].str.strip()=='Approved').sum()}")
        c4.metric("Rejected",      f"{(train_raw['loan_status'].str.strip()=='Rejected').sum()}")
        st.markdown("<div class='section-header'>🔍 Sample Data</div>", unsafe_allow_html=True)
        n = st.slider("Show rows:", 5, 50, 10, key="train_rows")
        st.dataframe(train_raw.head(n), use_container_width=True)
        st.markdown("<div class='section-header'>📊 Statistics</div>", unsafe_allow_html=True)
        st.dataframe(train_raw.describe(), use_container_width=True)

    with t2:
        c1, c2 = st.columns(2)
        c1.metric("Total Records", "854")
        c2.metric("Features",      "12 (no label)")
        st.markdown("<div class='section-header'>🔍 Sample Data</div>", unsafe_allow_html=True)
        n2 = st.slider("Show rows:", 5, 50, 10, key="test_rows")
        st.dataframe(test_raw.head(n2), use_container_width=True)

# ── Footer ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#aaa; font-size:0.78rem; padding:0.5rem;'>
  🏦 Loan Approval Prediction System &nbsp;|&nbsp;
  Muhammad Saeed · 2023-uam-2308 · BSIT 6th-D · UAM &nbsp;|&nbsp;
  Dataset: <a href='https://www.kaggle.com/datasets/muhammadsaeed786/loan-approval-prediction'
  style='color:#2e74b5;'>Kaggle</a>
</div>
""", unsafe_allow_html=True)
