"""
================================================================
  LOAN APPROVAL PREDICTION SYSTEM
  Student    : Muhammad Saeed
  Reg No     : 2023-uam-2308
  Department : BSIT 6th-D  (2023-27)
  University : University of Agriculture, Mirpur (UAM)
  Dataset    : https://www.kaggle.com/datasets/muhammadsaeed786/
               loan-approval-prediction
================================================================
Models Compared:
  1. Logistic Regression
  2. Random Forest Classifier
  3. Decision Tree Classifier  (baseline)
================================================================
"""

# ─────────────────────────────────────────────────────────────
# 0. IMPORTS
# ─────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os, warnings
warnings.filterwarnings('ignore')
np.random.seed(42)

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder, StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.tree            import DecisionTreeClassifier
from sklearn.metrics         import (accuracy_score, precision_score, recall_score,
                                     f1_score, confusion_matrix, classification_report,
                                     roc_curve, auc)

os.makedirs("outputs", exist_ok=True)

# ─────────────────────────────────────────────────────────────
# 1. LOAD REAL DATASET
# ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  LOAN APPROVAL PREDICTION SYSTEM")
print("  Student : Muhammad Saeed  |  2023-uam-2308")
print("  Dataset : Kaggle – muhammadsaeed786/loan-approval-prediction")
print("=" * 65)

TRAIN_PATH = "/mnt/user-data/uploads/loan_approval_train.csv"
TEST_PATH  = "/mnt/user-data/uploads/loan_approval_test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)

# Strip leading/trailing whitespace from string columns
train_df.columns = train_df.columns.str.strip()
test_df.columns  = test_df.columns.str.strip()
for col in train_df.select_dtypes(include='object').columns:
    train_df[col] = train_df[col].str.strip()
for col in test_df.select_dtypes(include='object').columns:
    test_df[col]  = test_df[col].str.strip()

print(f"\n[1] Dataset Loaded")
print(f"    Training samples : {train_df.shape[0]}  |  Features : {train_df.shape[1]-2}")
print(f"    Testing  samples : {test_df.shape[0]}   |  Features : {test_df.shape[1]-1}")
print(f"\n    Class distribution (Train):")
vc = train_df['loan_status'].value_counts()
for k,v in vc.items():
    print(f"      {k}: {v}  ({v/len(train_df)*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 2. EDA  – PLOTS
# ─────────────────────────────────────────────────────────────
print("\n[2] Exploratory Data Analysis …")

palette = {'Approved': '#2ecc71', 'Rejected': '#e74c3c'}

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Loan Approval Prediction – Exploratory Data Analysis',
             fontsize=15, fontweight='bold', y=1.01)

# (a) Loan Status Distribution
status_counts = train_df['loan_status'].value_counts()
axes[0,0].bar(status_counts.index, status_counts.values,
              color=[palette[x] for x in status_counts.index], edgecolor='white', linewidth=1.2)
axes[0,0].set_title('Loan Status Distribution', fontweight='bold')
axes[0,0].set_ylabel('Number of Applicants')
for bar, val in zip(axes[0,0].patches, status_counts.values):
    axes[0,0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+10,
                   f'{val}\n({val/len(train_df)*100:.1f}%)', ha='center', fontsize=10)

# (b) CIBIL Score by Loan Status
for status, grp in train_df.groupby('loan_status'):
    axes[0,1].hist(grp['cibil_score'], bins=30, alpha=0.7,
                   label=status, color=palette[status], edgecolor='white')
axes[0,1].set_title('CIBIL Score by Loan Status', fontweight='bold')
axes[0,1].set_xlabel('CIBIL Score')
axes[0,1].set_ylabel('Count')
axes[0,1].legend()
axes[0,1].axvline(600, color='grey', linestyle='--', linewidth=1.2, label='Score=600')

# (c) Annual Income by Loan Status
for status, grp in train_df.groupby('loan_status'):
    axes[0,2].hist(grp['income_annum']/1e6, bins=30, alpha=0.7,
                   label=status, color=palette[status], edgecolor='white')
axes[0,2].set_title('Annual Income by Loan Status', fontweight='bold')
axes[0,2].set_xlabel('Annual Income (Millions)')
axes[0,2].set_ylabel('Count')
axes[0,2].legend()

# (d) Education vs Loan Status
ct = pd.crosstab(train_df['education'], train_df['loan_status'])
ct.plot(kind='bar', ax=axes[1,0], color=[palette[c] for c in ct.columns],
        edgecolor='white', rot=0)
axes[1,0].set_title('Education vs Loan Status', fontweight='bold')
axes[1,0].set_ylabel('Count')

# (e) Self Employed vs Loan Status
ct2 = pd.crosstab(train_df['self_employed'], train_df['loan_status'])
ct2.plot(kind='bar', ax=axes[1,1], color=[palette[c] for c in ct2.columns],
         edgecolor='white', rot=0)
axes[1,1].set_title('Self-Employed vs Loan Status', fontweight='bold')
axes[1,1].set_ylabel('Count')

# (f) Loan Term Distribution
train_df['loan_term'].value_counts().sort_index().plot(
    kind='bar', ax=axes[1,2], color='#3498db', edgecolor='white')
axes[1,2].set_title('Loan Term Distribution (Years)', fontweight='bold')
axes[1,2].set_xlabel('Loan Term (years)')
axes[1,2].set_ylabel('Count')
axes[1,2].tick_params(rotation=0)

plt.tight_layout()
plt.savefig('outputs/eda_plots.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved → outputs/eda_plots.png")

# ─────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ─────────────────────────────────────────────────────────────
print("\n[3] Preprocessing …")

# No missing values in this dataset
print(f"   Missing values: {train_df.isnull().sum().sum()} (none)")

# Feature engineering
def engineer(df):
    df = df.copy()
    df['total_assets']    = (df['residential_assets_value'] +
                              df['commercial_assets_value'] +
                              df['luxury_assets_value'] +
                              df['bank_asset_value'])
    df['asset_to_loan']   = df['total_assets'] / (df['loan_amount'] + 1)
    df['income_to_loan']  = df['income_annum'] / (df['loan_amount'] + 1)
    df['loan_per_year']   = df['loan_amount'] / (df['loan_term'] + 1)
    df['cibil_band']      = pd.cut(df['cibil_score'],
                                    bins=[299,500,600,700,750,900],
                                    labels=[0,1,2,3,4]).astype(int)
    return df

train_df = engineer(train_df)
test_df  = engineer(test_df)

# Label Encoding
le_edu  = LabelEncoder()
le_emp  = LabelEncoder()
le_tgt  = LabelEncoder()

train_df['education']    = le_edu.fit_transform(train_df['education'])
train_df['self_employed']= le_emp.fit_transform(train_df['self_employed'])
train_df['loan_status']  = le_tgt.fit_transform(train_df['loan_status'])
# Approved=0, Rejected=1  → remap so Approved=1
# Check mapping:
mapping = dict(zip(le_tgt.classes_, le_tgt.transform(le_tgt.classes_)))
print(f"   Label mapping: {mapping}")
# If 'Approved' maps to 0, flip
if mapping['Approved'] == 0:
    train_df['loan_status'] = 1 - train_df['loan_status']
    print("   Flipped target so Approved=1")

test_df['education']     = le_edu.transform(test_df['education'])
test_df['self_employed'] = le_emp.transform(test_df['self_employed'])

# Feature set
FEATURES = ['no_of_dependents', 'education', 'self_employed',
            'income_annum', 'loan_amount', 'loan_term', 'cibil_score',
            'residential_assets_value', 'commercial_assets_value',
            'luxury_assets_value', 'bank_asset_value',
            'total_assets', 'asset_to_loan', 'income_to_loan',
            'loan_per_year', 'cibil_band']

X = train_df[FEATURES]
y = train_df['loan_status']

print(f"   Features used   : {len(FEATURES)}")

# Correlation heatmap (top features)
plt.figure(figsize=(12, 9))
corr_cols = FEATURES + ['loan_status']
corr = train_df[corr_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.4, annot_kws={'size':7}, cbar_kws={'shrink':0.8})
plt.title('Feature Correlation Heatmap', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("   Saved → outputs/correlation_heatmap.png")

# Train-Test split (from training data)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)

print(f"   Train split : {X_train.shape[0]}  |  Val split : {X_val.shape[0]}")

# ─────────────────────────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────────────────────────
print("\n[4] Training Models …")

lr_model = LogisticRegression(max_iter=2000, C=1.0, random_state=42)
lr_model.fit(X_train_sc, y_train)
lr_pred  = lr_model.predict(X_val_sc)
lr_prob  = lr_model.predict_proba(X_val_sc)[:,1]
print("   [✓] Logistic Regression trained")

rf_model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                   min_samples_leaf=4, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred  = rf_model.predict(X_val)
rf_prob  = rf_model.predict_proba(X_val)[:,1]
print("   [✓] Random Forest trained")

dt_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=6, random_state=42)
dt_model.fit(X_train, y_train)
dt_pred  = dt_model.predict(X_val)
dt_prob  = dt_model.predict_proba(X_val)[:,1]
print("   [✓] Decision Tree trained")

# ─────────────────────────────────────────────────────────────
# 5. EVALUATION
# ─────────────────────────────────────────────────────────────
print("\n[5] Evaluation Results")
print("=" * 65)

results = {}
def evaluate(name, y_true, y_pred, y_prob, model, X_full, y_full, scaled=False):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    skf  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    Xs   = scaler.transform(X_full) if scaled else X_full
    cv   = cross_val_score(model, Xs, y_full, cv=skf, scoring='accuracy')
    print(f"\n  ► {name}")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    CV 5-Fold : {cv.mean():.4f} ± {cv.std():.4f}")
    print(f"\n    Classification Report:\n")
    print(classification_report(y_true, y_pred,
          target_names=['Rejected','Approved'], zero_division=0))
    results[name] = (acc, prec, rec, f1)
    return acc, prec, rec, f1

lr_s = evaluate("Logistic Regression", y_val, lr_pred, lr_prob,
                 lr_model, X, y, scaled=True)
rf_s = evaluate("Random Forest",       y_val, rf_pred, rf_prob,
                 rf_model, X, y, scaled=False)
dt_s = evaluate("Decision Tree",       y_val, dt_pred, dt_prob,
                 dt_model, X, y, scaled=False)
print("=" * 65)

# ─────────────────────────────────────────────────────────────
# 6. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────
print("\n[6] Generating graphs …")

COLORS = {'LR':'#3498db', 'RF':'#2ecc71', 'DT':'#e67e22'}

# (a) Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Confusion Matrices – All Models', fontsize=14, fontweight='bold')
for ax, name, pred in zip(axes,
    ['Logistic Regression','Random Forest','Decision Tree'],
    [lr_pred, rf_pred, dt_pred]):
    cm = confusion_matrix(y_val, pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, linewidths=0.5,
                xticklabels=['Rejected','Approved'],
                yticklabels=['Rejected','Approved'],
                annot_kws={'size':14, 'weight':'bold'})
    ax.set_title(name, fontweight='bold')
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('outputs/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

# (b) ROC Curves
plt.figure(figsize=(8, 7))
for label, prob, color in [
    ('Logistic Regression', lr_prob, COLORS['LR']),
    ('Random Forest',       rf_prob, COLORS['RF']),
    ('Decision Tree',       dt_prob, COLORS['DT'])]:
    fpr, tpr, _ = roc_curve(y_val, prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2.2, color=color,
             label=f'{label}  (AUC = {roc_auc:.4f})')
plt.plot([0,1],[0,1],'k--', lw=1)
plt.fill_between([0,1],[0,1], alpha=0.03, color='grey')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves – Model Comparison', fontweight='bold', fontsize=13)
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# (c) Model Comparison Bar Chart
metrics = ['Accuracy','Precision','Recall','F1-Score']
m_vals  = [lr_s, rf_s, dt_s]
m_names = ['Logistic Regression','Random Forest','Decision Tree']
m_colors= [COLORS['LR'], COLORS['RF'], COLORS['DT']]

x = np.arange(len(metrics)); width = 0.25
fig, ax = plt.subplots(figsize=(12, 7))
for i, (name, vals, col) in enumerate(zip(m_names, m_vals, m_colors)):
    bars = ax.bar(x + i*width, vals, width, label=name, color=col,
                  edgecolor='white', linewidth=0.8)
    for bar in bars:
        ax.annotate(f'{bar.get_height():.3f}',
                    (bar.get_x()+bar.get_width()/2, bar.get_height()+0.005),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(x + width)
ax.set_xticklabels(metrics, fontsize=12)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
ax.legend(fontsize=11); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# (d) Feature Importance – Random Forest
fi = pd.DataFrame({'Feature': FEATURES,
                    'Importance': rf_model.feature_importances_})
fi = fi.sort_values('Importance', ascending=True)
colors_fi = ['#e74c3c' if v > 0.08 else '#3498db' for v in fi['Importance']]
plt.figure(figsize=(10, 7))
plt.barh(fi['Feature'], fi['Importance'], color=colors_fi)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Random Forest – Feature Importance', fontweight='bold', fontsize=13)
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# (e) Learning Curve
from sklearn.model_selection import learning_curve
train_sizes, tr_sc, val_sc = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    X, y, cv=5, n_jobs=-1, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10))

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, tr_sc.mean(axis=1),  'o-', lw=2, color='#3498db', label='Training Score')
plt.fill_between(train_sizes,
    tr_sc.mean(axis=1)-tr_sc.std(axis=1),
    tr_sc.mean(axis=1)+tr_sc.std(axis=1), alpha=0.15, color='#3498db')
plt.plot(train_sizes, val_sc.mean(axis=1), 'o-', lw=2, color='#e74c3c', label='Validation Score')
plt.fill_between(train_sizes,
    val_sc.mean(axis=1)-val_sc.std(axis=1),
    val_sc.mean(axis=1)+val_sc.std(axis=1), alpha=0.15, color='#e74c3c')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Random Forest – Learning Curve', fontweight='bold', fontsize=13)
plt.legend(fontsize=11); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/learning_curve.png', dpi=150, bbox_inches='tight')
plt.close()

# (f) CIBIL Score Box Plot by Loan Status
plt.figure(figsize=(8, 6))
approved = train_df[train_df['loan_status']==1]['cibil_score']
rejected = train_df[train_df['loan_status']==0]['cibil_score']
bp = plt.boxplot([rejected, approved], labels=['Rejected','Approved'],
                  patch_artist=True, notch=True,
                  medianprops={'color':'black','linewidth':2})
bp['boxes'][0].set_facecolor('#e74c3c')
bp['boxes'][1].set_facecolor('#2ecc71')
plt.ylabel('CIBIL Score', fontsize=12)
plt.title('CIBIL Score Distribution by Loan Status', fontweight='bold', fontsize=13)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/cibil_boxplot.png', dpi=150, bbox_inches='tight')
plt.close()

print("   All graphs saved in outputs/")

# ─────────────────────────────────────────────────────────────
# 7. PREDICT ON TEST SET (Kaggle test data)
# ─────────────────────────────────────────────────────────────
print("\n[7] Predictions on Kaggle Test Set …")
X_test_real = test_df[FEATURES]
X_test_sc   = scaler.transform(X_test_real)

lr_test = lr_model.predict(X_test_sc)
rf_test = rf_model.predict(X_test_real)

submission = test_df[['loan_id']].copy()
submission['loan_status_LR'] = ['Approved' if p==1 else 'Rejected' for p in lr_test]
submission['loan_status_RF'] = ['Approved' if p==1 else 'Rejected' for p in rf_test]
submission.to_csv('outputs/test_predictions.csv', index=False)
print(f"   LR – Approved: {(lr_test==1).sum()}  Rejected: {(lr_test==0).sum()}")
print(f"   RF – Approved: {(rf_test==1).sum()}  Rejected: {(rf_test==0).sum()}")
print("   Saved → outputs/test_predictions.csv")

# ─────────────────────────────────────────────────────────────
# 8. DEMO PREDICTION
# ─────────────────────────────────────────────────────────────
print("\n[8] Sample Prediction Demo")
print("-" * 65)
sample_raw = {
    'no_of_dependents':2, 'education':1, 'self_employed':0,
    'income_annum':5000000, 'loan_amount':12000000,
    'loan_term':12, 'cibil_score':720,
    'residential_assets_value':4000000, 'commercial_assets_value':3000000,
    'luxury_assets_value':8000000, 'bank_asset_value':2000000
}
s = pd.DataFrame([sample_raw])
s['total_assets']   = s[['residential_assets_value','commercial_assets_value',
                           'luxury_assets_value','bank_asset_value']].sum(axis=1)
s['asset_to_loan']  = s['total_assets'] / (s['loan_amount']+1)
s['income_to_loan'] = s['income_annum'] / (s['loan_amount']+1)
s['loan_per_year']  = s['loan_amount'] / (s['loan_term']+1)
s['cibil_band']     = pd.cut(s['cibil_score'], bins=[299,500,600,700,750,900],
                               labels=[0,1,2,3,4]).astype(int)
s_sc = scaler.transform(s[FEATURES])
print(f"  Applicant  : 2 Dependents | Graduate | Salaried | Income 5M | Loan 12M | CIBIL 720")
print(f"  LR Result  : {'APPROVED ✅' if lr_model.predict(s_sc)[0]==1 else 'REJECTED ❌'}")
print(f"  RF Result  : {'APPROVED ✅' if rf_model.predict(s[FEATURES])[0]==1 else 'REJECTED ❌'}")

print("\n[✓] All tasks completed. Graphs → outputs/")
print("=" * 65)
