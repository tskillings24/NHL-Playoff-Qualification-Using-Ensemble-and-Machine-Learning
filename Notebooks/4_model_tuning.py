#!/usr/bin/env python
# coding: utf-8

# # Model Tuning
# 
# This workbook focuses on tuning the top-performing baseline models from workbook 3. The models used in this workbook are Random Forest, Support Vector Machine (SVM), and Logistic Regression. Hyperparameters are tuned using RandomizedSearchCV with cross-validation, with ROC-AUC as the primary evaluation metric.

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


# In[2]:


df = pd.read_csv("final_hockey_datasetfake.csv")

print("Dataset shape:", df.shape)


# In[3]:


df = df.drop(columns=["Season"], errors="ignore")

target = "Playoffs_Bin"

X = df.drop(target, axis=1)
y = df[target]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[5]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[6]:


rf = RandomForestClassifier(random_state=42)

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    rf,
    rf_params,
    n_iter=15,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print("Best RF Params:", rf_search.best_params_)


# In[7]:


svm = SVC(probability=True, random_state=42)

svm_params = {
    "C": [0.1, 1, 10, 50],
    "gamma": ["scale", 0.01, 0.001],
    "kernel": ["rbf", "linear"]
}

svm_search = RandomizedSearchCV(
    svm,
    svm_params,
    n_iter=10,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

svm_search.fit(X_train_scaled, y_train)
best_svm = svm_search.best_estimator_

print("Best SVM Params:", svm_search.best_params_)


# In[8]:


log = LogisticRegression(max_iter=2000, random_state=42)

log_params = {
    "C": np.logspace(-3, 3, 20),
    "penalty": ["l1", "l2"],
    "solver": ["liblinear", "saga"]
}

log_search = RandomizedSearchCV(
    log,
    log_params,
    n_iter=15,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

log_search.fit(X_train_scaled, y_train)
best_log = log_search.best_estimator_

print("Best LogReg Params:", log_search.best_params_)


# In[9]:


def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:,1]
    cv = cross_val_score(model, X_tr, y_tr, cv=5).mean()

    return (
        accuracy_score(y_te, pred),
        roc_auc_score(y_te, prob),
        f1_score(y_te, pred),
        cv
    )

rf_metrics = evaluate_model(best_rf, X_train, X_test, y_train, y_test)
svm_metrics = evaluate_model(best_svm, X_train_scaled, X_test_scaled, y_train, y_test)
log_metrics = evaluate_model(best_log, X_train_scaled, X_test_scaled, y_train, y_test)


# In[10]:


results = pd.DataFrame({
    "Model": ["Random Forest", "SVM", "Logistic Regression"],
    "Accuracy": [rf_metrics[0], svm_metrics[0], log_metrics[0]],
    "ROC-AUC": [rf_metrics[1], svm_metrics[1], log_metrics[1]],
    "F1 Score": [rf_metrics[2], svm_metrics[2], log_metrics[2]],
    "CV Score": [rf_metrics[3], svm_metrics[3], log_metrics[3]]
})

print("\nTUNED MODEL RESULTS")
print("="*60)
print(results)


# ### Key Takeaways
# *the numbers and outcome reflect the real NHL data set used in this project
# 
# The model tuning slightly improved ROC-AUC for some models (notably SVM), but the overall performance remained relatively consistent with baseline results. This suggests that the models were already well-calibrated and that further gains are likely to come from improved feature engineering or more advanced modeling techniques rather than additional tuning.
