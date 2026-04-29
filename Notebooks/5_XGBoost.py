#!/usr/bin/env python
# coding: utf-8

# # XGBoost Model
# 
# This workbook implements XGBoost to capture complex nonlinear relationships in the data. XGBoost is evaluated both as a standalone model and as a candidate for being used in stacking model.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, roc_curve


# In[2]:


df = pd.read_csv("final_hockey_datasetfake.csv")

print("Shape:", df.shape)


# In[3]:


df = df.drop(columns=["Season"], errors="ignore")

X = df.drop("Playoffs_Bin", axis=1)
y = df["Playoffs_Bin"]


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)



# In[5]:


scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale_pos_weight: {scale_pos_weight:.2f}")



# In[6]:


xgb_baseline = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

xgb_baseline.fit(X_train, y_train)




# In[7]:


cv_baseline = cross_val_score(
    xgb_baseline, X_train, y_train, cv=5, scoring="roc_auc"
).mean()

y_pred = xgb_baseline.predict(X_test)
y_prob = xgb_baseline.predict_proba(X_test)[:, 1]

print("\nBaseline XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("F1:", f1_score(y_test, y_pred))
print("CV AUC:", cv_baseline)


# In[8]:


param_dist = {
    "n_estimators": [300, 500, 800],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
    "gamma": [0, 0.1, 0.3],
    "min_child_weight": [1, 3, 5]
}

xgb_model = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

xgb_random = RandomizedSearchCV(
    xgb_model,
    param_dist,
    n_iter=30,
    cv=5,
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42
)

xgb_random.fit(X_train, y_train)

print("\nBest Params:", xgb_random.best_params_)
print("Best CV Score:", xgb_random.best_score_)

best_xgb = xgb_random.best_estimator_



# In[9]:


cv_tuned = cross_val_score(
    best_xgb, X_train, y_train, cv=5, scoring="roc_auc"
).mean()

y_pred = best_xgb.predict(X_test)
y_prob = best_xgb.predict_proba(X_test)[:, 1]

print("\nTuned XGBoost")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("F1:", f1_score(y_test, y_pred))
print("CV AUC:", cv_tuned)



# In[10]:


fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.figure()
plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], linestyle="--")
plt.title("ROC Curve - XGBoost")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.show()


# In[11]:


importance = pd.Series(
    best_xgb.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

plt.figure()
importance.head(15).sort_values().plot(kind="barh")
plt.title("Top Features - XGBoost")
plt.show()


# ### Key Takeaways
# *the numbers and outcome reflect the real NHL data set used in this project
# 
# XGBoost preformance better than the baseline models by capturing nonlinear relationships and interactions between features. Hyperparameter tuning provides marginal gains, with improvements primarily seen in ROC-AUC. Feature importance aligns with earlier findings, with Goal Differential and Points remaining dominant predictors. XGBoost serves as a strong standalone model and a key component for a stacking model.
