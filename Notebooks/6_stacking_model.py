#!/usr/bin/env python
# coding: utf-8

# # Stacking Ensemble Model
# 
# This notebook builds a stacking model combining Random Forest, SVM, XGBoost. A Logistic Regression model is used as the meta-learner.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier


# In[2]:


df = pd.read_csv("final_hockey_datasetfake.csv")


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


rf = RandomForestClassifier(n_estimators=300, random_state=42)

svm = Pipeline([
    ("scaler", StandardScaler()),
    ("model", SVC(probability=True, random_state=42))
])

xgb = XGBClassifier(
    random_state=42,
    eval_metric="logloss",
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05
)


# In[6]:


stack_model = StackingClassifier(
    estimators=[("rf", rf), ("svm", svm), ("xgb", xgb)],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    passthrough=True,
    n_jobs=-1
)


# In[7]:


stack_model.fit(X_train, y_train)


# In[8]:


y_pred_stack = stack_model.predict(X_test)
y_prob_stack = stack_model.predict_proba(X_test)[:, 1]

cv_stack = cross_val_score(
    stack_model, X_train, y_train, cv=5, scoring="roc_auc"
).mean()

print("\nSTACKING RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_stack))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_stack))
print("F1:", f1_score(y_test, y_pred_stack))
print("CV AUC:", cv_stack)


# In[9]:


xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]

cv_xgb = cross_val_score(
    xgb, X_train, y_train, cv=5, scoring="roc_auc"
).mean()

print("\nXGBOOST RESULTS")
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_xgb))
print("F1:", f1_score(y_test, y_pred_xgb))
print("CV AUC:", cv_xgb)


# In[10]:


plt.figure()
plt.bar(["Stacking", "XGBoost"], [
    roc_auc_score(y_test, y_prob_stack),
    roc_auc_score(y_test, y_prob_xgb)
])
plt.ylim(0.5, 1.0)
plt.title("Stacking vs XGBoost")
plt.ylabel("ROC-AUC")
plt.show()


# ### Key Takeaways
# *the numbers and outcome reflect the real NHL data set used in this project
# 
# The stacking model achieved the best overall performance, with an ROC-AUC of approximately 0.84 and the highest F1 score, indicating strong predictive ability and balance between precision and recall. XGBoost produced a nearly identical ROC-AUC, but the stacking model showed better a better CV score, suggesting more consistent generalization across different data splits. The strong stacking results suggests that different models are capturing complementary patterns within the data.This reinforces that early-season team performance can be used to predict playoff outcomes accurately. The results also indicate that model performance is driven more by feature quality and data representation than by model complexity alone.
