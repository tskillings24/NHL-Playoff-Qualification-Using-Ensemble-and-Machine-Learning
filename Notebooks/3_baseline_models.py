#!/usr/bin/env python
# coding: utf-8

# # Baseline Model Comparison
# 
# This workbook evaluates multiple traditional machine learning models to predict NHL playoff qualification.
# 
# Models are compared using:
# - Accuracy
# - ROC-AUC
# - F1 Score
# - Cross-validation performance
# 
# These results provide a benchmark before exploring more advanced models such as XGBoost and stacking.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings("ignore")


# In[2]:


df = pd.read_csv("final_hockey_datasetfake.csv")

print("Dataset shape:", df.shape)
df.head()


# In[4]:


df = df.drop(columns=["Season"], errors="ignore")

target = "Playoffs_Bin"

X = df.drop(target, axis=1)
y = df[target]


# In[5]:


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# In[6]:


scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[7]:


models = {
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    "Neural Network": MLPClassifier(hidden_layer_sizes=(50,25), max_iter=500, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42)
}


# In[8]:


results = []

print("MODEL COMPARISON")
print("="*60)
print(f'{"Model":<20} {"Accuracy":<12} {"ROC-AUC":<12} {"F1":<10} {"CV Score":<12}')
print("-"*60)

for name, model in models.items():

    if name in ["KNN", "SVM", "Neural Network", "Logistic Regression"]:
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:,1]

        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

    else:
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    cv_mean = cv_scores.mean()

    results.append({
        "Model": name,
        "Accuracy": acc,
        "ROC-AUC": auc,
        "F1 Score": f1,
        "CV Score": cv_mean
    })

    print(f'{name:<20} {acc:<12.3f} {auc:<12.3f} {f1:<10.3f} {cv_mean:<12.3f}')

results_df = pd.DataFrame(results)


# In[9]:


fig, axes = plt.subplots(1,2, figsize=(14,5))

# ROC-AUC comparison
results_sorted = results_df.sort_values("ROC-AUC")

axes[0].barh(results_sorted["Model"], results_sorted["ROC-AUC"])
axes[0].set_xlabel("ROC-AUC")
axes[0].set_title("Model ROC-AUC Comparison")
axes[0].axvline(0.5, linestyle="--")

# Multi-metric comparison
x = np.arange(len(results_df))
width = 0.2

axes[1].bar(x-width, results_df["Accuracy"], width, label="Accuracy")
axes[1].bar(x, results_df["ROC-AUC"], width, label="ROC-AUC")
axes[1].bar(x+width, results_df["F1 Score"], width, label="F1")

axes[1].set_xticks(x)
axes[1].set_xticklabels(results_df["Model"], rotation=45)
axes[1].legend()

plt.tight_layout()
plt.show()


# In[10]:


rf_final = RandomForestClassifier(n_estimators=200, random_state=42)
rf_final.fit(X_train, y_train)

importance = pd.Series(
    rf_final.feature_importances_,
    index=X_train.columns
).sort_values(ascending=False)

print(importance)

top_features = importance.head(10)

plt.figure(figsize=(8,6))
plt.barh(top_features.index[::-1], top_features.values[::-1])
plt.xlabel("Feature Importance")
plt.title("Top Features Predicting NHL Playoffs")
plt.show()


# ### Key Takeaways
# *the numbers and outcome reflect the real NHL data set used in this project
# 
# Baseline model performance shows relatively consistent results across algorithms, with most models achieving accuracy between 0.66 and 0.71 and ROC-AUC scores in the 0.72–0.80 range.
# 
# Naive Bayes and Logistic Regression achieved the highest ROC-AUC scores (~0.80 and ~0.78), showing strong ability to distinguish between playoff and non-playoff teams. SVM and Random Forest also performed well, suggesting that both linear and nonlinear relationships are present in the data.
# 
# Feature importance from the Random Forest model highlights Goal Differential as the most influential predictor, followed by Points and Goals Against. This reinforces the importance of overall team performance metrics in predicting playoff outcomes. Injury-related features, such as Injury Rate and Total Injuries, contribute moderately, suggesting that while roster health plays a role, it is secondary to core performance indicators.
# 
# These baseline models show a strong start. Random Forest, SVM, and Logistic Regression demonstrate the most consistent performance across evaluation metrics and are selected as primary candidates for further tuning and optimization. More advanced approaches, such as XGBoost and ensemble methods will also be explored. 
