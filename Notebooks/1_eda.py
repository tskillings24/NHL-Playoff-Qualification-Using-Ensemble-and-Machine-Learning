#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)
# 
# This workbook explores the raw NHL dataset prior to preprocessing. 
# The goal is to understand early-season team performance and guide feature engineering decisions.
# 
# All analysis is conducted on the first 20 games to simulate early playoff prediction.

# In[12]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('fake_nhl_dataset.csv')

print("Original shape:", df.shape)
df.head()


# In[13]:


df_pre20 = df[df["Game_Number"] <= 20].copy()

print("Pre-20 shape:", df_pre20.shape)


# ## Raw Data Distributions
# 
# Examining goal production and goals allowed across teams in the first 20 games.

# In[14]:


plt.figure()
sns.histplot(df_pre20["GF"], kde=False)
plt.title("Goals For Distribution (First 20 Games)")
plt.show()

plt.figure()
sns.histplot(df_pre20["GA"], kde=False)
plt.title("Goals Against Distribution (First 20 Games)")
plt.show()


# There is clear variation in both goals scored and goals allowed across teams early in the season, suggesting differences in team strength.

# ## Feature Engineering
# 
# To better capture team performance features such as goal differential and injury rate are constructed at the team-season level.

# In[15]:


group_cols = ["Season", "Team_Code"]

df_agg = df_pre20.groupby(group_cols).agg({
    "GF": "mean",
    "GA": "mean",
    "Points": "mean",
    "F_count": "max",
    "D_count": "max",
    "G_count": "max",
    "Total_injuries": "mean",
    "Opp_Players_Out": "mean",
    "LowerBody_Count": "mean",
    "UpperBody_Count": "mean"
}).reset_index()

df_agg["Goal_Diff"] = df_agg["GF"] - df_agg["GA"]

df_agg["Roster_Size"] = (
    df_agg["F_count"] + df_agg["D_count"] + df_agg["G_count"]
)

df_agg["Injury_Rate"] = df_agg["Total_injuries"] / df_agg["Roster_Size"]


# In[16]:


plt.figure()
sns.histplot(df_agg["Goal_Diff"], kde=False)
plt.title("Goal Differential Distribution")
plt.show()

plt.figure()
sns.histplot(df_agg["Injury_Rate"], kde=False)
plt.title("Injury Rate Distribution")
plt.show()


# Goal differential provides a clearer measure of team strength, while injury rate captures roster health. Both likely will influence playoff outcomes.

# ## Feature Relationships
# 
# Examining correlations between selected variables to identify redundancy and relationships.

# In[17]:


selected_cols = [
    "GF", "GA", "Points", "Goal_Diff",
    "Total_injuries", "Injury_Rate"
]

corr = df_agg[selected_cols].corr()

plt.figure(figsize=(8,6))
plt.imshow(corr, cmap="Blues")

plt.xticks(range(len(selected_cols)), selected_cols, rotation=45, ha="right")
plt.yticks(range(len(selected_cols)), selected_cols)

for i in range(len(selected_cols)):
    for j in range(len(selected_cols)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center", color="black")

plt.title("Feature Correlation")
plt.colorbar()
plt.tight_layout()
plt.show()


# Some correlation exists between performance-related variables, but not at levels that would prevent being used together. This shows that keeping several different features helps the model perform better.
