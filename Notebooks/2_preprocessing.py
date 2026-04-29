#!/usr/bin/env python
# coding: utf-8

# # Data Cleaning and Feature Engineering
# 
# This workbook cleans raw NHL game data into a structured dataset for modeling playoff prediction. It will aggregate team performance over the first 20 games, create meaningful engineered features and remove redundancy and noise.

# In[4]:


import pandas as pd
import numpy as np

df = pd.read_csv('fake_nhl_dataset.csv')

print("Original shape:", df.shape)


# In[5]:


df_pre20 = df[df["Game_Number"] <= 20].copy()
print("Pre-20 shape:", df_pre20.shape)


# In[6]:


group_cols = ["Season", "Team_Code"]

df_agg = df_pre20.groupby(group_cols).agg({
    "GF": "mean",
    "GA": "mean",
    "Points": "mean",
    "F_count": "max",
    "D_count": "max",
    "G_count": "max",
    "F_cost": "mean",
    "D_cost": "mean",
    "G_cost": "mean",
    "Total_injuries": "mean",
    "Opp_Players_Out": "mean",
    "LowerBody_Count": "mean",
    "UpperBody_Count": "mean"
}).reset_index()


# In[7]:


df_agg["Goal_Diff"] = df_agg["GF"] - df_agg["GA"]
df_agg["Points_Per_Game"] = df_agg["Points"] / 20

df_agg["Roster_Size"] = (
    df_agg["F_count"] + df_agg["D_count"] + df_agg["G_count"]
)

df_agg["Injury_Rate"] = df_agg["Total_injuries"] / df_agg["Roster_Size"]

df_agg["Opp_Injury_Diff"] = (
    df_agg["Total_injuries"] - df_agg["Opp_Players_Out"]
)

df_agg["Total_Injury_Type"] = (
    df_agg["LowerBody_Count"] + df_agg["UpperBody_Count"]
)


# In[8]:


df_target = (
    df_pre20
    .groupby(group_cols)["Playoffs_Bin"]
    .max()
    .reset_index()
)

df_model = df_agg.merge(df_target, on=group_cols)


# In[9]:


df_model = df_model.drop(
    columns=["Team_Code", "Playoffs", "Playoffs_B"],
    errors="ignore"
)

low_var_cols = df_model.columns[df_model.nunique() <= 1]
df_model = df_model.drop(columns=low_var_cols)

corr_matrix = df_model.drop("Playoffs_Bin", axis=1).corr().abs()

upper = corr_matrix.where(
    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
)

high_corr_cols = [
    col for col in upper.columns if any(upper[col] > 0.90)
]

df_model = df_model.drop(columns=high_corr_cols)


# In[10]:


print("Final shape:", df_model.shape)
df_model.head()

df_model.to_csv("final_hockey_datasetfake.csv", index=False)

