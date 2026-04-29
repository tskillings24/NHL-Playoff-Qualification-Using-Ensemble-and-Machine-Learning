# NHL-Playoff-Qualification-Using-Ensemble-and-Machine-Learning
## Early Season Prediction for Post Season Qualification

### **Project Overview**
This practicum project investigates whether performance data from the first 20 games of an NHL season can accurately predict playoff qualification. While mid-season trades and late-season changes are common narratives in hockey, this analysis demonstrates that "early-window" team metrics provide a high-confidence signal for long-term success.

This study utilizes a Stacking Ensemble Model to outperform traditional baseline models.

**Dataset & Scope**
* **Timeframe:** 2008–2024 NHL Seasons
* **Observations:** 522 unique team-seasons
* **Evaluation Window:** The first 20 games of each respective season
* **Inclusion:** All active NHL franchises during the sample period

### **Early Season Importance**
A common debate in sports analytics is how much weight to give to October and November games. Through Exploratory Data Analysis (EDA), this project identified that core efficiency metrics, specifically Goal Differential and Points Per Game, stabilize early. By using these features from the first 20 games, we can filter out late-season schedule fluctuations to identify true playoff contenders well before the halfway mark.

## **Analytical Approach**
The project progressed from simple baseline classifiers to a sophisticated Stacking Ensemble.
* **Features:** Goal Differential, Points Per Game, Roster Size, and engineered Injury Impact scores.
* **Outcome:** Playoff Qualification (Binary: Yes/No)
* **Pre-processing:** Removal of possibe data leakage and correlation filtering (threshold > 0.9) to ensure model stability and proper fitting.

### **Model Evolution & Performance**
The project evaluated several architectures, finding that a stacking model best captured the non-linear nature of NHL standings.

| Model | ROC-AUC | Accuracy | F1-Score |
| :--- | :--- | :--- | :--- |
| Logistic Regression (Baseline) | 0.78 | 0.72 | 0.74 |
| XGBoost | 0.82 | 0.75 | 0.78 |
| Stacking Ensemble (Final)| 0.84 | 0.77 | 0.80 |

### **Key Findings**
* **Stacking Success:** Combining Random Forest, SVM, and XGBoost into a Logistic Regression meta-learner provided a 6% lift in AUC over baseline models.
* **Predictive Dominance:** Goal Differential was the strongest early-season predictor, significantly outweighing power-play percentages or roster size.
* **Injury Resilience:** While injury-related features showed moderate impact, team depth (roster size) in the first 20 games served as a secondary stabilizer for playoff-bound teams.

## **Repository Structure**

1. **`Notebooks/`** – Chronological workflow:
   - `1_data_cleaning_eda.ipynb` – Pipeline for filtering the "20-game window" and handling historical data inconsistencies.
   - `2_feature_analysis.ipynb` – Statistical validation of feature importance and distribution analysis.
   - `3_baseline_models.ipynb` – Evaluation of KNN, Naive Bayes, Decision Trees, and standard Logistic Regression.
   - `4_model_tuning.ipynb` – Hyperparameter optimization using RandomizedSearchCV for RF and SVM.
   - `5_xgboost.ipynb` – Implementation of Gradient Boosting with class imbalance handling.
   - `6_stacking_model.ipynb` – The final production model and ensemble architecture.
 
2. **`data/`** – Contains `fake_nhl_dataset.csv`. This synthetic data mirrors the general structure of the private master dataset used in the full analysis to allow for pipeline replication.

3. **`visuals/`** – High-resolution outputs:
   - **ROC Curves:** Visualizing the trade-off between sensitivity and specificity. in the XGBoost Model
   - **Feature Importance:** Ranking the 20-game metrics by predictive importance.
   - **Model Comparison Bar Chart:** Benchmarking all 7+ models.
   - **EDA:** Startictaicl spreads of goal differential, goals for, goals against, and injury rate

## **Strategic & Professional Implications**
This analysis supports the front-office concept that:
* The first quarter of the season is highly influencial on post season success
* Goal Differential is a more reliable "success signal" than raw points early in the season.
* Management can use these early thresholds to make "Buyer vs. Seller" decisions well ahead of the trade deadline.

## **Important Limitations**
* **Data Privacy:** The primary dataset is proprietary; results were validated on private data, while the repo uses a synthetic proxy.
* **Roster Volatility:** Models do not currently account for mid-season superstar trades or goaltending changes occurring after the 20-game mark. The model focuses on soley the frst 20 games.


### **Contact**
**Terah Skillings** Regis University – M.S. Data Science  
GitHub: [tskillings24](https://github.com/tskillings24)
