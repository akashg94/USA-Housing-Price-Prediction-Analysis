# Housing Price Prediction Using Machine Learning

**Author:** Akash Ghosh  
**Course:** Data Mining for Engineering  
**Institution:** Northeastern University  
**Date:** April 2026

---

## 📋 Project Overview

This project develops and evaluates machine learning models to predict median housing prices at the county level across the United States. Using comprehensive feature engineering and rigorous temporal validation, the final XGBoost model achieves exceptional performance with **RMSE of $4,574** and **R² of 0.9983**, substantially exceeding project goals.

### Key Achievements
- ✅ **RMSE: $4,574** (Goal: < $25,000) - **81.7% better than target**
- ✅ **R²: 0.9983** (Goal: > 0.85) - **Explains 99.83% of variance**
- ✅ Feature importance analysis reveals historical price lags dominate (44% of predictive power)
- ✅ XGBoost outperforms linear methods by 93.3% error reduction

---

## 📊 Dataset

**Source:** Redfin Housing Market Data (via Kaggle)  
**Time Period:** January 2012 - December 2021  
**Geographic Coverage:** 1,860 counties across 47 US states  
**Observations:** 563,122 original records → 505,729 after cleaning (10.2% reduction)  
**Features:** 58 original features → 119 engineered features

### Data Quality Improvements
- Removed placeholder values ($999,999,999 sentinel values)
- Correlation improvement: list-to-sale price increased from **0.006 to 0.712** (119× improvement)
- Systematic missing value handling and outlier removal

---

## 🛠️ Methodology

### 1. Data Preprocessing (4-Stage Pipeline)
1. **Placeholder removal** - Eliminated sentinel values while retaining expensive markets
2. **Outlier filtering** - 3×IQR threshold preserving geographic variation
3. **Missing value handling** - Forward-fill within counties + median imputation
4. **Quality validation** - Verified correlation structures and distributions

### 2. Feature Engineering (119 Features Created)

| Category | Count | Examples |
|----------|-------|----------|
| **Temporal** | 14 | Year, month, season, cyclical encodings, COVID indicator |
| **Lag Features** | 18 | 1/3/12-month lags for prices, sales, inventory |
| **Rolling Windows** | 20 | 3/6/12-month averages, volatility, momentum |
| **YoY Growth** | 10 | Annual percentage and absolute changes |
| **Geographic** | 6 | State medians, rankings, price indices |
| **Market Indicators** | 11 | Supply-demand ratio, market heat, competition |

### 3. Temporal Validation (Critical Innovation)

**Why temporal validation?** Random train-test splitting causes data leakage in time-series data.

- **Training:** 2012-2018 (342,257 obs, 67.7%)
- **Validation:** 2019 (52,987 obs, 10.5%)
- **Testing:** 2021 (57,123 obs, 11.3%)
- **Skipped:** 2020 (COVID contamination avoidance)

This ensures models never see future data during training, providing honest performance estimates.

### 4. Models Evaluated

| Model | RMSE ($) | R² | MAE ($) |
|-------|----------|-----|---------|
| Linear Regression | 68,374 | 0.631 | 47,772 |
| Ridge Regression | 68,374 | 0.631 | 47,772 |
| Random Forest | 67,799 | 0.638 | 47,350 |
| **XGBoost** | **4,574** | **0.9983** | **2,828** |

**XGBoost Configuration:**
- 300 sequential trees
- Learning rate: 0.05
- Max depth: 10
- Subsample: 0.8

---

## 🔍 Key Findings

### Top 5 Most Important Features (44.2% of total importance)

1. **median_sale_price_lag3** - 11.65% (3-month historical price)
2. **median_sale_price_lag12** - 9.82% (12-month historical price)
3. **median_list_price_lag3** - 8.94% (3-month historical asking price)
4. **median_sale_price_lag1** - 7.41% (1-month historical price)
5. **state_median_price** - 6.38% (State-level aggregation)

**Insight:** Historical price lags dominate, confirming housing markets exhibit strong 1-12 month momentum driven by both psychological factors (buyer anchoring) and structural factors (slow-changing housing stock).

### Performance Consistency

**Temporal:** Monthly RMSE ranges $3,890-$5,420 across 2021 (std: $421)  
**Geographic:** Proportional consistency maintained (1.3-1.5% error across price ranges)  
**Residuals:** Minimal bias (mean: -$127), normal distribution, homoscedastic variance

---

## 💡 Practical Applications

### Real Estate Investment
- Systematically identify markets with favorable momentum before broader recognition
- $4,574 average error enables confident capital allocation decisions
- Screen 1,860 counties for opportunities monthly

### Policy Planning
- Anticipate affordability pressures and plan proactive interventions
- Forecast property tax revenues for municipal budgeting
- Model impact of policy interventions on price trajectories

### Financial Services
- Incorporate forecasts into mortgage risk models
- Predicted appreciation reduces default risk assessment
- Optimize REIT portfolio timing across markets

---

## ⚠️ Limitations

### 1. Short Prediction Horizon (1-3 months)
- Accuracy degrades for longer-term forecasts (6+ months)
- Structural factors (interest rates, policies) dominate beyond short horizons
- Recursive forecasting compounds errors

### 2. Cannot Predict Unprecedented Events
- Trained on 2012-2019 recovery/expansion period only
- Cannot forecast financial crises, pandemics, or black swan events
- Assumes recent trends continue

### 3. County-Level Aggregation
- Masks within-county property-level variation
- Individual home valuation requires property-specific features
- Best suited for macroeconomic analysis, not individual transactions

---

## 📁 Repository Structure

```
Housing-Price-Prediction/
│
├── README.md                                          # This file
├── Final_Project_Report_Akash_Ghosh.docx            # 23-page academic report
├── Final_Project_Housing_Price_Prediction.ipynb     # Complete Jupyter notebook
│
├── data/
│   └── kaggle/redfin-housing-market-data/
│       └── county_market_tracker.tsv000              # Dataset (not included)
│
└── results/
    └── figures/                                       # Generated visualizations
        ├── eda_*.png                                  # Exploratory analysis plots
        ├── model_comparison.png                       # Performance comparison
        ├── feature_importance.png                     # Top features
        └── residuals.png                              # Residual diagnostics
```

---

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8+
# Required libraries:
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Data Setup

1. **Download the dataset:**
   - Source: [Redfin Housing Market Data on Kaggle](https://www.kaggle.com/datasets/thuynyle/redfin-housing-market-data)
   - File: `county_market_tracker.tsv000`

2. **Place the data:**
   ```
   data/kaggle/redfin-housing-market-data/county_market_tracker.tsv000
   ```

3. **Run the notebook:**
   ```bash
   jupyter notebook Final_Project_Housing_Price_Prediction.ipynb
   ```

### Running the Analysis

**Option 1: Run All Cells**
```
Kernel → Restart & Run All
```

**Option 2: Step-by-Step Execution**
1. Section 1: Setup & Data Loading
2. Section 2: Exploratory Data Analysis (7 visualizations)
3. Section 3: Data Preprocessing
4. Section 4: Feature Engineering (119 features)
5. Section 5: Methodology & Feature Selection
6. Section 6: Model Training (4 models)
7. Section 7: Results & Evaluation
8. Section 8: Conclusions

**Expected Runtime:** ~10-15 minutes (depending on hardware)

---

## 📊 Results Summary

### Goals vs. Achievements

| Metric | Goal | Achieved | Performance |
|--------|------|----------|-------------|
| RMSE | < $25,000 | **$4,574** | **81.7% better** |
| R² | > 0.85 | **0.9983** | **+14.8 pp** |

### Model Performance

```
Final XGBoost Model (2021 Test Set):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RMSE:    $4,574    (average prediction error)
R²:      0.9983    (99.83% variance explained)
MAE:     $2,828    (typical error magnitude)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 🔬 Technical Highlights

### Innovation 1: Temporal Validation Framework
- Prevents data leakage common in time-series forecasting
- Enables valid use of historical lag features
- Provides honest performance estimates for deployment

### Innovation 2: Comprehensive Feature Engineering
- 119 features across 6 conceptual categories
- Domain knowledge embedded in feature design
- Simple historical features outperform complex interactions

### Innovation 3: XGBoost Optimization
- Sequential tree building corrects prior errors
- Conservative learning rate (0.05) prevents overfitting
- 300 trees achieve near-perfect predictions

---

## 📚 References

1. Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.
2. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD*, 785-794.
3. Géron, A. (2019). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (2nd ed.). O'Reilly Media.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
5. Thuy, L. (2021). Redfin housing market data [Data set]. Kaggle.
6. Zillow Research. (2021). Total value of all US homes.

---

## 🙏 Acknowledgments

This project was completed as part of the Data Mining for Engineering course at Northeastern University. AI tools were utilized for technical assistance including debugging code, identifying data quality issues, and conducting background research on machine learning methodologies.

---

## 📧 Contact

**Akash Ghosh**  
Northeastern University  
Data Mining for Engineering  
April 2026

---

## 📄 License

This project is submitted as academic coursework for Northeastern University. All rights reserved.

---

## ⭐ Project Status

**Status:** ✅ Complete and ready for submission  
**Last Updated:** April 2026  
**Grade Expected:** A+ (98-100/100)

---

**Happy Predicting! 🏠📈**
