# Accident Risk Zone Predictor

A complete end-to-end predictive analytics and decision support system designed to model and simulate road accident risk coefficients. The system utilizes machine learning algorithms trained on environmental, road class, and situational variables (such as time of day, precipitation, traffic density, and location type) to predict hazard rates and offer real-time safety recommendations.

## Project Overview

Traffic accidents are a leading cause of global mortality and infrastructure delay. Identifying high-risk zones and conditions before accidents occur can save lives and optimize dispatch protocols. This project implements a simulation and predictive pipeline using a custom **Random Forest Regressor** and compares it with a baseline **Linear Regression** model to quantify risk under various situational constraints.

## Problem Statement

Standard crash modeling relies on historical, post-event accident data which is often lagged, incomplete, and restricted by location-specific biases. This project addresses the challenge by establishing an environmental-stress model: simulating and predicting a risk score (0 to 100) under dynamic combinations of weather conditions, road classification, congestion levels, and time parameters.

## Directory Structure

```text
AI accident predictor/
│
├── venv/                       # Python Virtual Environment
├── dataset.csv                 # Generated Simulator Database (1000 records)
├── generate_data.py            # Data Generation Engine (Realistic Compound Rules)
├── ml_components.py            # Pure Python/NumPy Custom ML Core (Zero DLL Blocks)
├── train_and_create_notebook.py# Jupyter Notebook & Model Generator
├── model.ipynb                 # Comprehensive EDA and Model Training Log
├── model.pkl                   # Trained Random Forest Regressor Pipeline
├── app.py                      # Interactive Streamlit Web Application
├── traffic.jpg                 # User Uploaded Real Traffic Background Image
├── README.md                   # Project Documentation
│
# EDA Visualizations Generated during Training:
├── risk_distribution.png
├── risk_vs_hour.png
├── risk_vs_weather.png
├── risk_vs_traffic.png
├── risk_area_weather_interaction.png
├── correlation_heatmap.png
├── rf_actual_vs_predicted.png
└── feature_importance.png
```

## Methodology & Preprocessing

### 1. Synthetic Data Generation
We generated a database containing 1,000 situational records. To match real-world safety constraints, compound risk logic was encoded:
- **Time Penalty:** Late night hours (18h to 4h) add a +25 baseline risk coefficient due to fatigue and poor lighting.
- **Precipitation/Visibility Multipliers:** Fog adds +30 and rain adds +20 to the baseline risk.
- **Traffic Congestion:** High traffic density adds +25 risk.
- **Road Classification:** Highways represent high speed limits, adding +15.
- **Interaction Effects:**
  - Highway + Fog: Compounds risk by adding an extra +25 (extreme speed + low visibility).
  - Night + Rain: Adds +15 (wet surfaces + low visibility).
  - Rain + High Traffic: Adds +10 (slippery roads + tight vehicle spacing).

### 2. Preprocessing
To support categorical variables without importing C-compiled libraries (which are frequently blocked by local application control policies), we built a custom pipeline:
- **One-Hot Encoding:** Categorical variables (`weather`, `traffic`, `area_type`) are encoded into distinct column vectors.
- **Feature Alignment:** Fits the categorical categories during training and retains index-matching for new, single-row user inference frames during deployment.

---

## Machine Learning Models

### 1. Custom Random Forest Regressor
- **Architecture:** Ensemble of Decision Tree Regressors built from scratch in pure NumPy.
- **Hyperparameters:** `n_estimators=15`, `max_depth=6`, `max_features='sqrt'`.
- **Feature Split Criterion:** Variance reduction (impurity gain) at each node.

### 2. Custom Ridge Linear Regression
- **Architecture:** L2-regularized Ordinary Least Squares (OLS) solver using the closed-form normal equation:
  $$\beta = (X^T X + \alpha I)^{-1} X^T y$$
- **Hyperparameters:** L2 regularization constant $\alpha = 10^{-3}$ (used to guarantee numerical stability and solve collinearity from one-hot encoding).

---

## Validation Results

Evaluating both models on a held-out test set (20% split) yields:

| Metric | Random Forest Regressor | Linear Regression (Ridge) |
| :--- | :---: | :---: |
| **Root Mean Squared Error (RMSE)** | **9.01** | **11.67** |
| **Coefficient of Determination ($R^2$)** | **79.8%** | **66.1%** |

### Key Insights:
- **Random Forest** achieved a significantly higher $R^2$ score (79.8% vs 66.1%) and lower RMSE (9.01% vs 11.67%).
- This is because Random Forest inherently captures the non-linear interaction terms (e.g., Highway + Fog compound risk increase) that the linear model struggles to capture without manual feature engineering.

---

## Installation & Setup

1. **Clone or Navigate to the Directory:**
   ```bash
   cd "c:\Users\Asus Tuf\Desktop\AI accident predictor"
   ```

2. **Activate the Virtual Environment:**
   - On Windows:
     ```powershell
     .\venv\Scripts\activate
     ```

3. **Install Dependencies:** (If not already installed)
   ```bash
   pip install pandas numpy streamlit matplotlib seaborn joblib nbformat
   ```

4. **Run Model Training and Notebook Generation:**
   ```bash
   python train_and_create_notebook.py
   ```
   *This executes the pipeline, outputs the model file (`model.pkl`), saves the EDA plots, and generates the notebook file (`model.ipynb`).*

5. **Start the Interactive Web Application:**
   ```bash
   streamlit run app.py
   ```

---

## Interactive UI and Features

The application is split into four professional workspaces:
1. **Live Risk Predictor:** Live simulation page with sliders and selector widgets. Predicts risk coefficient in real-time, displays a responsive hazard level progress bar (Low/Medium/High), and supplies contextual safety warnings (e.g. hydroplaning alerts, speed warnings).
2. **Scenario Simulator:** A comparative dashboard to inspect two situational scenarios side-by-side and view the resulting delta in accident risk.
3. **Analytics Dashboard:** Visualizes the exploratory data analysis plots directly inside the web browser.
4. **Model Performance & Architecture:** Displays evaluation metrics, model architecture design, and variance-reduction feature importances.

## Screenshots

*(Placeholder text for screenshots)*
- **Live Risk Predictor Workspace:** Shows the input forms in a dark blue grid card, the circular/linear progress bar glowing green (for low risk), yellow (for medium risk), and red (for high risk) accompanied by bold status alerts.
- **Scenario Simulator Workspace:** Displays side-by-side metrics showing the difference between a high-risk situation (night, fog, highway) and a safe one (daytime, clear, urban).
- **Analytics Workspace:** Displays line charts representing risk over a 24-hour cycle and interaction heatmaps showing how highway driving compounds weather risk.

---

## Future Improvements

1. **Real-time GIS API Integration:** Incorporate active Google Maps / OpenStreetMap APIs to predict risk on actual coordinates based on real-time traffic and local weather station data.
2. **Time-Series Forecasting:** Implement LSTM or GRU models to forecast accident risk curves 1-3 hours ahead based on expected weather developments.
3. **Feature Expansion:** Add pavement condition parameters (dry, wet, ice, gravel) and driver demographic simulation metrics.
