import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import base64

# Page Configuration - Clean Professional Layout
st.set_page_config(
    page_title="Accident Risk Zone Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Append current directory to system path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ml_components import CustomPreprocessor, SimpleRandomForestRegressor, SimpleLinearRegression, CustomPipeline

# Load the trained Model Pipeline
@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        return joblib.load("model.pkl")
    else:
        st.error("Model file 'model.pkl' not found. Please run the training script first.")
        return None

model = load_model()

# Load Dataset for dynamic analytics
@st.cache_data
def load_data():
    if os.path.exists("dataset.csv"):
        return pd.read_csv("dataset.csv")
    return None

df = load_data()

# Helper to convert local image to base64 for background injection
def get_base64_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    return ""

traffic_img_base64 = get_base64_image("traffic.jpg")

# Inject Custom Premium Stylesheet (Professional Dark/Sleek Theme)
st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Dashboard Cards */
    .metric-card {
        background: rgba(30, 41, 59, 0.45);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 8px 30px 0 rgba(0, 0, 0, 0.2);
        margin-bottom: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: rgba(255, 255, 255, 0.15);
    }
    
    /* Hero Section Container */
    .hero-container {
        position: relative;
        padding: 60px 40px;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        margin-bottom: 30px;
        overflow: hidden;
    }
    
    /* Risk Badges */
    .risk-badge {
        padding: 6px 14px;
        border-radius: 4px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 1px;
        display: inline-block;
        margin-bottom: 15px;
    }
    .risk-low {
        background-color: rgba(16, 185, 129, 0.1);
        color: #10B981;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
    .risk-medium {
        background-color: rgba(245, 158, 11, 0.1);
        color: #F59E0B;
        border: 1px solid rgba(245, 158, 11, 0.2);
    }
    .risk-high {
        background-color: rgba(239, 68, 68, 0.1);
        color: #EF4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 0 10px rgba(239, 68, 68, 0.15);
    }
    
    /* Progress Bar Container */
    .progress-bg {
        background-color: #1e293b;
        border-radius: 4px;
        width: 100%;
        height: 10px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.05);
        margin: 15px 0;
    }
    
    /* Bullet Points */
    .insight-bullet {
        margin-left: 5px;
        padding-left: 12px;
        border-left: 3px solid #3B82F6;
        margin-bottom: 12px;
        font-size: 0.95rem;
        color: #E2E8F0;
    }
    
    /* Divider */
    hr {
        border: 0;
        height: 1px;
        background: rgba(255, 255, 255, 0.08);
        margin: 25px 0;
    }
    
    /* Alert details box */
    .alert-details {
        padding: 12px 16px;
        border-radius: 6px;
        font-size: 0.95rem;
        margin-top: 15px;
        line-height: 1.5;
    }
    .alert-low {
        background-color: rgba(16, 185, 129, 0.05);
        border-left: 4px solid #10B981;
        color: #E2E8F0;
    }
    .alert-medium {
        background-color: rgba(245, 158, 11, 0.05);
        border-left: 4px solid #F59E0B;
        color: #E2E8F0;
    }
    .alert-high {
        background-color: rgba(239, 68, 68, 0.05);
        border-left: 4px solid #EF4444;
        color: #E2E8F0;
    }
    </style>
""", unsafe_allow_html=True)

# Hero Section with low opacity background image
if traffic_img_base64:
    hero_html = f"""
    <div class="hero-container" style="
        background: linear-gradient(rgba(15, 23, 42, 0.86), rgba(15, 23, 42, 0.86)), url(data:image/jpeg;base64,{traffic_img_base64});
        background-size: cover;
        background-position: center;
    ">
        <h1 style="color: #ffffff; font-size: 2.5rem; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.5px;">Accident Risk Zone Predictor</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; font-weight: 400; margin: 0; max-width: 800px;">Predictive Road Safety Analytics and Decision Support System using Machine Learning. Adjust situational factors to estimate hazard rates.</p>
    </div>
    """
else:
    hero_html = """
    <div class="hero-container" style="
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    ">
        <h1 style="color: #ffffff; font-size: 2.5rem; font-weight: 800; margin: 0 0 8px 0; letter-spacing: -0.5px;">Accident Risk Zone Predictor</h1>
        <p style="color: #94A3B8; font-size: 1.1rem; font-weight: 400; margin: 0; max-width: 800px;">Predictive Road Safety Analytics and Decision Support System using Machine Learning. Adjust situational factors to estimate hazard rates.</p>
    </div>
    """

st.markdown(hero_html, unsafe_allow_html=True)

# Calculate dynamic statistics from the dataset
if df is not None:
    hour_risk = df.groupby('hour')['accident_risk'].mean()
    peak_hour = hour_risk.idxmax()
    weather_risk = df.groupby('weather')['accident_risk'].mean()
    worst_weather = weather_risk.idxmax()
else:
    worst_weather = "fog"

# Sidebar System Control Panel
with st.sidebar:
    st.markdown("### System Control Panel")
    st.write(
        "This platform predicts road accident risk coefficients using a trained Random Forest Regressor "
        "optimized for environmental and traffic patterns. Adjust variables to simulate safety margins."
    )
    
    st.markdown("---")
    st.markdown("### Real-Time Insights")
    st.markdown("**Peak Danger Windows**\nAverage risk increases significantly during late evening hours (6 PM – 11 PM).")
    st.markdown(f"**Adverse Weather Penalty**\n{worst_weather.capitalize()} conditions multiply baseline risk by up to 2.5x.")
    st.markdown("**Highway Rule**\nHigh-speed highway travel compounded by low visibility triggers critical safety warnings.")

# Tab setup - Emojis removed
tab_predict, tab_simulator, tab_analytics, tab_model = st.tabs([
    "Live Risk Predictor", 
    "Scenario Simulator", 
    "Analytics and EDA Dashboard", 
    "Model Architecture and Metrics"
])

# ----------------- TAB 1: LIVE RISK PREDICTOR -----------------
with tab_predict:
    st.markdown("### Real-Time Risk Simulation")
    st.write("Modify the environmental features below to evaluate the immediate road risk score.")
    
    col_input, col_output = st.columns([1, 1], gap="large")
    
    with col_input:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### Environmental Variables")
        
        hour = st.slider("Hour of Day (24h clock)", min_value=0, max_value=23, value=14, step=1,
                         help="Hour of the day from midnight (0) to 11 PM (23).")
        
        weather = st.selectbox("Weather Condition", options=['clear', 'rain', 'fog'], index=0,
                               format_func=lambda x: x.capitalize())
        
        traffic = st.selectbox("Traffic Density", options=['low', 'medium', 'high'], index=1,
                               format_func=lambda x: x.capitalize())
        
        area_type = st.selectbox("Area Type / Road Classification", options=['urban', 'highway', 'rural'], index=0,
                                 format_func=lambda x: x.capitalize())
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_output:
        st.markdown("<div class='metric-card' style='height: 100%;'>", unsafe_allow_html=True)
        st.write("#### Prediction Summary")
        
        if model is not None:
            input_df = pd.DataFrame({
                'hour': [hour],
                'weather': [weather],
                'traffic': [traffic],
                'area_type': [area_type]
            })
            
            # Predict
            risk_score = model.predict(input_df)[0]
            risk_score = float(np.clip(risk_score, 0, 100))
            
            # Categorize
            if risk_score <= 40:
                category = "LOW"
                badge_class = "risk-low"
                badge_desc = "Safe driving conditions. Normal precaution required."
                bar_color = "#10B981"
                alert_class = "alert-low"
                alert_text = "**Safe Conditions:** Road parameters represent baseline safety. Standard speed limits and road rules apply."
            elif risk_score <= 70:
                category = "MEDIUM"
                badge_class = "risk-medium"
                badge_desc = "Elevated risk. Driver alertness must be increased."
                bar_color = "#F59E0B"
                alert_class = "alert-medium"
                alert_text = "**Caution Advised:** Environmental factors indicate elevated risk. Maintain safe distance and reduce speed."
            else:
                category = "HIGH"
                badge_class = "risk-high"
                badge_desc = "Dangerous environment. Exercise high vigilance and reduce speeds."
                bar_color = "#EF4444"
                alert_class = "alert-high"
                alert_text = "**Critical Risk Warning:** Extreme risk detected. Avoid unessential travel or implement double distance rules and safety lights."

            # Display risk score
            st.markdown(f"<h3>Risk Coefficient: <span style='color:{bar_color}; font-size:3rem;'>{risk_score:.1f}%</span></h3>", unsafe_allow_html=True)
            st.markdown(f"<span class='risk-badge {badge_class}'>{category} RISK ZONE</span>", unsafe_allow_html=True)
            st.write(f"*{badge_desc}*")
            
            # Progress bar
            st.markdown(f"""
                <div class='progress-bg'>
                    <div style='background-color: {bar_color}; width: {risk_score}%; height: 100%; 
                                transition: width 0.5s ease-in-out;'></div>
                </div>
            """, unsafe_allow_html=True)
            
            # Custom styled alert box
            st.markdown(f"<div class='alert-details {alert_class}'>{alert_text}</div>", unsafe_allow_html=True)
            st.write("---")
            
            st.markdown("##### Contextual Safety Insights:")
            
            # Rules-based messages (emojis removed)
            insights = []
            if weather == 'fog' and area_type == 'highway':
                insights.append("**Fog-Highway Compound Danger:** Speed differentials on highways in dense fog create immediate pile-up hazards. Reduce speed immediately.")
            if 18 <= hour <= 23 or 0 <= hour <= 4:
                insights.append("**Night-time Visual Fatigue:** Fatigue and headlight glare increases reaction time. Night hours represent a higher baseline risk multiplier.")
            if weather == 'rain':
                insights.append("**Hydroplaning Hazard:** Rain reduces tire grip. Increase braking distance by at least 2x.")
            if traffic == 'high' and weather == 'rain':
                insights.append("**Congestion Slip Hazard:** Slippery roads combined with dense traffic yields high rear-end collision statistics.")
            if len(insights) == 0:
                insights.append("**Optimal Environment:** Clear weather and daytime visibility yield maximum driving safety parameters.")
                
            for insight in insights:
                st.markdown(f"<div class='insight-bullet'>{insight}</div>", unsafe_allow_html=True)
        else:
            st.error("Model not loaded. Risk prediction unavailable.")
            
        st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 2: SCENARIO SIMULATOR -----------------
with tab_simulator:
    st.markdown("### Side-by-Side Environmental Comparison")
    st.write("Compare the accident risk difference between two distinct situational scenarios.")
    
    col_sc1, col_sc2 = st.columns(2)
    
    with col_sc1:
        st.markdown("<div class='metric-card' style='border-left: 4px solid #3B82F6;'>", unsafe_allow_html=True)
        st.subheader("Scenario A")
        s1_hour = st.slider("Hour (A)", min_value=0, max_value=23, value=12, key="s1_h")
        s1_weather = st.selectbox("Weather (A)", options=['clear', 'rain', 'fog'], index=0, key="s1_w")
        s1_traffic = st.selectbox("Traffic (A)", options=['low', 'medium', 'high'], index=0, key="s1_t")
        s1_area = st.selectbox("Area Type (A)", options=['urban', 'highway', 'rural'], index=1, key="s1_a")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_sc2:
        st.markdown("<div class='metric-card' style='border-left: 4px solid #EC4899;'>", unsafe_allow_html=True)
        st.subheader("Scenario B")
        s2_hour = st.slider("Hour (B)", min_value=0, max_value=23, value=22, key="s2_h")
        s2_weather = st.selectbox("Weather (B)", options=['clear', 'rain', 'fog'], index=2, key="s2_w")
        s2_traffic = st.selectbox("Traffic (B)", options=['low', 'medium', 'high'], index=2, key="s2_t")
        s2_area = st.selectbox("Area Type (B)", options=['urban', 'highway', 'rural'], index=1, key="s2_a")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Calculate predictions
    if model is not None:
        df_a = pd.DataFrame({'hour': [s1_hour], 'weather': [s1_weather], 'traffic': [s1_traffic], 'area_type': [s1_area]})
        df_b = pd.DataFrame({'hour': [s2_hour], 'weather': [s2_weather], 'traffic': [s2_traffic], 'area_type': [s2_area]})
        
        risk_a = float(np.clip(model.predict(df_a)[0], 0, 100))
        risk_b = float(np.clip(model.predict(df_b)[0], 0, 100))
        
        diff = risk_b - risk_a
        
        col_res1, col_res2, col_res3 = st.columns([1, 1, 1.2])
        
        with col_res1:
            st.metric(label="Scenario A Risk", value=f"{risk_a:.1f}%")
        with col_res2:
            st.metric(label="Scenario B Risk", value=f"{risk_b:.1f}%", delta=f"{diff:+.1f}%", delta_color="inverse")
        with col_res3:
            st.markdown("<div class='metric-card' style='padding: 15px;'>", unsafe_allow_html=True)
            if diff > 0:
                st.write(f"Scenario B has a **{abs(diff):.1f}% higher** predicted risk than Scenario A.")
            elif diff < 0:
                st.write(f"Scenario A has a **{abs(diff):.1f}% higher** predicted risk than Scenario B.")
            else:
                st.write("Both scenarios have identical predicted risk coefficients.")
            st.markdown("</div>", unsafe_allow_html=True)

# ----------------- TAB 3: ANALYTICS & EDA DASHBOARD -----------------
with tab_analytics:
    st.markdown("### Exploratory Data Analysis and Simulator Trends")
    st.write("Explore historical accident risk trends and correlations discovered during synthetic dataset analysis.")
    
    # Emojis removed from sub-tabs
    tab_dist, tab_hour, tab_env, tab_heat = st.tabs([
        "Risk Distribution", 
        "Hour-by-Hour Risks", 
        "Weather and Traffic Trends", 
        "Correlation and Heatmaps"
    ])
    
    with tab_dist:
        st.write("#### Accident Risk Score Distribution")
        st.write("The distribution shows the density of different risk levels in the simulator database. Notice the multi-modal peaks representing standard daytime risks vs severe compound risk events.")
        if os.path.exists("risk_distribution.png"):
            st.image("risk_distribution.png", caption="Frequency and Kernel Density Estimate of Accident Risk Scores", use_container_width=True)
        else:
            st.warning("risk_distribution.png plot file not found. Check training output.")
            
    with tab_hour:
        st.write("#### Average Risk score across 24 Hours")
        st.write("Our predictive insights clearly show that nighttime (6 PM to 4 AM) is characterized by elevated risk levels. There are also smaller peaks during the morning (8-9 AM) and evening (4-6 PM) rush hours.")
        if os.path.exists("risk_vs_hour.png"):
            st.image("risk_vs_hour.png", caption="Line Chart showing Hour of Day vs Average Risk Score", use_container_width=True)
            
    with tab_env:
        col_w, col_t = st.columns(2)
        with col_w:
            st.write("#### Weather Conditions Impact")
            if os.path.exists("risk_vs_weather.png"):
                st.image("risk_vs_weather.png", caption="Average Risk Score by Weather Condition", use_container_width=True)
        with col_t:
            st.write("#### Traffic Levels Impact")
            if os.path.exists("risk_vs_traffic.png"):
                st.image("risk_vs_traffic.png", caption="Accident Risk Distribution by Traffic Level", use_container_width=True)
                
    with tab_heat:
        st.write("#### Multivariable Interaction Analysis")
        col_i1, col_i2 = st.columns([1.2, 1])
        with col_i1:
            if os.path.exists("risk_area_weather_interaction.png"):
                st.image("risk_area_weather_interaction.png", caption="Risk Score across Weather & Area Type Combinations", use_container_width=True)
        with col_i2:
            if os.path.exists("correlation_heatmap.png"):
                st.image("correlation_heatmap.png", caption="Linear Correlation Heatmap of Features", use_container_width=True)

# ----------------- TAB 4: MODEL ARCHITECTURE & METRICS -----------------
with tab_model:
    st.markdown("### Machine Learning Pipeline and Metrics")
    st.write("Review the model selection, pipeline components, features, and predictive metrics.")
    
    col_met1, col_met2 = st.columns([1, 1.2])
    
    with col_met1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### Validation Performance")
        st.write("Comparison of the trained models on the test set:")
        
        st.markdown("""
        | Model Name | Root Mean Squared Error (RMSE) | R² Score (Variance Explained) |
        | :--- | :---: | :---: |
        | **Random Forest Regressor** | **9.01** | **79.8%** |
        | Linear Regression | 11.67 | 66.1% |
        """)
        
        st.write("##### Why Random Forest performed best:")
        st.markdown(
            "1. **Non-linear interactions:** Our dataset contains compound variables (e.g. Highway + Fog = super high risk). "
            "Linear regression struggles to model these without explicit interaction terms, whereas Random Forest trees naturally capture them.\n"
            "2. **Outlier robustness:** Decision trees are robust to outlier noise, yielding a lower RMSE."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col_met2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.write("#### Feature Importances")
        st.write("Feature importance computed by the Random Forest model:")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Variance-reduction Feature Importances", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown("---")
    st.write("#### Architecture Design")
    st.code("""
# Custom Pipeline Architecture
class CustomPipeline:
    def __init__(self, preprocessor, regressor):
        self.preprocessor = preprocessor
        self.regressor = regressor
        
    def fit(self, X_df, y):
        X_trans = self.preprocessor.fit_transform(X_df)
        self.regressor.fit(X_trans, y)
        return self
        
    def predict(self, X_df):
        X_trans = self.preprocessor.transform(X_df)
        return self.regressor.predict(X_trans)
    """)
    st.write("This pipeline uses CustomPreprocessor which fits category indexes to maintain column alignment consistency between training data and active user session inputs, and SimpleRandomForestRegressor implemented in pure NumPy.")
