import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="ESG Profit Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f8ff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #4682b4;
        margin: 0.5rem 0;
    }
    .prediction-result {
        background-color: #e8f5e8;
        padding: 2rem;
        border-radius: 15px;
        border: 3px solid #32cd32;
        text-align: center;
        margin: 1rem 0;
    }
    .sidebar-header {
        color: #2E8B57;
        font-weight: bold;
    }
    .data-card {
        background-color: #f9f9f9;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header"> ESG Profit Predictor</h1>', unsafe_allow_html=True)
st.markdown("### *Predict company profitability using Environmental, Social & Governance metrics*")
st.markdown("---")

st.sidebar.markdown('<p class="sidebar-header"> Configuration</p>', unsafe_allow_html=True)

model_choice = st.sidebar.radio(
    "Choose Prediction Model:",
    ["Random Forest", "Linear Regression"],
    help="Random Forest is more accurate for non-linear relationships"
)

st.sidebar.markdown('<p class="sidebar-header"> Data Source</p>', unsafe_allow_html=True)
use_demo_data = st.sidebar.checkbox("Use Demo Dataset", value=True)

@st.cache_data
def generate_demo_data():
    np.random.seed(42)
    n_samples = 1000
    
    industries = ['Technology', 'Healthcare', 'Finance', 'Energy', 'Manufacturing', 'Retail', 'Utilities']
    regions = ['North America', 'Europe', 'Asia Pacific', 'Latin America', 'Africa']
    
    data = []
    for i in range(n_samples):
        company_data = {
            'CompanyID': i + 1,
            'CompanyName': f'Company_{i+1:03d}',
            'Industry': np.random.choice(industries),
            'Region': np.random.choice(regions),
            'Year': np.random.choice([2020, 2021, 2022, 2023, 2024]),
            
            'Revenue_M': np.random.lognormal(mean=6, sigma=2),  
            'MarketCap_M': np.random.lognormal(mean=7, sigma=2.5),  
            'Employees': np.random.randint(100, 50000),
            'GrowthRate': np.random.normal(8, 12),  
            
            'ESG_Environmental': max(0, min(100, np.random.normal(55, 20))),
            'ESG_Social': max(0, min(100, np.random.normal(60, 18))),
            'ESG_Governance': max(0, min(100, np.random.normal(58, 22))),
            
            'CO2_Emissions_tons': np.random.lognormal(8, 1.5),
            'Water_Usage_m3': np.random.lognormal(10, 2),
            'Energy_Consumption_MWh': np.random.lognormal(12, 1.8),
            'Renewable_Energy_Pct': max(0, min(100, np.random.normal(25, 20))),
            
            'Employee_Satisfaction': max(0, min(100, np.random.normal(70, 15))),
            'Diversity_Index': max(0, min(100, np.random.normal(60, 20))),
            'Safety_Score': max(0, min(100, np.random.normal(75, 18))),
            
            'Board_Independence': max(0, min(100, np.random.normal(65, 20))),
            'Ethics_Score': max(0, min(100, np.random.normal(70, 18))),
            'Transparency_Score': max(0, min(100, np.random.normal(68, 22)))
        }
        
        company_data['ESG_Overall'] = (
            company_data['ESG_Environmental'] * 0.33 +
            company_data['ESG_Social'] * 0.33 +
            company_data['ESG_Governance'] * 0.34
        )
        
        base_profit = (
            company_data['ESG_Overall'] * 0.15 +
            company_data['GrowthRate'] * 0.3 +
            np.log(company_data['Revenue_M']) * 2 +
            np.random.normal(0, 8)
        )
        
        industry_multipliers = {
            'Technology': 1.3, 'Healthcare': 1.2, 'Finance': 1.1,
            'Energy': 0.8, 'Manufacturing': 0.9, 'Retail': 0.85, 'Utilities': 0.95
        }
        
        company_data['ProfitMargin'] = max(-25, min(45, 
            base_profit * industry_multipliers.get(company_data['Industry'], 1.0)
        ))
        
        data.append(company_data)
    
    return pd.DataFrame(data)

if use_demo_data:
    df = generate_demo_data()
    st.sidebar.success(f" Demo dataset loaded: {len(df)} companies")
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success(f" File uploaded: {len(df)} rows")
        except Exception as e:
            st.sidebar.error(f" Error: {e}")
            st.stop()
    else:
        st.warning(" Please upload a CSV file or use demo data")
        st.stop()

def prepare_features(data):
    """Prepare features for modeling"""
    df_prep = data.copy()
    
    df_prep['ESG_Balance'] = 100 - df_prep[['ESG_Environmental', 'ESG_Social', 'ESG_Governance']].std(axis=1)
    df_prep['Financial_Health'] = np.log(df_prep['Revenue_M']) + np.log(df_prep['MarketCap_M'])
    df_prep['Sustainability_Index'] = (df_prep['ESG_Environmental'] + df_prep['Renewable_Energy_Pct']) / 2
    
    le_industry = LabelEncoder()
    le_region = LabelEncoder()
    
    df_prep['Industry_Code'] = le_industry.fit_transform(df_prep['Industry'])
    df_prep['Region_Code'] = le_region.fit_transform(df_prep['Region'])
    
    return df_prep, le_industry, le_region

df_processed, industry_encoder, region_encoder = prepare_features(df)

feature_columns = [
    'ESG_Overall', 'ESG_Environmental', 'ESG_Social', 'ESG_Governance',
    'Revenue_M', 'MarketCap_M', 'GrowthRate', 'Employees',
    'ESG_Balance', 'Financial_Health', 'Sustainability_Index',
    'Employee_Satisfaction', 'Board_Independence', 'Renewable_Energy_Pct',
    'Industry_Code', 'Region_Code', 'Year'
]

tab1, tab2, tab3, tab4 = st.tabs([" Predict", " Analyze", " Performance", " Batch Process"])

with tab1:
    st.header(" Individual Company Prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader(" Company Info")
        company_name = st.text_input("Company Name", "NewCorp Inc.")
        industry = st.selectbox("Industry", df['Industry'].unique())
        region = st.selectbox("Region", df['Region'].unique())
        year = st.selectbox("Year", [2024, 2023, 2022, 2021, 2020])
    
    with col2:
        st.subheader(" Financial Data")
        revenue = st.number_input("Revenue (Million $)", 1.0, 100000.0, 1000.0)
        market_cap = st.number_input("Market Cap (Million $)", 1.0, 500000.0, 5000.0)
        employees = st.number_input("Number of Employees", 10, 200000, 5000)
        growth_rate = st.slider("Growth Rate (%)", -30.0, 50.0, 8.0)
    
    with col3:
        st.subheader(" ESG Metrics")
        esg_env = st.slider("Environmental Score", 0, 100, 60)
        esg_social = st.slider("Social Score", 0, 100, 65)
        esg_gov = st.slider("Governance Score", 0, 100, 70)
        renewable_pct = st.slider("Renewable Energy %", 0, 100, 30)
    
    col1, col2 = st.columns(2)
    with col1:
        employee_sat = st.slider("Employee Satisfaction", 0, 100, 75)
    with col2:
        board_indep = st.slider("Board Independence", 0, 100, 65)
    
    if st.button(" Predict Profit Margin", type="primary"):
        esg_overall = (esg_env * 0.33 + esg_social * 0.33 + esg_gov * 0.34)
        
        input_data = pd.DataFrame({
            'CompanyName': [company_name],
            'Industry': [industry],
            'Region': [region],
            'Year': [year],
            'Revenue_M': [revenue],
            'MarketCap_M': [market_cap],
            'Employees': [employees],
            'GrowthRate': [growth_rate],
            'ESG_Overall': [esg_overall],
            'ESG_Environmental': [esg_env],
            'ESG_Social': [esg_social],
            'ESG_Governance': [esg_gov],
            'Renewable_Energy_Pct': [renewable_pct],
            'Employee_Satisfaction': [employee_sat],
            'Board_Independence': [board_indep]
        })
        
        input_processed, _, _ = prepare_features(input_data)
        
        X = df_processed[feature_columns].fillna(df_processed[feature_columns].median())
        y = df_processed['ProfitMargin']
        
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
        else:
            model = LinearRegression()
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        model.fit(X, y)
        
        X_input = input_processed[feature_columns].fillna(df_processed[feature_columns].median())
        if model_choice == "Linear Regression":
            X_input = scaler.transform(X_input)
        
        prediction = model.predict(X_input)[0]
        
        st.markdown("---")
        st.markdown(f"""
        <div class="prediction-result">
            <h2> Predicted Profit Margin</h2>
            <h1 style="color: #228B22; font-size: 4rem;">{prediction:.1f}%</h1>
            <h3>for {company_name}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if prediction > 20:
                st.success(" **Exceptional**")
            elif prediction > 12:
                st.success(" **Strong**")
            elif prediction > 5:
                st.info(" **Moderate**")
            elif prediction > 0:
                st.warning(" **Weak**")
            else:
                st.error(" **Loss**")
        
        with col2:
            industry_avg = df[df['Industry'] == industry]['ProfitMargin'].mean()
            delta = prediction - industry_avg
            st.metric("vs Industry", f"{delta:+.1f}%", f"{delta:.1f}%")
        
        with col3:
            market_avg = df['ProfitMargin'].mean()
            delta_market = prediction - market_avg
            st.metric("vs Market", f"{delta_market:+.1f}%", f"{delta_market:.1f}%")
        
        with col4:
            esg_impact = (esg_overall - 50) * 0.2  
            st.metric("ESG Impact", f"~{esg_impact:+.1f}%", f"{esg_impact:.1f}%")

with tab2:
    st.header(" ESG-Profit Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Companies", f"{len(df):,}")
    with col2:
        st.metric("Avg Profit", f"{df['ProfitMargin'].mean():.1f}%")
    with col3:
        st.metric("Avg ESG Score", f"{df['ESG_Overall'].mean():.1f}")
    with col4:
        correlation = df['ESG_Overall'].corr(df['ProfitMargin'])
        st.metric("ESG-Profit Correlation", f"{correlation:.3f}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("ESG vs Profit Margin")
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(df['ESG_Overall'], df['ProfitMargin'], 
                           c=df['Revenue_M'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('ESG Overall Score')
        ax.set_ylabel('Profit Margin (%)')
        ax.set_title('ESG Score vs Profit Margin')
        plt.colorbar(scatter, label='Revenue (Million $)')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Industry Performance")
        fig, ax = plt.subplots(figsize=(10, 6))
        industry_profit = df.groupby('Industry')['ProfitMargin'].mean().sort_values(ascending=True)
        industry_profit.plot(kind='barh', ax=ax, color='skyblue')
        ax.set_xlabel('Average Profit Margin (%)')
        ax.set_title('Average Profit Margin by Industry')
        st.pyplot(fig)
    
    st.subheader(" ESG Components Correlation")
    fig, ax = plt.subplots(figsize=(12, 8))
    corr_cols = ['ESG_Environmental', 'ESG_Social', 'ESG_Governance', 'ESG_Overall', 'ProfitMargin']
    correlation_matrix = df[corr_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu_r', center=0, ax=ax)
    ax.set_title('ESG Components Correlation Matrix')
    st.pyplot(fig)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Profit Margin Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['ProfitMargin'], bins=30, alpha=0.7, color='lightgreen')
        ax.axvline(df['ProfitMargin'].mean(), color='red', linestyle='--', label=f'Mean: {df["ProfitMargin"].mean():.1f}%')
        ax.set_xlabel('Profit Margin (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Profit Margins')
        ax.legend()
        st.pyplot(fig)
    
    with col2:
        st.subheader("ESG Score Distribution")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df['ESG_Overall'], bins=30, alpha=0.7, color='lightblue')
        ax.axvline(df['ESG_Overall'].mean(), color='red', linestyle='--', label=f'Mean: {df["ESG_Overall"].mean():.1f}')
        ax.set_xlabel('ESG Overall Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of ESG Scores')
        ax.legend()
        st.pyplot(fig)

with tab3:
    st.header(" Model Performance Evaluation")
    
    X = df_processed[feature_columns].fillna(df_processed[feature_columns].median())
    y = df_processed['ProfitMargin']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    lr_model = LinearRegression()
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    rf_model.fit(X_train, y_train)
    lr_model.fit(X_train_scaled, y_train)
    
    rf_pred = rf_model.predict(X_test)
    lr_pred = lr_model.predict(X_test_scaled)
    
    rf_r2 = r2_score(y_test, rf_pred)
    lr_r2 = r2_score(y_test, lr_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    lr_mae = mean_absolute_error(y_test, lr_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
    lr_rmse = np.sqrt(mean_squared_error(y_test, lr_pred))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(" Random Forest")
        st.metric("R² Score", f"{rf_r2:.3f}")
        st.metric("MAE", f"{rf_mae:.2f}%")
        st.metric("RMSE", f"{rf_rmse:.2f}%")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        top_features = feature_importance.head(10)
        ax.barh(range(len(top_features)), top_features['Importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Feature Importance (Random Forest)')
        st.pyplot(fig)
    
    with col2:
        st.subheader(" Linear Regression")
        st.metric("R² Score", f"{lr_r2:.3f}")
        st.metric("MAE", f"{lr_mae:.2f}%")
        st.metric("RMSE", f"{lr_rmse:.2f}%")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_test, rf_pred, alpha=0.5, label='Random Forest')
        ax.scatter(y_test, lr_pred, alpha=0.5, label='Linear Regression')
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_xlabel('Actual Profit Margin (%)')
        ax.set_ylabel('Predicted Profit Margin (%)')
        ax.set_title('Predicted vs Actual Values')
        ax.legend()
        st.pyplot(fig)
    
    st.subheader(" Model Comparison")
    comparison_df = pd.DataFrame({
        'Model': ['Random Forest', 'Linear Regression'],
        'R² Score': [rf_r2, lr_r2],
        'MAE': [rf_mae, lr_mae],
        'RMSE': [rf_rmse, lr_rmse]
    })
    st.dataframe(comparison_df, use_container_width=True)

with tab4:
    st.header(" Batch Prediction Processing")
    
    st.markdown("""
    ### Upload CSV File for Batch Predictions
    Your CSV file should contain the following columns:
    - CompanyName, Industry, Region, Year
    - Revenue_M, MarketCap_M, Employees, GrowthRate
    - ESG_Environmental, ESG_Social, ESG_Governance
    - Renewable_Energy_Pct, Employee_Satisfaction, Board_Independence
    """)
    
    batch_file = st.file_uploader("Choose CSV file", type=['csv'], key='batch_upload')
    
    if batch_file:
        try:
            batch_df = pd.read_csv(batch_file)
            st.success(f" File uploaded successfully! {len(batch_df)} rows detected.")
            
            st.subheader(" Data Preview")
            st.dataframe(batch_df.head())
            
            if st.button(" Process Batch Predictions", type="primary"):
                with st.spinner("Processing predictions..."):
                    batch_processed, _, _ = prepare_features(batch_df)
                    
                    X = df_processed[feature_columns].fillna(df_processed[feature_columns].median())
                    y = df_processed['ProfitMargin']
                    
                    if model_choice == "Random Forest":
                        model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
                        model.fit(X, y)
                        X_batch = batch_processed[feature_columns].fillna(df_processed[feature_columns].median())
                        predictions = model.predict(X_batch)
                    else:
                        model = LinearRegression()
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X)
                        model.fit(X_scaled, y)
                        X_batch = batch_processed[feature_columns].fillna(df_processed[feature_columns].median())
                        X_batch_scaled = scaler.transform(X_batch)
                        predictions = model.predict(X_batch_scaled)
                    
                    results_df = batch_df.copy()
                    results_df['Predicted_ProfitMargin'] = predictions
                    results_df['Prediction_Category'] = pd.cut(
                        predictions, 
                        bins=[-np.inf, 0, 5, 12, 20, np.inf],
                        labels=['Loss', 'Weak', 'Moderate', 'Strong', 'Exceptional']
                    )
                    
                    st.subheader(" Prediction Results")
                    st.dataframe(results_df)
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Average Prediction", f"{predictions.mean():.1f}%")
                    with col2:
                        st.metric("Best Performance", f"{predictions.max():.1f}%")
                    with col3:
                        st.metric("Worst Performance", f"{predictions.min():.1f}%")
                    with col4:
                        profitable = (predictions > 0).sum()
                        st.metric("Profitable Companies", f"{profitable}/{len(predictions)}")
                    
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label=" Download Results CSV",
                        data=csv,
                        file_name="esg_profit_predictions.csv",
                        mime="text/csv"
                    )
                    
                    st.subheader(" Results Distribution")
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    ax1.hist(predictions, bins=20, alpha=0.7, color='lightgreen')
                    ax1.set_xlabel('Predicted Profit Margin (%)')
                    ax1.set_ylabel('Frequency')
                    ax1.set_title('Distribution of Predictions')
                    ax1.axvline(predictions.mean(), color='red', linestyle='--', 
                               label=f'Mean: {predictions.mean():.1f}%')
                    ax1.legend()
                    
                    category_counts = results_df['Prediction_Category'].value_counts()
                    ax2.bar(category_counts.index, category_counts.values, color='skyblue')
                    ax2.set_xlabel('Prediction Category')
                    ax2.set_ylabel('Count')
                    ax2.set_title('Predictions by Category')
                    ax2.tick_params(axis='x', rotation=45)
                    
                    st.pyplot(fig)
                    
        except Exception as e:
            st.error(f" Error processing file: {str(e)}")
    else:
        st.subheader(" Sample Data Format")
        sample_data = df[['CompanyName', 'Industry', 'Region', 'Revenue_M', 'ESG_Environmental', 
                         'ESG_Social', 'ESG_Governance', 'GrowthRate']].head(3)
        st.dataframe(sample_data)

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><strong> ESG Profit Predictor</strong> | By Bayarmaa T.</p>
    <p><em>Predicting sustainable profitability through ESG metrics</em></p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("---")
    st.markdown("###  Dataset Info")
    st.write(f"**Total Companies:** {len(df):,}")
    st.write(f"**Industries:** {df['Industry'].nunique()}")
    st.write(f"**Regions:** {df['Region'].nunique()}")
    st.write(f"**Year Range:** {df['Year'].min()} - {df['Year'].max()}")
    
    st.markdown("###  Model Info")
    st.write(f"**Active Model:** {model_choice}")
    st.write(f"**Features Used:** {len(feature_columns)}")
    
    if st.checkbox("Show Feature List"):
        for i, feature in enumerate(feature_columns, 1):
            st.write(f"{i}. {feature}")