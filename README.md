# ESG Profit Predictor

Predict company profit margins using Environmental, Social, and Governance (ESG) metrics and financial data. This project provides an interactive Streamlit app and a Jupyter notebook for data exploration, modeling, and batch prediction.

## About The Project

This project explores the relationship between a company's ESG performance and its financial profitability. It features:
- A Streamlit web application for interactive profit margin prediction
- A Jupyter Notebook for exploratory data analysis, feature engineering, and model evaluation
- Batch prediction capability for multiple companies

The goal is to provide a user-friendly platform for stakeholders, investors, and analysts to assess the financial implications of ESG strategies using machine learning.

## Key Features

- **Interactive Profit Margin Prediction**
  - Select Random Forest or Linear Regression models
  - Input company financials and ESG scores for instant prediction
  - Qualitative profitability assessment (Exceptional, Strong, Moderate, etc.)
- **Data Analysis & Visualization**
  - ESG vs Profit scatter plots
  - Industry performance bar charts
  - ESG component correlation heatmaps
  - Distribution histograms
- **Model Performance Evaluation**
  - Compare Random Forest and Linear Regression (R², MAE, RMSE)
  - Feature importance chart for Random Forest
- **Batch Prediction**
  - Upload CSV for batch profit margin predictions
  - Downloadable results
- **Demo Dataset**
  - Pre-loaded demo data for instant exploration

## Technologies Used

- Python
- Streamlit
- Pandas
- Scikit-learn
- Matplotlib & Seaborn
- Jupyter Notebook

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/esg-profit-predictor.git
   cd esg-profit-predictor
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install streamlit pandas numpy matplotlib seaborn scikit-learn
   ```
4. Run the Streamlit app:
   ```bash
   streamlit run .venv/esg_app.py
   ```
5. Open `esg.ipynb` in Jupyter or VS Code for notebook-based analysis.

## File Descriptions

- `.venv/esg_app.py`: Main Streamlit app
- `esg.ipynb`: Jupyter notebook for data exploration and modeling
- `company_esg_financial_dataset.csv`: Example dataset (optional)

## Data Analysis Insights

- **Industry Performance:** Finance and Technology sectors show higher average ESG scores
- **Regional Differences:** European companies tend to have higher ESG scores
- **ESG Component Correlations:** Social and Governance pillars are more strongly correlated with overall ESG scores

## Model Performance

- Random Forest outperforms Linear Regression in profit margin prediction (higher R², lower error)
- MarketCap and Revenue are the most significant predictors, followed by ESG metrics

## Future Work

- Explore advanced models (XGBoost, LightGBM)
- Expand dataset with more companies and granular ESG data
- Add time-series analysis of ESG scores
- Deploy the app to Heroku, AWS, or similar platforms

## Credits

This project was created by Bayarmaa T.
