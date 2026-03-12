"""
Kenya Food Price Inflation Tracker - Streamlit Dashboard
=========================================================

Author: Portfolio Project
Repository: https://github.com/Kaks753/food-inflation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Kenya Food Price Tracker",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Data loading with caching
@st.cache_data
def load_data():
    """Load all datasets with caching"""
    try:
        # Core datasets
        staples = pd.read_csv('data/clean/wfp_core_staples.csv', parse_dates=['date'])
        monthly = pd.read_csv('data/clean/wfp_monthly_avg.csv', parse_dates=['date'])
        maize_features = pd.read_csv('data/clean/maize_features.csv', parse_dates=['date'])
        
        # Model outputs (if they exist)
        try:
            prophet_fc = pd.read_csv('models/trained/prophet_forecast.csv', parse_dates=['ds'])
            model_comparison = pd.read_csv('models/trained/model_comparison.csv', index_col=0)
        except:
            prophet_fc = None
            model_comparison = None
        
        return staples, monthly, maize_features, prophet_fc, model_comparison
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Load data
staples, monthly, maize_features, prophet_fc, model_comparison = load_data()

# Sidebar
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", [
    "📊 Overview",
    "📈 Price Trends",
    "🗺️ Regional Analysis",
    "💰 Inflation Calculator",
    "🔮 Forecasts",
    "📝 Insights"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info("""
**Kenya Food Price Inflation Tracker**

Monitoring and forecasting food prices across Kenya using WFP data (2006-2021).

**Data**: 8,884 records, 34 commodities, 45 markets

**Models**: Prophet, SARIMA, XGBoost
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### Links")
st.sidebar.markdown("[📂 GitHub Repository](https://github.com/Kaks753/food-inflation)")
st.sidebar.markdown("[👤 Portfolio](https://muemastephenportfolio.netlify.app/)")

# Main content
if page == "📊 Overview":
    st.markdown("<h1 class='main-header'>📊 Kenya Food Price Inflation Tracker</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Welcome to the Kenya Food Price Inflation Tracker
    
    This dashboard provides comprehensive insights into food price dynamics in Kenya, focusing on staple commodities
    that impact millions of Kenyans daily.
    """)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📅 Data Period",
            value="2006-2021",
            delta="15 years"
        )
    
    with col2:
        st.metric(
            label="🌾 Commodities",
            value=staples['mp_commodityname'].nunique(),
            delta=None
        )
    
    with col3:
        st.metric(
            label="📍 Markets",
            value=staples['mkt_name'].nunique(),
            delta=None
        )
    
    with col4:
        st.metric(
            label="📊 Records",
            value=f"{len(staples):,}",
            delta=None
        )
    
    # Dataset overview
    st.markdown("<h2 class='sub-header'>Dataset Overview</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Commodity distribution
        commodity_counts = staples['mp_commodityname'].value_counts().head(8)
        fig = px.bar(
            x=commodity_counts.values,
            y=commodity_counts.index,
            orientation='h',
            title='Top Commodities by Record Count',
            labels={'x': 'Number of Records', 'y': 'Commodity'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Regional distribution
        region_counts = staples['adm1_name'].value_counts()
        fig = px.pie(
            values=region_counts.values,
            names=region_counts.index,
            title='Records by Region'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Quick stats
    st.markdown("<h2 class='sub-header'>Key Statistics</h2>", unsafe_allow_html=True)
    
    maize_data = staples[staples['mp_commodityname'] == 'Maize (white) - Retail']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="🌽 Avg Maize Price",
            value=f"{maize_data['mp_price'].mean():.2f} KES/kg"
        )
    
    with col2:
        st.metric(
            label="📈 Max Maize Price",
            value=f"{maize_data['mp_price'].max():.2f} KES/kg",
            delta=f"+{((maize_data['mp_price'].max() / maize_data['mp_price'].min() - 1) * 100):.0f}%"
        )
    
    with col3:
        st.metric(
            label="📉 Min Maize Price",
            value=f"{maize_data['mp_price'].min():.2f} KES/kg"
        )

elif page == "📈 Price Trends":
    st.markdown("<h1 class='main-header'>📈 Price Trends Analysis</h1>", unsafe_allow_html=True)
    
    # Commodity selector
    commodities = sorted(staples['mp_commodityname'].unique())
    selected_commodity = st.selectbox("Select Commodity", commodities, index=0)
    
    # Filter data
    commodity_data = staples[staples['mp_commodityname'] == selected_commodity]
    
    # National average time series
    national_avg = commodity_data.groupby('date')['mp_price'].mean().reset_index()
    national_avg.columns = ['date', 'price']
    
    # Time series plot
    fig = px.line(
        national_avg,
        x='date',
        y='price',
        title=f'{selected_commodity} - National Average Price Over Time',
        labels={'price': 'Price (KES)', 'date': 'Date'}
    )
    fig.add_hline(
        y=national_avg['price'].mean(),
        line_dash='dash',
        annotation_text=f'Mean: {national_avg["price"].mean():.2f} KES',
        line_color='red'
    )
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Price", f"{national_avg['price'].mean():.2f} KES")
    
    with col2:
        st.metric("Median Price", f"{national_avg['price'].median():.2f} KES")
    
    with col3:
        st.metric("Std Dev", f"{national_avg['price'].std():.2f} KES")
    
    with col4:
        cv = (national_avg['price'].std() / national_avg['price'].mean()) * 100
        st.metric("Volatility (CV)", f"{cv:.1f}%")
    
    # Seasonal pattern
    st.markdown("<h2 class='sub-header'>Seasonal Patterns</h2>", unsafe_allow_html=True)
    
    national_avg['month'] = national_avg['date'].dt.month
    monthly_avg = national_avg.groupby('month')['price'].mean().reset_index()
    
    fig = px.line(
        monthly_avg,
        x='month',
        y='price',
        title='Average Price by Month (Across All Years)',
        labels={'price': 'Average Price (KES)', 'month': 'Month'},
        markers=True
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Year-over-year comparison
    st.markdown("<h2 class='sub-header'>Year-over-Year Comparison</h2>", unsafe_allow_html=True)
    
    national_avg['year'] = national_avg['date'].dt.year
    recent_years = national_avg[national_avg['year'] >= 2015]
    
    fig = px.line(
        recent_years,
        x='month',
        y='price',
        color='year',
        title='Price Comparison by Year (2015-2021)',
        labels={'price': 'Price (KES)', 'month': 'Month', 'year': 'Year'}
    )
    fig.update_xaxes(tickmode='linear', tick0=1, dtick=1)
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

elif page == "🗺️ Regional Analysis":
    st.markdown("<h1 class='main-header'>🗺️ Regional Price Analysis</h1>", unsafe_allow_html=True)
    
    # Commodity selector
    commodities = sorted(staples['mp_commodityname'].unique())
    selected_commodity = st.selectbox("Select Commodity", commodities, index=0)
    
    commodity_data = staples[staples['mp_commodityname'] == selected_commodity]
    
    # Regional average prices
    st.markdown("<h2 class='sub-header'>Average Prices by Region</h2>", unsafe_allow_html=True)
    
    regional_avg = commodity_data.groupby('adm1_name')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
    regional_avg.columns = ['Region', 'Average Price', 'Std Dev', 'Observations']
    regional_avg = regional_avg.sort_values('Average Price', ascending=False)
    
    fig = px.bar(
        regional_avg,
        x='Region',
        y='Average Price',
        error_y='Std Dev',
        title=f'{selected_commodity} - Regional Price Comparison',
        labels={'Average Price': 'Average Price (KES)', 'Region': 'Region'}
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    # Regional time series
    st.markdown("<h2 class='sub-header'>Regional Price Trends</h2>", unsafe_allow_html=True)
    
    regional_ts = commodity_data.groupby(['date', 'adm1_name'])['mp_price'].mean().reset_index()
    regional_ts.columns = ['date', 'region', 'price']
    
    fig = px.line(
        regional_ts,
        x='date',
        y='price',
        color='region',
        title=f'{selected_commodity} - Price Trends by Region',
        labels={'price': 'Price (KES)', 'date': 'Date', 'region': 'Region'}
    )
    fig.update_layout(height=500, hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
    
    # Market-level details
    st.markdown("<h2 class='sub-header'>Top Markets</h2>", unsafe_allow_html=True)
    
    market_avg = commodity_data.groupby('mkt_name')['mp_price'].agg(['mean', 'count']).reset_index()
    market_avg.columns = ['Market', 'Average Price', 'Observations']
    market_avg = market_avg.sort_values('Average Price', ascending=False).head(10)
    
    st.dataframe(market_avg, use_container_width=True, height=400)

elif page == "💰 Inflation Calculator":
    st.markdown("<h1 class='main-header'>💰 Inflation Calculator</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ### Calculate Food Price Inflation
    
    See how food prices have changed over time and calculate the purchasing power impact.
    """)
    
    # Maize inflation (most tracked commodity)
    maize_data = staples[staples['mp_commodityname'] == 'Maize (white) - Retail']
    maize_monthly = maize_data.groupby('date')['mp_price'].mean().reset_index()
    maize_monthly.columns = ['date', 'price']
    maize_monthly = maize_monthly.sort_values('date')
    
    # Calculate YoY inflation
    maize_monthly['price_lag_12m'] = maize_monthly['price'].shift(12)
    maize_monthly['inflation_yoy'] = ((maize_monthly['price'] - maize_monthly['price_lag_12m']) / maize_monthly['price_lag_12m']) * 100
    
    # Plot inflation rate
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=maize_monthly['date'],
        y=maize_monthly['inflation_yoy'],
        mode='lines',
        name='YoY Inflation',
        line=dict(color='royalblue')
    ))
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(
        title='Maize Price - Year-over-Year Inflation Rate',
        xaxis_title='Date',
        yaxis_title='Inflation Rate (%)',
        height=500,
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Interactive calculator
    st.markdown("<h2 class='sub-header'>Custom Calculation</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        start_year = st.selectbox("Start Year", list(range(2006, 2022)), index=0)
    
    with col2:
        end_year = st.selectbox("End Year", list(range(2006, 2022)), index=15)
    
    if end_year > start_year:
        start_price = maize_monthly[maize_monthly['date'].dt.year == start_year]['price'].mean()
        end_price = maize_monthly[maize_monthly['date'].dt.year == end_year]['price'].mean()
        inflation = ((end_price - start_price) / start_price) * 100
        years = end_year - start_year
        cagr = ((end_price / start_price) ** (1/years) - 1) * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Start Price", f"{start_price:.2f} KES")
        
        with col2:
            st.metric("End Price", f"{end_price:.2f} KES")
        
        with col3:
            st.metric("Total Inflation", f"{inflation:.1f}%")
        
        with col4:
            st.metric("CAGR", f"{cagr:.1f}%")
        
        st.info(f"""
        **Interpretation**: Between {start_year} and {end_year}, maize prices increased by {inflation:.1f}%, 
        representing an average annual growth rate (CAGR) of {cagr:.1f}%. This means that what cost 
        {start_price:.2f} KES in {start_year} cost {end_price:.2f} KES in {end_year}.
        """)
    else:
        st.warning("End year must be after start year.")

elif page == "🔮 Forecasts":
    st.markdown("<h1 class='main-header'>🔮 Price Forecasts</h1>", unsafe_allow_html=True)
    
    if prophet_fc is not None and model_comparison is not None:
        st.markdown("""
        ### 12-Month Forecast for Maize Prices
        
        Based on historical data and machine learning models (Prophet, SARIMA, XGBoost).
        """)
        
        # Model comparison
        st.markdown("<h2 class='sub-header'>Model Performance</h2>", unsafe_allow_html=True)
        
        st.dataframe(model_comparison.round(2), use_container_width=True)
        
        best_model = model_comparison['MAPE'].idxmin()
        st.success(f"✅ Best Model: **{best_model}** (MAPE: {model_comparison.loc[best_model, 'MAPE']:.2f}%)")
        
        # Forecast visualization
        st.markdown("<h2 class='sub-header'>Forecast Visualization</h2>", unsafe_allow_html=True)
        
        # Use Prophet forecast for visualization
        historical = maize_features[['date', 'price']].rename(columns={'date': 'ds', 'price': 'y'})
        forecast_future = prophet_fc[prophet_fc['ds'] > historical['ds'].max()]
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=historical['ds'],
            y=historical['y'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'],
            y=forecast_future['yhat'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_future['ds'].tolist() + forecast_future['ds'].tolist()[::-1],
            y=forecast_future['yhat_upper'].tolist() + forecast_future['yhat_lower'].tolist()[::-1],
            fill='toself',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Interval'
        ))
        
        fig.update_layout(
            title='Maize Price Forecast (12 Months)',
            xaxis_title='Date',
            yaxis_title='Price (KES/kg)',
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Forecast table
        st.markdown("<h2 class='sub-header'>Forecast Values</h2>", unsafe_allow_html=True)
        
        forecast_table = forecast_future[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
        forecast_table.columns = ['Date', 'Predicted Price', 'Lower Bound', 'Upper Bound']
        forecast_table['Date'] = forecast_table['Date'].dt.strftime('%Y-%m')
        
        st.dataframe(forecast_table.round(2), use_container_width=True)
        
    else:
        st.warning("⚠️ Forecast models have not been trained yet. Please run the modeling notebooks first.")
        st.info("""
        **To generate forecasts:**
        1. Run `notebooks/04a_prophet_forecasting.ipynb`
        2. Run `notebooks/04b_sarima_forecasting.ipynb`
        3. Run `notebooks/04c_ml_forecasting.ipynb`
        4. Run `notebooks/04d_model_comparison.ipynb`
        5. Refresh this dashboard
        """)

elif page == "📝 Insights":
    st.markdown("<h1 class='main-header'>📝 Key Insights & Recommendations</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    ## 🔍 Key Findings
    
    ### 1. Price Trends (2006-2021)
    - 📈 Maize prices increased **~45%** over 15 years (CAGR: 2.5%)
    - 💸 Minimum: **6 KES/kg** (2006)
    - 💰 Maximum: **100 KES/kg** (2017 drought)
    - 📊 Average: **48 KES/kg**
    
    ### 2. Volatility
    - ⚡ Coefficient of variation: **~35%**
    - 🌪️ Highest price spikes during droughts (2011, 2017)
    - 🛢️ Imported commodities (rice, oil) show more stability
    
    ### 3. Seasonality
    - 🌾 **Harvest months** (Oct-Dec): 15-20% lower prices
    - 📉 **Lean season** (Feb-May): 15-20% higher prices
    - 📐 Peak-to-trough swing: **15-20 KES/kg**
    
    ### 4. Regional Variation
    - 🏙️ **Nairobi & Coast**: Higher prices (50-55 KES/kg avg)
    - 🌾 **Rift Valley**: Lower prices (40-45 KES/kg avg)
    - 📏 Price gap: **20-30 KES/kg** between regions
    
    ---
    
    ## 💡 Recommendations
    
    ### For Policymakers 🏛️
    1. **Strategic Grain Reserves**: Build 3-6 month buffer stocks
    2. **Market Integration**: Improve transport infrastructure
    3. **Early Warning System**: Implement price monitoring dashboard
    4. **Subsidy Timing**: Target lean season (Feb-May) for maximum impact
    5. **Regional Balance**: Facilitate inter-regional trade
    
    ### For Farmers 👨‍🌾
    1. **Harvest Timing**: Store and sell during lean season (+15-20% premium)
    2. **Crop Diversification**: Reduce exposure to price volatility
    3. **Market Information**: Use price forecasts for decisions
    4. **Cooperatives**: Form groups to reduce transaction costs
    
    ### For Consumers 🏘️
    1. **Bulk Buying**: Purchase during harvest (15-20% savings)
    2. **Storage**: Invest in proper storage facilities
    3. **Alternatives**: Consider substitute staples
    4. **Budget Planning**: Expect +3-7% annual inflation
    
    ### For Traders & Businesses 🏢
    1. **Arbitrage**: Exploit regional price differences
    2. **Inventory Management**: Stock during harvest
    3. **Hedging**: Use forecasts to manage risk
    4. **Supply Chain**: Optimize logistics
    
    ---
    
    ## ⚠️ Limitations
    
    - 📅 Data ends in 2021 (need 2022-2024 updates)
    - 🌡️ Limited external factors (weather, fuel, exchange rates)
    - 📍 National aggregates mask local variations
    - 🌽 Single commodity focus (maize only)
    
    ---
    
    ## 🚀 Future Work
    
    1. **Additional Data**: Weather, fuel prices, USD/KES rates
    2. **Multi-Commodity**: Expand to food basket index
    3. **Geospatial Analysis**: Interactive regional heatmaps
    4. **Real-Time Updates**: Automate data collection
    5. **Advanced Models**: LSTM, VAR, probabilistic forecasts
    6. **Mobile App**: SMS-based price alerts
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Kenya Food Price Inflation Tracker | Built with Streamlit</p>
    <p>Data Source: World Food Programme (WFP) | 2006-2021</p>
</div>
""", unsafe_allow_html=True)
