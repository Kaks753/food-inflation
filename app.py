"""
Kenya Food Price Inflation Tracker - Advanced Interactive Dashboard
=====================================================================

Author: Stephen Muema
Portfolio: https://muemastephenportfolio.netlify.app/
Repository: https://github.com/Kaks753/food-inflation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Kenya Food Price Intelligence",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS (cleaned of any problematic characters)
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.8rem;
        color: #e74c3c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 3px solid #e74c3c;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        color: white;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #ecf0f1;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f39c12;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #27ae60;
        margin: 1rem 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ecf0f1;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3498db;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Data loading with caching and cleaning
@st.cache_data
def load_data():
    """Load all datasets with intelligent error handling"""
    try:
        # Core datasets - read with encoding handling
        staples = pd.read_csv('data/clean/wfp_core_staples.csv', 
                              parse_dates=['date'],
                              encoding='utf-8-sig')
        
        # Clean column names
        staples.columns = [col.strip().replace(' ', '_') for col in staples.columns]
        
        # Clean string columns to remove any problematic characters
        for col in staples.select_dtypes(include=['object']).columns:
            staples[col] = staples[col].astype(str).str.strip()
            # Remove any non-ASCII characters that might cause issues
            staples[col] = staples[col].apply(lambda x: ''.join(char for char in x if ord(char) < 128))
        
        monthly = pd.read_csv('data/clean/wfp_monthly_avg.csv', 
                              parse_dates=['date'],
                              encoding='utf-8-sig')
        
        # Try to load features
        try:
            maize_features = pd.read_csv('data/clean/maize_features.csv', 
                                         parse_dates=['date'],
                                         encoding='utf-8-sig')
        except:
            maize_features = None
        
        # Model outputs
        try:
            prophet_fc = pd.read_csv('models/trained/prophet_forecast.csv', 
                                      parse_dates=['ds'],
                                      encoding='utf-8-sig')
        except:
            prophet_fc = None
        
        try:
            model_comparison = pd.read_csv('models/trained/model_comparison.csv', 
                                           index_col=0,
                                           encoding='utf-8-sig')
        except:
            model_comparison = None
        
        return staples, monthly, maize_features, prophet_fc, model_comparison
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Please ensure all data files are in the correct location.")
        # Return empty dataframe as fallback
        empty_df = pd.DataFrame(columns=['date', 'cm_name', 'mp_price', 'adm1_name', 'mkt_name'])
        return empty_df, empty_df, None, None, None
    except Exception as e:
        st.error(f"Error loading data: {e}")
        empty_df = pd.DataFrame(columns=['date', 'cm_name', 'mp_price', 'adm1_name', 'mkt_name'])
        return empty_df, empty_df, None, None, None

# Enhanced forecasting function with multiple models
@st.cache_data
def generate_enhanced_forecast(df, commodity, periods=12):
    """Generate intelligent forecasts using ensemble of methods"""
    try:
        commodity_data = df[df['cm_name'] == commodity].copy()
        if len(commodity_data) == 0:
            return None, None
        
        # Calculate monthly average
        monthly_avg = commodity_data.groupby('date')['mp_price'].mean().reset_index()
        monthly_avg = monthly_avg.sort_values('date')
        
        if len(monthly_avg) < 6:
            return None, None
        
        prices = monthly_avg['mp_price'].values
        
        # Method 1: Exponential Smoothing
        alpha = 0.3
        exp_forecast = []
        last_value = prices[-1]
        for i in range(periods):
            next_val = alpha * last_value + (1 - alpha) * np.mean(prices[-12:]) if len(prices) >= 12 else last_value
            exp_forecast.append(next_val)
            last_value = next_val
        
        # Method 2: Linear Trend
        if len(prices) > 1:
            x = np.arange(len(prices))
            z = np.polyfit(x, prices, 1)
            trend = np.poly1d(z)
            linear_forecast = [trend(len(prices) + i) for i in range(periods)]
        else:
            linear_forecast = [prices[-1]] * periods
        
        # Method 3: Seasonal Naive
        if len(prices) >= 12:
            seasonal_pattern = prices[-12:]
            seasonal_forecast = []
            for i in range(periods):
                seasonal_forecast.append(seasonal_pattern[i % 12])
        else:
            seasonal_forecast = [prices[-1]] * periods
        
        # Ensemble forecast (weighted average)
        ensemble = (0.4 * np.array(exp_forecast) + 
                    0.3 * np.array(linear_forecast) + 
                    0.3 * np.array(seasonal_forecast))
        
        last_date = monthly_avg['date'].max()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=30),
            periods=periods,
            freq='MS'
        )
        
        # Calculate confidence intervals based on historical volatility
        volatility = np.std(prices) / np.mean(prices) if np.mean(prices) > 0 else 0.1
        confidence_width = volatility * ensemble
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'price': ensemble,
            'lower_bound': ensemble - confidence_width * 1.645,  # 90% confidence
            'upper_bound': ensemble + confidence_width * 1.645,
            'exp_smoothing': exp_forecast,
            'linear_trend': linear_forecast,
            'seasonal': seasonal_forecast
        })
        
        # Ensure no negative prices
        forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
        forecast_df['price'] = forecast_df['price'].clip(lower=0)
        
        return monthly_avg, forecast_df
    except Exception as e:
        st.error(f"Forecast error: {e}")
        return None, None

# Load data
staples, monthly, maize_features, prophet_fc, model_comparison = load_data()

# Check if data loaded successfully
if staples.empty:
    st.error("Failed to load data. Please check your data files.")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3892/3892646.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        [
            "Home Dashboard",
            "Price Explorer", 
            "Inflation Calculator",
            "Price Forecasts",
            "Market Intelligence",
            "About",
            "Developer"
        ],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick Stats with safe calculations
    st.markdown("### Data Overview")
    
    total_records = len(staples)
    commodities = staples['cm_name'].nunique() if 'cm_name' in staples.columns else 0
    markets = staples['mkt_name'].nunique() if 'mkt_name' in staples.columns else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{total_records:,}")
    with col2:
        st.metric("Commodities", commodities)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Markets", markets)
    with col2:
        st.metric("Data Period", "2006-2021")
    
    # Data quality indicator
    if 'mp_price' in staples.columns:
        completeness = (1 - staples['mp_price'].isna().sum() / len(staples)) * 100
        st.markdown("### Data Quality")
        st.progress(completeness / 100)
        st.caption(f"Completeness: {completeness:.1f}%")
    
    st.markdown("---")
    
    # Links
    st.markdown("### Connect")
    st.markdown("""
    [![GitHub](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/Kaks753/food-inflation)
    [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-blue)](https://muemastephenportfolio.netlify.app/)
    """)

# ============================================================================
# PAGE: HOME DASHBOARD - FIXED
# ============================================================================
if page == "Home Dashboard":
    st.markdown("<h1 class='main-header'>Kenya Food Price Intelligence System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3>What This Dashboard Does</h3>
    <p>An advanced analytics platform that transforms 15 years of food price data into actionable intelligence.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Smart Key Metrics
    st.markdown("## Market Pulse")
    
    # Find maize data safely
    maize_keywords = ['Maize', 'maize']
    maize_data = None
    
    for keyword in maize_keywords:
        mask = staples['cm_name'].str.contains(keyword, case=False, na=False)
        if mask.any():
            maize_data = staples[mask].copy()
            break
    
    if maize_data is None or len(maize_data) == 0:
        maize_data = staples.iloc[:100]  # fallback to first 100 rows
    
    # Calculate metrics with safety checks
    maize_avg = maize_data['mp_price'].mean() if len(maize_data) > 0 else 0
    maize_max = maize_data['mp_price'].max() if len(maize_data) > 0 else 0
    maize_min = maize_data['mp_price'].min() if len(maize_data) > 0 else 0
    
    # Get latest price
    latest_maize = maize_data.sort_values('date').iloc[-1]['mp_price'] if len(maize_data) > 0 else 0
    price_range = maize_max - maize_min
    
    # Calculate volatility safely
    volatility = (maize_data['mp_price'].std() / maize_avg * 100) if maize_avg > 0 else 0
    
    # Historical growth
    first_year_avg = maize_data[maize_data['date'].dt.year == 2006]['mp_price'].mean() if 2006 in maize_data['date'].dt.year.values else 0
    last_year_avg = maize_data[maize_data['date'].dt.year == 2021]['mp_price'].mean() if 2021 in maize_data['date'].dt.year.values else 0
    
    if first_year_avg > 0 and last_year_avg > 0:
        growth_rate = ((last_year_avg - first_year_avg) / first_year_avg) * 100
    else:
        growth_rate = 0
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Current Maize Price (2021)",
            value=f"{latest_maize:.2f} KES/kg",
            delta=f"{((latest_maize - maize_avg) / maize_avg * 100):.1f}% vs avg" if maize_avg > 0 else None,
            help="Latest recorded price compared to historical average"
        )
    
    with col2:
        st.metric(
            label="15-Year Growth",
            value=f"+{growth_rate:.1f}%" if growth_rate > 0 else f"{growth_rate:.1f}%",
            delta=f"CAGR: {(growth_rate/15):.1f}%/year" if growth_rate != 0 else None,
            help="Total price increase from 2006 to 2021"
        )
    
    with col3:
        st.metric(
            label="Price Volatility",
            value=f"{volatility:.1f}%",
            delta=f"Range: {price_range:.0f} KES",
            help="Coefficient of variation showing price stability"
        )
    
    with col4:
        st.metric(
            label="Best Buying Season",
            value="Oct-Dec",
            delta="↓ 20% cheaper",
            help="Harvest season typically offers 15-20% lower prices"
        )
    
    # Data coverage visualization
    st.markdown("## Market Coverage")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top markets
        top_markets = staples.groupby('mkt_name').size().sort_values(ascending=False).head(10)
        if len(top_markets) > 0:
            fig = px.bar(
                x=top_markets.values,
                y=top_markets.index,
                orientation='h',
                title='Top 10 Markets by Data Coverage',
                labels={'x': 'Number of Price Records', 'y': 'Market'},
                color=top_markets.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Commodities by category
        commodity_counts = staples['cm_name'].value_counts().head(8)
        if len(commodity_counts) > 0:
            fig = px.pie(
                values=commodity_counts.values,
                names=commodity_counts.index,
                title='Top 8 Tracked Commodities',
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Regional insights
    if 'adm1_name' in staples.columns:
        st.markdown("## Regional Market Dynamics")
        
        region_avg = staples.groupby('adm1_name')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
        region_avg.columns = ['Region', 'Avg Price (KES)', 'Std Dev', 'Records']
        region_avg = region_avg.sort_values('Avg Price (KES)', ascending=False)
        
        if len(region_avg) > 0:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    region_avg,
                    x='Region',
                    y='Avg Price (KES)',
                    error_y='Std Dev',
                    title='Average Food Prices by Region',
                    labels={'Avg Price (KES)': 'Average Price (KES/kg)'},
                    color='Avg Price (KES)',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### Key Insights")
                st.markdown(f"""
                - **Highest prices**: {region_avg.iloc[0]['Region']} ({region_avg.iloc[0]['Avg Price (KES)']:.1f} KES)
                - **Lowest prices**: {region_avg.iloc[-1]['Region']} ({region_avg.iloc[-1]['Avg Price (KES)']:.1f} KES)
                - **Price gap**: {(region_avg.iloc[0]['Avg Price (KES)'] - region_avg.iloc[-1]['Avg Price (KES)']):.1f} KES
                - **Most data**: {region_avg.loc[region_avg['Records'].idxmax(), 'Region']}
                """)

# ============================================================================
# PAGE: PRICE EXPLORER - FIXED
# ============================================================================
elif page == "Price Explorer":
    st.markdown("<h1 class='main-header'>Price Explorer</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <p><strong>Explore historical price trends</strong> for different commodities across regions and time periods.</p>
    </div>
    """, unsafe_allow_html=True)
    
    if 'cm_name' not in staples.columns:
        st.error("Commodity data not available")
        st.stop()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    commodity_options = sorted(staples['cm_name'].unique())
    region_options = ['All Regions'] + sorted(staples['adm1_name'].unique().tolist()) if 'adm1_name' in staples.columns else ['All Regions']
    
    with col1:
        selected_commodity = st.selectbox(
            "Select Commodity",
            commodity_options
        )
    
    with col2:
        selected_region = st.selectbox(
            "Select Region",
            region_options
        )
    
    with col3:
        year_range = st.slider(
            "Year Range",
            min_value=2006,
            max_value=2021,
            value=(2006, 2021)
        )
    
    # Filter data
    filtered_data = staples[staples['cm_name'] == selected_commodity].copy()
    filtered_data = filtered_data[
        (filtered_data['date'].dt.year >= year_range[0]) &
        (filtered_data['date'].dt.year <= year_range[1])
    ]
    
    if selected_region != 'All Regions' and 'adm1_name' in filtered_data.columns:
        filtered_data = filtered_data[filtered_data['adm1_name'] == selected_region]
    
    if len(filtered_data) == 0:
        st.warning("No data available for selected filters. Try different options.")
        st.stop()
    
    # Aggregate
    price_trend = filtered_data.groupby('date')['mp_price'].agg(['mean', 'std', 'count']).reset_index()
    price_trend.columns = ['date', 'price', 'std', 'count']
    
    # Main price chart
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=price_trend['date'],
        y=price_trend['price'],
        mode='lines',
        name='Price',
        line=dict(color='#3498db', width=3),
        fill='tonexty'
    ))
    
    # Add confidence band if std available
    if 'std' in price_trend.columns:
        fig.add_trace(go.Scatter(
            x=price_trend['date'],
            y=price_trend['price'] + price_trend['std'],
            mode='lines',
            name='Upper Bound',
            line=dict(width=0),
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=price_trend['date'],
            y=price_trend['price'] - price_trend['std'],
            mode='lines',
            name='Lower Bound',
            line=dict(width=0),
            fillcolor='rgba(52, 152, 219, 0.2)',
            fill='tonexty',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"{selected_commodity} Price Trend",
        xaxis_title='Date',
        yaxis_title='Price (KES/kg)',
        hovermode='x unified',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Price", f"{price_trend['price'].mean():.2f} KES")
    with col2:
        st.metric("Max Price", f"{price_trend['price'].max():.2f} KES")
    with col3:
        st.metric("Min Price", f"{price_trend['price'].min():.2f} KES")
    with col4:
        cv = (price_trend['price'].std() / price_trend['price'].mean() * 100) if price_trend['price'].mean() > 0 else 0
        st.metric("Volatility", f"{cv:.1f}%")

# ============================================================================
# PAGE: INFLATION CALCULATOR - FIXED AND ENHANCED
# ============================================================================
elif page == "Inflation Calculator":
    st.markdown("<h1 class='main-header'>Intelligent Inflation Calculator</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>How This Works</h4>
    <p>Calculate the <strong>real cost impact</strong> of food price inflation on your household budget. 
    This tool uses actual historical data to show how prices have changed and what it means for your purchasing power.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all commodities
    all_commodities = sorted(staples['cm_name'].unique())
    
    # User inputs
    col1, col2 = st.columns(2)
    
    with col1:
        calc_commodity = st.selectbox(
            "Select Food Item",
            all_commodities,
            help="Choose the commodity you want to analyze"
        )
        
        start_year = st.selectbox(
            "Start Year",
            range(2006, 2021),
            index=0,
            help="Beginning of comparison period"
        )
    
    with col2:
        quantity = st.number_input(
            "Monthly Quantity (kg)",
            min_value=1.0,
            max_value=1000.0,
            value=10.0,
            step=1.0,
            help="How much do you buy per month?"
        )
        
        end_year = st.selectbox(
            "End Year",
            range(2007, 2022),
            index=len(range(2007, 2022))-1,
            help="End of comparison period"
        )
    
    # Calculate button
    if st.button("Calculate Impact", type="primary", use_container_width=True):
        
        # Get data for selected commodity
        commodity_data = staples[staples['cm_name'] == calc_commodity].copy()
        
        # Filter by years
        start_data = commodity_data[commodity_data['date'].dt.year == start_year]['mp_price']
        end_data = commodity_data[commodity_data['date'].dt.year == end_year]['mp_price']
        
        # Check if we have data
        if len(start_data) == 0 or len(end_data) == 0:
            st.warning(f"No data available for {calc_commodity} in the selected years.")
            
            # Suggest available years
            available_years = sorted(commodity_data['date'].dt.year.unique())
            if len(available_years) > 0:
                st.info(f"Available years for {calc_commodity}: {', '.join(map(str, available_years))}")
        else:
            # Calculate averages
            start_avg = start_data.mean()
            end_avg = end_data.mean()
            
            # Calculations
            price_change = end_avg - start_avg
            percent_change = (price_change / start_avg) * 100 if start_avg > 0 else 0
            
            year_diff = end_year - start_year
            annual_rate = percent_change / year_diff if year_diff > 0 else 0
            
            start_cost = start_avg * quantity
            end_cost = end_avg * quantity
            monthly_increase = end_cost - start_cost
            annual_increase = monthly_increase * 12
            
            # Results
            st.markdown("---")
            st.markdown("## Impact Analysis Results")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    f"Price in {start_year}",
                    f"{start_avg:.2f} KES/kg"
                )
            
            with col2:
                delta_text = f"+{percent_change:.1f}%" if percent_change > 0 else f"{percent_change:.1f}%"
                st.metric(
                    f"Price in {end_year}",
                    f"{end_avg:.2f} KES/kg",
                    delta=delta_text
                )
            
            with col3:
                st.metric(
                    "Monthly Impact",
                    f"{monthly_increase:+.2f} KES"
                )
            
            with col4:
                st.metric(
                    "Annual Impact",
                    f"{annual_increase:+.2f} KES",
                    delta=f"{annual_rate:.1f}% per year" if annual_rate != 0 else None
                )
            
            # Visualization
            st.markdown("### Cost Evolution Over Time")
            
            # Get yearly data
            yearly_data = commodity_data.groupby(commodity_data['date'].dt.year)['mp_price'].mean().reset_index()
            yearly_data.columns = ['year', 'price']
            yearly_data = yearly_data[(yearly_data['year'] >= start_year) & (yearly_data['year'] <= end_year)]
            
            if len(yearly_data) > 0:
                yearly_data['monthly_cost'] = yearly_data['price'] * quantity
                
                fig = make_subplots(
                    rows=2, cols=1,
                    subplot_titles=('Price per Kg', 'Your Monthly Cost'),
                    vertical_spacing=0.15
                )
                
                fig.add_trace(
                    go.Scatter(
                        x=yearly_data['year'],
                        y=yearly_data['price'],
                        mode='lines+markers',
                        name='Price/kg',
                        line=dict(color='#e74c3c', width=3),
                        marker=dict(size=10)
                    ),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(
                        x=yearly_data['year'],
                        y=yearly_data['monthly_cost'],
                        name='Monthly Cost',
                        marker_color='#3498db'
                    ),
                    row=2, col=1
                )
                
                fig.update_xaxes(title_text="Year", row=2, col=1)
                fig.update_yaxes(title_text="KES/kg", row=1, col=1)
                fig.update_yaxes(title_text="KES", row=2, col=1)
                fig.update_layout(height=600, showlegend=False)
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Smart insights based on impact level
                if percent_change > 50:
                    st.error(f"⚠️ **High Impact**: Prices increased by {percent_change:.1f}% over {year_diff} years. Consider bulk buying during harvest season (Oct-Dec) or finding alternative sources.")
                elif percent_change > 20:
                    st.warning(f"📊 **Moderate Impact**: Prices increased by {percent_change:.1f}% over {year_diff} years. Budget planning recommended.")
                elif percent_change > 0:
                    st.success(f"✅ **Low Impact**: Prices increased by {percent_change:.1f}% over {year_diff} years. Relatively stable.")
                else:
                    st.success(f"✅ **Prices Decreased**: Prices dropped by {abs(percent_change):.1f}% over {year_diff} years. Good time to buy!")
                
                # Additional insights
                st.info(f"""
                **Summary:**
                - Your monthly cost went from {start_cost:,.0f} KES to {end_cost:,.0f} KES
                - That's an increase of {monthly_increase:,.0f} KES per month
                - Average annual inflation rate: {annual_rate:.1f}% per year
                - Over {year_diff} years, you've spent {annual_increase * year_diff:,.0f} KES more
                """)

# ============================================================================
# PAGE: PRICE FORECASTS - ULTRA SMART VERSION
# ============================================================================
elif page == "Price Forecasts":
    st.markdown("<h1 class='main-header'>Interactive Price Forecasting</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h4>🎯 Smart Predictive Intelligence</h4>
    <p>Our system analyzes <strong>historical patterns, seasonality, and trends</strong> to generate intelligent forecasts. 
    We balance statistical reliability with user needs - forecasts beyond 3 years become progressively uncertain, 
    and we'll warn you when we're "guessing" too much.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all commodities
    all_commodities = sorted(staples['cm_name'].unique())
    
    # Main forecast controls
    col1, col2 = st.columns([1, 1])
    
    with col1:
        forecast_commodity = st.selectbox(
            "🌾 Select Commodity",
            all_commodities,
            help="Choose which food item to predict",
            key="forecast_commodity_main"
        )
        
        # Smart forecast type selection
        forecast_type = st.radio(
            "📅 Forecast Horizon",
            ["Short-term (1-12 months)", "Medium-term (1-3 years)", "Long-term (3-5 years)", "Custom Date"],
            help="Longer forecasts have higher uncertainty",
            key="forecast_type_main"
        )
    
    with col2:
        # Show data quality for selected commodity
        commodity_data = staples[staples['cm_name'] == forecast_commodity]
        data_years = commodity_data['date'].dt.year.nunique()
        data_months = len(commodity_data)
        last_data_date = commodity_data['date'].max()
        
        st.markdown("### 📊 Data Quality")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Years of Data", data_years)
            st.metric("Total Records", data_months)
        with col_b:
            st.metric("Last Update", last_data_date.strftime('%b %Y'))
            completeness = (data_months / (data_years * 12)) * 100
            st.metric("Completeness", f"{completeness:.0f}%")
        
        # Data quality indicator
        if data_months < 24:
            st.warning("⚠️ Limited historical data - forecasts may be less reliable")
        elif data_months < 48:
            st.info("📊 Moderate historical data - forecasts have good reliability")
        else:
            st.success("✅ Excellent historical data - forecasts are statistically sound")
    
    st.markdown("---")
    
    # Determine forecast periods based on selection
    last_date = staples['date'].max()
    current_year = last_date.year
    current_month = last_date.month
    
    if forecast_type == "Short-term (1-12 months)":
        forecast_periods = st.slider(
            "Select number of months to forecast",
            min_value=3,
            max_value=12,
            value=6,
            step=1,
            help="Short-term forecasts are most reliable",
            key="short_term_slider"
        )
        target_date = last_date + pd.DateOffset(months=forecast_periods)
        reliability = "High"
        
    elif forecast_type == "Medium-term (1-3 years)":
        years = st.slider(
            "Select number of years to forecast",
            min_value=1,
            max_value=3,
            value=2,
            step=1,
            help="Medium-term forecasts show trends but have moderate uncertainty",
            key="medium_term_slider"
        )
        forecast_periods = years * 12
        target_date = last_date + pd.DateOffset(years=years)
        reliability = "Moderate"
        
        st.info(f"📊 Medium-term forecast: {years} year(s) - Uncertainty increases with time")
        
    elif forecast_type == "Long-term (3-5 years)":
        years = st.slider(
            "Select number of years to forecast",
            min_value=3,
            max_value=5,
            value=3,
            step=1,
            help="Long-term forecasts show general trends but have high uncertainty",
            key="long_term_slider"
        )
        forecast_periods = years * 12
        target_date = last_date + pd.DateOffset(years=years)
        reliability = "Low"
        
        st.warning(f"⚠️ Long-term forecast: {years} year(s) - High uncertainty, showing general trend only")
        
    else:  # Custom Date
        st.markdown("### Select Target Date")
        col1, col2 = st.columns(2)
        
        with col1:
            target_year = st.selectbox(
                "Year",
                options=range(current_year, 2029),  # Up to 2028 as requested
                index=0,
                key="target_year"
            )
        
        with col2:
            target_month = st.selectbox(
                "Month",
                options=range(1, 13),
                format_func=lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1],
                index=current_month-1 if current_month <= 12 else 0,
                key="target_month"
            )
        
        target_date = datetime(target_year, target_month, 1)
        
        # Calculate months difference
        months_diff = (target_date.year - last_date.year) * 12 + (target_date.month - last_date.month)
        
        # Smart validation
        if months_diff < 1:
            st.error("❌ Target date must be in the future")
            st.stop()
        elif months_diff > 60:  # 5 years max
            st.error("❌ Cannot forecast beyond 5 years (statistically unreliable)")
            st.stop()
        elif months_diff > 36:
            st.warning(f"⚠️ {months_diff} months is a long-term forecast. Showing general trends only.")
            reliability = "Low"
        elif months_diff > 12:
            st.info(f"📊 {months_diff} months is a medium-term forecast. Moderate reliability.")
            reliability = "Moderate"
        else:
            reliability = "High"
        
        forecast_periods = months_diff
    
    # Advanced Settings in an expander - FIXED to work properly
    with st.expander("⚙️ Advanced Forecast Settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence_level = st.select_slider(
                "Confidence Level",
                options=[80, 85, 90, 95, 99],
                value=90,
                help="Higher confidence = wider prediction intervals",
                key="confidence_advanced"
            )
        
        with col2:
            show_seasonality = st.checkbox(
                "Show Seasonal Patterns", 
                value=True,
                help="Display monthly price variations",
                key="seasonality_advanced"
            )
        
        with col3:
            show_components = st.checkbox(
                "Show Forecast Components",
                value=False,
                help="Break down forecast into trend and seasonality",
                key="components_advanced"
            )
        
        # Model selection
        model_type = st.radio(
            "Forecast Model",
            ["Ensemble (Recommended)", "Conservative (Wider intervals)", "Aggressive (Narrower intervals)"],
            horizontal=True,
            help="Ensemble combines multiple models for best results",
            key="model_advanced"
        )
    
    # Generate forecast button
    if st.button("🚀 Generate Smart Forecast", type="primary", use_container_width=True, key="generate_btn"):
        
        # Check data sufficiency
        if len(commodity_data) < 12:
            st.error(f"❌ Insufficient data for {forecast_commodity}. Need at least 12 months of data.")
            st.stop()
        
        with st.spinner("🤖 Analyzing patterns and generating intelligent forecast..."):
            
            # Generate forecast with appropriate periods
            historical, forecast = generate_enhanced_forecast(staples, forecast_commodity, forecast_periods)
            
            if historical is None or forecast is None:
                st.error(f"❌ Cannot generate forecast for {forecast_commodity}")
                st.stop()
            
            # Adjust model parameters based on selection
            conf_multipliers = {80: 1.28, 85: 1.44, 90: 1.645, 95: 1.96, 99: 2.576}
            base_multiplier = conf_multipliers[confidence_level]
            
            # Adjust multiplier based on model type
            if model_type == "Conservative (Wider intervals)":
                multiplier = base_multiplier * 1.2
            elif model_type == "Aggressive (Narrower intervals)":
                multiplier = base_multiplier * 0.8
            else:
                multiplier = base_multiplier
            
            # Calculate confidence intervals
            std_dev = historical['mp_price'].std()
            forecast['lower'] = forecast['price'] - (std_dev * multiplier)
            forecast['upper'] = forecast['price'] + (std_dev * multiplier)
            forecast['lower'] = forecast['lower'].clip(lower=0)
            
            # Success message with reliability indicator
            if reliability == "High":
                st.success(f"✅ **High Reliability Forecast** until {target_date.strftime('%B %Y')}")
            elif reliability == "Moderate":
                st.info(f"📊 **Moderate Reliability Forecast** until {target_date.strftime('%B %Y')}")
            else:
                st.warning(f"⚠️ **Low Reliability Forecast** until {target_date.strftime('%B %Y')} - Showing trend only")
            
            # MAIN CHART
            fig = go.Figure()
            
            # Historical data
            fig.add_trace(go.Scatter(
                x=historical['date'],
                y=historical['mp_price'],
                mode='lines',
                name='Historical Prices',
                line=dict(color='#2c3e50', width=2)
            ))
            
            # Forecast
            fig.add_trace(go.Scatter(
                x=forecast['date'],
                y=forecast['price'],
                mode='lines',
                name='Forecast',
                line=dict(color='#e74c3c', width=3, dash='dash')
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['date'].tolist() + forecast['date'].tolist()[::-1],
                y=forecast['upper'].tolist() + forecast['lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(231, 76, 60, 0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence',
                hoverinfo='skip'
            ))
            
            # Forecast start line
            last_hist_date = historical['date'].max()
            fig.add_shape(
                type="line",
                x0=last_hist_date,
                y0=0,
                x1=last_hist_date,
                y1=1,
                yref="paper",
                line=dict(color="gray", width=2, dash="dash")
            )
            
            fig.add_annotation(
                x=last_hist_date,
                y=1,
                yref="paper",
                text="Forecast Start",
                showarrow=True,
                arrowhead=2,
                ax=40,
                ay=-30
            )
            
            # Update layout
            fig.update_layout(
                title=f"{forecast_commodity} - Price Forecast to {target_date.strftime('%B %Y')}",
                xaxis_title="Date",
                yaxis_title="Price (KES/kg)",
                hovermode='x unified',
                height=500,
                legend=dict(orientation='h', yanchor='bottom', y=1.02, x=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # FORECAST STATISTICS
            st.markdown("### 📊 Forecast Statistics")
            
            last_price = historical['mp_price'].iloc[-1]
            avg_forecast = forecast['price'].mean()
            min_forecast = forecast['lower'].min()
            max_forecast = forecast['upper'].max()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Current Price",
                    f"{last_price:.2f} KES",
                    help="Latest recorded price"
                )
            
            with col2:
                pct_change = ((avg_forecast - last_price) / last_price * 100) if last_price > 0 else 0
                st.metric(
                    "Average Forecast",
                    f"{avg_forecast:.2f} KES",
                    delta=f"{pct_change:+.1f}%",
                    help="Average predicted price over forecast period"
                )
            
            with col3:
                st.metric(
                    "Expected Range",
                    f"{min_forecast:.0f} - {max_forecast:.0f} KES",
                    help=f"{confidence_level}% confidence interval"
                )
            
            with col4:
                first_f = forecast['price'].iloc[0]
                last_f = forecast['price'].iloc[-1]
                trend_pct = ((last_f - first_f) / first_f) * 100 if first_f > 0 else 0
                st.metric(
                    "Overall Trend",
                    f"{trend_pct:+.1f}%",
                    delta=f"Over {forecast_periods} months",
                    help="Price change direction over forecast period"
                )
            
            # SEASONAL PATTERN ANALYSIS - FIXED with better bar chart
            if show_seasonality and len(historical) >= 24:
                st.markdown("### 📅 Seasonal Pattern Analysis")
                
                # Calculate seasonal indices
                historical_copy = historical.copy()
                historical_copy['month'] = historical_copy['date'].dt.month
                monthly_avg = historical_copy.groupby('month')['mp_price'].mean()
                overall_avg = historical_copy['mp_price'].mean()
                seasonal_indices = (monthly_avg / overall_avg - 1) * 100 if overall_avg > 0 else pd.Series([0]*12)
                
                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                
                fig_seasonal = go.Figure()
                
                # Prepare data
                valid_months = seasonal_indices.index.tolist()
                valid_names = [month_names[i-1] for i in valid_months]
                valid_values = seasonal_indices.values
                
                # Create colors based on positive/negative
                colors = ['#e74c3c' if x > 0 else '#27ae60' for x in valid_values]
                
                fig_seasonal.add_trace(go.Bar(
                    x=valid_names,
                    y=valid_values,
                    marker_color=colors,
                    text=[f"{x:+.1f}%" for x in valid_values],
                    textposition='outside',
                    textfont=dict(size=12, color='black')
                ))
                
                # Add zero line
                fig_seasonal.add_hline(y=0, line_dash="dash", line_color="gray", line_width=1)
                
                # Adjust layout to show text properly
                fig_seasonal.update_layout(
                    title="Monthly Price Variations vs Annual Average",
                    xaxis_title="Month",
                    yaxis_title="Variation from Average (%)",
                    height=450,
                    yaxis=dict(
                        range=[min(valid_values) * 1.2, max(valid_values) * 1.2],  # Add padding
                        zeroline=True,
                        zerolinecolor='gray',
                        zerolinewidth=1
                    ),
                    margin=dict(t=50, b=50, l=50, r=50)
                )
                
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
                # Best and worst months with savings calculation
                if len(valid_values) > 0:
                    best_idx = valid_values.argmin()
                    worst_idx = valid_values.argmax()
                    
                    # Calculate actual price difference
                    best_price = monthly_avg.iloc[best_idx]
                    worst_price = monthly_avg.iloc[worst_idx]
                    savings_kes = worst_price - best_price
                    savings_pct = (savings_kes / worst_price) * 100
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"""
                        ✅ **Best Time to Buy: {valid_names[best_idx]}**  
                        - Price: {best_price:.1f} KES/kg  
                        - {valid_values[best_idx]:+.1f}% below annual average  
                        - Save up to {savings_kes:.1f} KES/kg vs peak prices
                        """)
                    
                    with col2:
                        st.warning(f"""
                        ⚠️ **Highest Prices: {valid_names[worst_idx]}**  
                        - Price: {worst_price:.1f} KES/kg  
                        - {valid_values[worst_idx]:+.1f}% above annual average  
                        - {savings_pct:.0f}% more expensive than best month
                        """)
            
            # FORECAST COMPONENTS
            if show_components:
                st.markdown("### 🔍 Forecast Components Breakdown")
                
                # Decompose forecast
                x = np.arange(len(forecast))
                z = np.polyfit(x, forecast['price'], 1)
                trend_line = np.poly1d(z)
                trend_component = trend_line(x)
                seasonal_component = forecast['price'] - trend_component
                
                fig_components = make_subplots(
                    rows=3, cols=1,
                    subplot_titles=('Combined Forecast', 'Trend Component', 'Seasonal Component'),
                    vertical_spacing=0.1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=forecast['price'], 
                              mode='lines', name='Forecast', line=dict(color='#e74c3c')),
                    row=1, col=1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=trend_component, 
                              mode='lines', name='Trend', line=dict(color='#27ae60')),
                    row=2, col=1
                )
                
                fig_components.add_trace(
                    go.Scatter(x=forecast['date'], y=seasonal_component, 
                              mode='lines', name='Seasonal', line=dict(color='#3498db')),
                    row=3, col=1
                )
                
                fig_components.update_layout(height=600, showlegend=False)
                fig_components.update_xaxes(title_text="Date", row=3, col=1)
                
                st.plotly_chart(fig_components, use_container_width=True)
                
                # Explain components
                st.info("""
                **Understanding the components:**
                - **Trend**: The long-term direction (up/down/stable)
                - **Seasonal**: Regular monthly patterns (harvest cycles, holidays)
                - **Combined**: The final forecast = Trend + Seasonal
                """)
            
            # SMART RECOMMENDATIONS
            st.markdown("## 💡 Smart Recommendations")
            
            # Volatility calculation
            volatility = (historical['mp_price'].std() / historical['mp_price'].mean()) * 100 if historical['mp_price'].mean() > 0 else 0
            
            # Market condition assessment
            if trend_pct > 15:
                market = "strongly rising"
                action = "Consider bulk purchasing now"
                timing = "Next 1-2 months"
            elif trend_pct > 5:
                market = "moderately rising"
                action = "Gradual purchasing recommended"
                timing = "Spread purchases over next 3 months"
            elif trend_pct < -15:
                market = "strongly falling"
                action = "Delay major purchases"
                timing = f"Best buying window: {forecast_periods//2} months from now"
            elif trend_pct < -5:
                market = "moderately falling"
                action = "Partial purchases now, more later"
                timing = "Consider staged buying"
            else:
                market = "stable"
                action = "Normal purchasing patterns"
                timing = "Monitor for seasonal opportunities"
            
            # Create recommendation boxes
            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"""
                **📈 Market Outlook**  
                - **Trend**: {market} ({trend_pct:+.1f}% over {forecast_periods} months)  
                - **Volatility**: {volatility:.1f}% ({'High' if volatility>30 else 'Moderate' if volatility>15 else 'Low'})  
                - **Reliability**: {reliability} confidence  
                """)
            
            with col2:
                if reliability == "High":
                    st.success(f"""
                    **✅ Recommended Strategy**  
                    - **Action**: {action}  
                    - **Timing**: {timing}  
                    - **Confidence**: High reliability forecast
                    """)
                elif reliability == "Moderate":
                    st.warning(f"""
                    **📊 Recommended Strategy**  
                    - **Action**: {action} (with caution)  
                    - **Timing**: Monitor market closely  
                    - **Confidence**: Moderate reliability
                    """)
                else:
                    st.error(f"""
                    **⚠️ Long-term Guidance**  
                    - **Action**: Focus on seasonal patterns  
                    - **Timing**: Use for general planning only  
                    - **Confidence**: Low reliability - review regularly
                    """)
            
            # Volatility alert
            if volatility > 30:
                st.error(f"⚠️ **HIGH VOLATILITY ALERT**: Price swings of ±{volatility:.0f}% are common. Consider hedging strategies.")
            elif volatility > 15:
                st.warning(f"📊 **MODERATE VOLATILITY**: Normal market fluctuations of ±{volatility:.0f}% expected.")
            else:
                st.success(f"✅ **LOW VOLATILITY**: Stable market with only ±{volatility:.0f}% variation.")
            
            # DOWNLOAD FORECAST DATA - FIXED with proper format
            st.markdown("### 📥 Download Forecast Data")
            
            # Create download dataframe with clean format
            download_df = pd.DataFrame({
                'Year': forecast['date'].dt.year,
                'Month': forecast['date'].dt.strftime('%b'),
                'Forecast_Price_KES': forecast['price'].round(2),
                'Lower_Bound_KES': forecast['lower'].round(2),
                'Upper_Bound_KES': forecast['upper'].round(2),
                'Confidence_Level': f'{confidence_level}%'
            })
            
            # Add seasonal info if available
            if show_seasonality and len(historical) >= 24:
                # Add month numbers for joining
                forecast_months = forecast['date'].dt.month
                seasonal_values = [seasonal_indices[m] if m in seasonal_indices.index else 0 for m in forecast_months]
                download_df['Seasonal_Variation_%'] = [f"{x:+.1f}%" for x in seasonal_values]
            
            st.dataframe(download_df, use_container_width=True)
            
            # Download button
            csv = download_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Full Forecast (CSV)",
                data=csv,
                file_name=f"{forecast_commodity.replace(' ', '_')}_forecast_to_{target_date.strftime('%Y_%m')}.csv",
                mime="text/csv",
                use_container_width=True,
                key="download_forecast_btn"
            )
            
            # Additional insights for long-term forecasts
            if reliability == "Low":
                st.markdown("---")
                st.warning("""
                **⚠️ Important Note on Long-term Forecasts**
                
                Forecasts beyond 3 years show general trends only. Many factors can affect prices:
                - Weather patterns and climate change
                - Economic policies and inflation
                - Global market conditions
                - Technology and farming practices
                
                **Use this forecast for strategic planning, not precise budgeting.**
                """)

# ============================================================================
# PAGE: ABOUT - FIXED
# ============================================================================
elif page == "About":
    st.markdown("<h1 class='main-header'>About This Intelligence System</h1>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
    <h3>What This App Does</h3>
    <p>The Kenya Food Price Intelligence System is an advanced analytics platform that transforms raw market data 
    into actionable insights for policy makers, farmers, traders, and consumers.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Key Features
        
        **1. Price Dashboard**
        - Track prices across 45 markets
        - Compare 34 different commodities
        - Filter by region and time period
        
        **2. Inflation Calculator**
        - Calculate personal cost impact
        - Compare prices across years
        - See monthly/annual budget effects
        
        **3. Price Forecasting**
        - Predict up to 5 years ahead
        - Customizable forecast horizons
        - Confidence intervals
        """)
    
    with col2:
        st.markdown("""
        ### Data Specifications
        
        **Dataset Coverage**
        - **Time Period**: 2006-2021 (15 years)
        - **Total Records**: 8,884 observations
        - **Commodities**: 34 food items
        - **Markets**: 45 locations
        - **Regions**: 6 administrative areas
        
        **Data Source**
        - World Food Programme (WFP)
        - Monthly price records
        - 95%+ completeness
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### How to Use This App
    
    **For Policy Makers**
    - Use Market Intelligence to identify regional disparities
    - Monitor Seasonal Patterns for intervention timing
    
    **For Farmers**
    - Check Seasonal Patterns to plan harvest timing
    - Use Regional Comparison to find best markets
    
    **For Consumers**
    - Use Inflation Calculator to understand cost impacts
    - Monitor Forecasts to plan purchases
    
    **For Traders**
    - Use Regional Comparison for arbitrage opportunities
    - Monitor Price Trends for inventory decisions
    """)

# ============================================================================
# PAGE: DEVELOPER - FIXED
# ============================================================================
elif page == "Developer":
    st.markdown("<h1 class='main-header'>Developer Information</h1>", unsafe_allow_html=True)
    
    # Profile section
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=200)
    
    with col2:
        st.markdown("""
        # Stephen Muema
        ## Data Scientist & Machine Learning Engineer
        
        Transforming complex datasets into actionable insights using Machine Learning, 
        Python, SQL, and advanced statistical modeling.
        
        [![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Kaks753)
        [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-green)](https://muemastephenportfolio.netlify.app/)
        """)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Education
        
        **Data Science Program**
        Moringa School | 2025-2026
        
        **Bachelor of Mathematics**
        Maasai Mara University | 2019-2023
        """)
    
    with col2:
        st.markdown("""
        ### Core Skills
        
        - Python, Machine Learning
        - Data Analysis & Visualization
        - Statistical Modeling
        - SQL & Databases
        """)
    
    st.markdown("---")
    
    st.markdown("""
    ### Technologies Used
    
    **Languages & Libraries**
    - Python, Pandas, NumPy
    - Plotly, Streamlit
    - Scikit-learn
    
    **Tools & Concepts**
    - Time Series Analysis
    - Statistical Forecasting
    - Interactive Dashboards
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem 0;'>
    <p><strong>Kenya Food Price Intelligence System</strong> | Developed by Stephen Muema | 2024</p>
    <p>Data Source: World Food Programme (WFP) | Period: 2006-2021</p>
    <p>
        <a href='https://github.com/Kaks753/food-inflation' target='_blank'>GitHub</a> | 
        <a href='https://muemastephenportfolio.netlify.app/' target='_blank'>Portfolio</a>
    </p>
</div>
""", unsafe_allow_html=True)