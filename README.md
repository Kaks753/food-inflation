# 📊 Kenya Food Price Inflation Tracker

> A comprehensive data science project tracking food price inflation in Kenya using multi-source data integration, time series analysis, and interactive visualizations.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Latest-green.svg)](https://pandas.pydata.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🎯 Project Overview

The **Kenya Food Price Inflation Tracker** is a data-driven project that analyzes and visualizes food price trends across Kenya. This project combines:

- **Real-world data** from WFP, FAO, and World Bank
- **Time series forecasting** using ARIMA, SARIMA, and Prophet
- **Geospatial analysis** of price variations across 45+ markets
- **Interactive dashboards** for policy insights
- **Machine learning** for price prediction

### Why This Project Matters

Food inflation affects everyone. This project provides:
- **Policy makers**: Data-driven insights for intervention
- **Businesses**: Price trend forecasting for planning
- **Consumers**: Transparency on food cost trends
- **Researchers**: Open-source analysis framework

---

## 📁 Project Structure

```
webapp/
├── data/
│   ├── raw/                          # Original downloaded datasets
│   │   ├── wfp_food_prices_kenya_full.csv      # 8,884 records (2006-2021)
│   │   ├── wfp_food_prices_global.csv          # Global context data
│   │   ├── kenya_food_prices_sample.csv        # Sample synthetic data
│   │   └── *.dta                               # Housing data (legacy)
│   ├── processed/                    # Cleaned and transformed data
│   │   └── *.csv                               # Converted datasets
│   └── external/                     # Additional data sources
│
├── notebooks/                        # Jupyter notebooks for analysis
│   ├── 01_Data_Understanding.ipynb             # Initial setup
│   └── 02_Data_Loading_and_Initial_Exploration.ipynb  # Main exploration
│
├── src/                             # Source code modules
│   ├── data_processing/             # Data cleaning & ETL
│   │   └── download_food_data.py              # Data acquisition script
│   ├── models/                      # ML models & forecasting
│   ├── utils/                       # Helper functions
│   └── visualization/               # Plotting utilities
│
├── models/                          # Trained models
│   ├── trained/                     # Saved model files
│   └── configs/                     # Model configurations
│
├── deployment/                      # Deployment files
│   ├── streamlit/                   # Streamlit dashboard
│   └── scripts/                     # Deployment scripts
│
├── reports/                         # Generated reports
│   ├── figures/                     # Saved visualizations
│   └── analysis/                    # Analysis reports
│
├── tests/                           # Unit tests
└── docs/                            # Documentation
```

---

## 📊 Datasets

### 1. **WFP Global Food Prices Database** 🌍
- **Source**: [Humanitarian Data Exchange](https://data.humdata.org/dataset/global-wfp-food-prices)
- **Coverage**: Kenya 2006-2021 (8,884 records)
- **Commodities**: 34 food items
- **Markets**: 45 locations across 6 regions
- **Format**: CSV
- **License**: CC-BY 4.0

**Key Commodities**:
- Maize (white) - Retail
- Beans (dry) - Wholesale/Retail
- Rice
- Vegetable Oil
- Milk (pasteurized)
- Sorghum, Potatoes, and more

**Major Markets**:
- Nairobi (1,748 records)
- Eldoret (1,011 records)
- Kisumu (974 records)
- Mombasa (632 records)
- Nakuru (535 records)

### 2. **FAO Food Price Index** 📈
- **Source**: FAO Food Price Monitoring and Analysis (FPMA)
- **Purpose**: Global context for Kenya trends
- **URL**: https://fpma.fao.org/

### 3. **Sample Generated Data** 🧪
- **Purpose**: Testing and development
- **Coverage**: 2020-2024 with realistic trends
- **Features**: Inflation, seasonality, regional variation

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
pip
virtualenv (recommended)
```

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd webapp
```

2. **Create virtual environment**
```bash
python -m venv myenv
source myenv/bin/activate  # On Windows: myenv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn
pip install scikit-learn statsmodels prophet
pip install jupyter notebook
pip install streamlit plotly folium
```

4. **Verify data**
```bash
ls data/raw/
# Should show: wfp_food_prices_kenya_full.csv and other files
```

### Quick Start

1. **Explore the data**
```bash
jupyter notebook notebooks/02_Data_Loading_and_Initial_Exploration.ipynb
```

2. **Run data download script** (if needed)
```bash
python src/data_processing/download_food_data.py
```

---

## 📖 Usage

### Jupyter Notebooks

#### Notebook 1: Data Understanding
```bash
jupyter notebook notebooks/01_Data_Understanding.ipynb
```
- Library imports
- Initial setup

#### Notebook 2: Data Loading & Exploration
```bash
jupyter notebook notebooks/02_Data_Loading_and_Initial_Exploration.ipynb
```
- Load WFP Kenya food prices
- Commodity analysis
- Market distribution
- Price trends
- Inflation calculations

### Python Scripts

#### Download Additional Data
```python
python src/data_processing/download_food_data.py
```

---

## 🔍 Analysis Overview

### 1. **Descriptive Statistics**
- Price distributions by commodity
- Temporal coverage analysis
- Geographic market analysis

### 2. **Time Series Analysis**
- Trend decomposition (trend, seasonality, residuals)
- Year-over-year inflation rates
- Price volatility analysis

### 3. **Forecasting Models** (Coming Soon)
- ARIMA/SARIMA for short-term forecasts
- Prophet for seasonal patterns
- Machine learning models for multi-feature predictions

### 4. **Geospatial Analysis** (Coming Soon)
- Heat maps of price variations
- Urban vs. rural price differences
- Supply chain insights

### 5. **Dashboard** (Coming Soon)
- Interactive Streamlit application
- Real-time price tracking
- Forecast visualizations

---

## 📈 Key Findings (Preliminary)

From WFP Kenya dataset (2006-2021):

1. **Most Tracked Commodities**:
   - Maize (white): 1,052 retail records
   - Beans (dry): 733 wholesale records
   - Sorghum: 708 wholesale records

2. **Price Trends**:
   - Maize prices show cyclical patterns with seasonal spikes
   - Overall upward trend indicating inflation
   - Significant volatility around 2011 and 2017

3. **Regional Insights**:
   - Nairobi has most comprehensive data coverage
   - Coastal markets (Mombasa) show different patterns
   - Rural markets have less frequent data

---

## 🛠️ Technologies Used

### Data Processing
- **Pandas**: Data manipulation
- **NumPy**: Numerical computations

### Visualization
- **Matplotlib**: Static plots
- **Seaborn**: Statistical visualizations
- **Plotly**: Interactive charts
- **Folium**: Geospatial mapping

### Time Series & ML
- **Statsmodels**: ARIMA/SARIMA
- **Prophet**: Facebook's forecasting tool
- **Scikit-learn**: ML algorithms

### Deployment
- **Streamlit**: Web dashboard
- **Docker**: Containerization (planned)

---

## 📚 Data Sources & Links

| Source | Description | URL |
|--------|-------------|-----|
| WFP Food Prices | Global food price database | https://data.humdata.org/dataset/global-wfp-food-prices |
| FAO FPMA Tool | Food Price Monitoring | https://fpma.fao.org/ |
| World Bank Data | Economic indicators | https://microdata.worldbank.org/ |
| KNBS | Kenya National Bureau of Statistics | https://www.knbs.or.ke/ |

---

## 🗺️ Roadmap

### Phase 1: MVP (Current) ✅
- [x] Project structure setup
- [x] Data acquisition (WFP Kenya)
- [x] Initial exploratory analysis
- [x] Basic visualizations

### Phase 2: Analysis & Modeling (In Progress)
- [ ] Data cleaning and preprocessing
- [ ] Feature engineering
- [ ] Time series forecasting models
- [ ] Model validation and comparison

### Phase 3: Advanced Analytics
- [ ] Geospatial analysis with maps
- [ ] Correlation with external factors (fuel, exchange rates)
- [ ] Multi-commodity inflation index

### Phase 4: Deployment
- [ ] Streamlit dashboard
- [ ] Automated data updates
- [ ] Cloud deployment (Render/Streamlit Cloud)
- [ ] API endpoints for data access

---

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Your Name**
- GitHub: [@yourusername](https://github.com/yourusername)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/yourprofile)
- Portfolio: [yourportfolio.com](https://yourportfolio.com)

---

## 🙏 Acknowledgments

- **World Food Programme (WFP)** for comprehensive food price data
- **FAO** for global food price monitoring tools
- **Humanitarian Data Exchange** for data accessibility
- **Kenya National Bureau of Statistics** for official statistics

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please reach out:
- Email: your.email@example.com
- Twitter: [@yourhandle](https://twitter.com/yourhandle)

---

**⭐ If you find this project useful, please give it a star!**

---

*Last Updated: March 2026*
