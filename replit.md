# Overview

This is an advanced Elliott Wave analysis application built with Streamlit for professional trading analysis. The system provides comprehensive technical analysis focusing on Elliott Wave patterns, Fibonacci retracements, and market sentiment analysis for financial instruments including Gold (XAU/USD), NASDAQ-100 (NDX100), and German DAX (GER40). The application combines multiple data sources, real-time price fetching, news sentiment analysis, and sophisticated pattern recognition to generate trading signals and market insights.

# User Preferences

Preferred communication style: Simple, everyday language.

# System Architecture

## Frontend Architecture
- **Streamlit Framework**: Web-based dashboard with wide layout configuration and bilingual support (English/Arabic)
- **Interactive Components**: Sidebar controls for asset selection, timeframe selection, and trading parameters
- **Visualization**: Plotly integration for advanced charting and technical analysis visualization
- **Multi-page Structure**: Main app with advanced analysis features and simplified versions for different use cases

## Backend Architecture
- **Modular Design**: Component-based architecture with specialized analyzers for different aspects of market analysis
- **Data Processing Pipeline**: Sequential analysis flow from data fetching → Elliott Wave detection → Fibonacci calculations → signal generation
- **Caching Strategy**: Streamlit resource caching for component initialization to improve performance
- **Error Handling**: Graceful fallbacks for data source failures and API limitations

## Core Analysis Components
- **Elliott Wave Analyzer**: Pattern recognition for impulse and corrective waves with complex pattern support (diagonal triangles, double zigzags, expanded flats)
- **Fibonacci Calculator**: Multi-level retracement and extension calculations with advanced ratios and time-based analysis
- **Trading Signal Generator**: Weighted signal system combining Elliott Wave, Fibonacci, trend, and momentum indicators
- **Price Channel Analyzer**: Trend channel, regression channel, and Elliott Wave specific channel analysis
- **News Analyzer**: Market sentiment analysis with keyword-based asset correlation and impact assessment

## Data Management
- **Multi-Source Data Provider**: Hierarchical data fetching from Alpha Vantage, Yahoo Finance, Finnhub, and TwelveData APIs
- **Real-time Price Fetching**: Dedicated module for current market prices with multiple fallback sources
- **Data Validation**: Price data validation and realistic range checking
- **Historical Data**: OHLC data retrieval with configurable timeframes and periods

## Analysis Engine
- **Pattern Recognition**: Advanced Elliott Wave pattern detection including complex corrections and diagonal patterns
- **Technical Indicators**: Fibonacci retracements, extensions, and time-based ratios
- **Risk Assessment**: Multi-level risk analysis with conservative, moderate, and aggressive strategies
- **Confidence Scoring**: Probabilistic analysis with confidence intervals for pattern recognition and signal generation

# External Dependencies

## Financial Data APIs
- **Alpha Vantage API**: Primary data source for real-time quotes and historical data
- **Yahoo Finance API**: Secondary data source for current market prices and chart data
- **Finnhub API**: Alternative financial data provider for market data
- **TwelveData API**: Additional data source for comprehensive market coverage

## Python Libraries
- **Streamlit**: Web application framework for interactive dashboards
- **Plotly**: Advanced charting and visualization library
- **Pandas/NumPy**: Data manipulation and numerical analysis
- **Requests**: HTTP client for API communication
- **Trafilatura**: Web scraping for news content extraction

## News and Sentiment Sources
- **Financial News Websites**: Investing.com, MarketWatch, Yahoo Finance, CNBC, Bloomberg
- **Web Scraping**: Real-time news extraction and sentiment analysis
- **Keyword-based Analysis**: Asset-specific news correlation and market impact assessment

## Configuration Management
- **Environment Variables**: API keys and configuration through environment variables and Streamlit secrets
- **Fallback Systems**: Multiple data source redundancy for high availability
- **Symbol Mapping**: Cross-platform symbol translation for different data providers