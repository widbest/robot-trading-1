import streamlit as st
import plotly.graph_objects as go
import plotly.subplots as sp
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from elliott_wave_analyzer import ElliottWaveAnalyzer
from fibonacci_calculator import FibonacciCalculator
from data_provider import DataProvider
from trading_signals import TradingSignalGenerator
from price_channels import PriceChannelAnalyzer
from news_analyzer import NewsAnalyzer

# Page configuration
st.set_page_config(
    page_title="Elliott Wave Analysis - Professional Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize components
@st.cache_resource
def init_components():
    data_provider = DataProvider()
    elliott_analyzer = ElliottWaveAnalyzer()
    fibonacci_calc = FibonacciCalculator()
    signal_generator = TradingSignalGenerator()
    channel_analyzer = PriceChannelAnalyzer()
    news_analyzer = NewsAnalyzer()
    return data_provider, elliott_analyzer, fibonacci_calc, signal_generator, channel_analyzer, news_analyzer

def main():
    st.title("🌊 Elliott Wave Analysis Dashboard")
    st.markdown("### Professional Elliott Wave Analysis with Fibonacci Retracements")
    
    # Initialize components
    data_provider, elliott_analyzer, fibonacci_calc, signal_generator, channel_analyzer, news_analyzer = init_components()
    
    # Sidebar controls
    with st.sidebar:
        st.header("📊 Trading Parameters")
        
        # Asset selection
        selected_asset = st.selectbox(
            "Select Asset:",
            ["XAU/USD", "NDX100", "GER40"],
            index=0
        )
        
        # Timeframe selection
        timeframe = st.selectbox(
            "Timeframe:",
            ["5min", "1H", "4H", "Daily"],
            index=1
        )
        
        # Analysis parameters
        st.subheader("Elliott Wave Settings")
        wave_sensitivity = st.slider("Wave Detection Sensitivity", 0.5, 2.0, 1.0, 0.1)
        fibonacci_levels = st.multiselect(
            "Fibonacci Levels:",
            [23.6, 38.2, 50.0, 61.8, 78.6, 100.0],
            default=[23.6, 38.2, 50.0, 61.8, 78.6]
        )
        
        # Display current market prices
        st.subheader("📊 Current Market Prices")
        try:
            current_prices = data_provider._fetch_current_market_prices()
            for instrument, price in current_prices.items():
                if instrument == selected_asset:
                    st.metric(f"{instrument}", f"${price:.2f}", delta="Live Price")
                else:
                    st.caption(f"{instrument}: ${price:.2f}")
        except:
            pass
        
        st.divider()
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto Refresh (30s)", value=False)
        
        if st.button("🔄 Manual Refresh"):
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([3, 1])
    
    # Initialize variables
    price_data = None
    wave_analysis = None
    fibonacci_levels_data = None
    channel_analysis = None
    news_analysis = None
    
    with col1:
        # Price chart with Elliott Wave analysis
        st.subheader(f"📈 {selected_asset} - {timeframe} Elliott Wave Analysis")
        
        try:
            # Get price data
            with st.spinner("Loading market data..."):
                price_data = data_provider.get_price_data(selected_asset, timeframe)
            
            if price_data is not None and not price_data.empty:
                # Perform Elliott Wave analysis
                wave_analysis = elliott_analyzer.analyze_waves(
                    price_data, 
                    sensitivity=wave_sensitivity,
                    asset_type=selected_asset
                )
                
                # Calculate Fibonacci levels
                fibonacci_levels_data = fibonacci_calc.calculate_levels(
                    price_data, 
                    wave_analysis,
                    levels=fibonacci_levels
                )
                
                # Perform price channel analysis
                channel_analysis = channel_analyzer.analyze_channels(price_data, wave_analysis)
                
                # Perform news analysis
                news_analysis = news_analyzer.analyze_market_news(selected_asset, hours_back=24)
                
                # Create the main chart
                fig = create_elliott_wave_chart(
                    price_data, 
                    wave_analysis, 
                    fibonacci_levels_data,
                    selected_asset,
                    channel_analysis
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display wave analysis results
                display_wave_analysis(wave_analysis)
                
            else:
                st.error("Unable to load market data. Please check your internet connection.")
                st.info("Authentic market data is required for accurate Elliott Wave analysis.")
                
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    with col2:
        # Trading signals and wave information
        st.subheader("🎯 Trading Signals")
        
        if price_data is not None and wave_analysis and fibonacci_levels_data:
            # Generate enhanced trading signals with channel and news analysis
            signals = signal_generator.generate_signals(
                price_data, 
                wave_analysis, 
                fibonacci_levels_data,
                channel_analysis,
                news_analysis
            )
            
            display_trading_signals(signals, selected_asset)
            display_current_wave_info(wave_analysis)
            display_fibonacci_levels(fibonacci_levels_data)
            display_channel_analysis(channel_analysis)
            display_news_analysis(news_analysis, selected_asset)
        else:
            st.info("Loading analysis...")
    
    # Auto-refresh mechanism
    if auto_refresh:
        time.sleep(30)
        st.rerun()

def create_elliott_wave_chart(price_data, wave_analysis, fibonacci_levels, asset_name, channel_analysis=None):
    """Create the main Elliott Wave chart with Fibonacci levels and price channels"""
    
    fig = go.Figure()
    
    # Add candlestick chart
    fig.add_trace(go.Candlestick(
        x=price_data.index,
        open=price_data['open'],
        high=price_data['high'],
        low=price_data['low'],
        close=price_data['close'],
        name=asset_name,
        increasing_line_color='#00FF88',
        decreasing_line_color='#FF4444'
    ))
    
    # Add Elliott Wave lines and labels
    if wave_analysis and 'wave_points' in wave_analysis:
        wave_points = wave_analysis['wave_points']
        
        # Draw wave lines
        for i in range(len(wave_points) - 1):
            fig.add_trace(go.Scatter(
                x=[wave_points[i]['timestamp'], wave_points[i+1]['timestamp']],
                y=[wave_points[i]['price'], wave_points[i+1]['price']],
                mode='lines+markers',
                line=dict(color='#FFD700', width=2),
                marker=dict(size=8, color='#FFD700'),
                name=f"Wave {wave_points[i]['label']} to {wave_points[i+1]['label']}",
                showlegend=False
            ))
        
        # Add wave labels
        for point in wave_points:
            fig.add_annotation(
                x=point['timestamp'],
                y=point['price'],
                text=f"<b>{point['label']}</b>",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor='#FFD700',
                bgcolor='rgba(255, 215, 0, 0.8)',
                bordercolor='#FFD700',
                borderwidth=1,
                font=dict(color='black', size=12, family='Arial Black')
            )
    
    # Add Fibonacci retracement levels
    if fibonacci_levels and 'levels' in fibonacci_levels:
        for level in fibonacci_levels['levels']:
            fig.add_hline(
                y=level['price'],
                line_dash="dash",
                line_color=level['color'],
                annotation_text=f"Fib {level['percentage']:.1f}% - {level['price']:.2f}",
                annotation_position="right"
            )
    
    # Add price channels
    if channel_analysis:
        trend_channel = channel_analysis.get('trend_channel', {})
        if trend_channel:
            # Add trend channel lines
            x_values = price_data.index
            
            # Upper trend line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=[trend_channel.get('upper_line', 0)] * len(x_values),
                mode='lines',
                line=dict(color='cyan', width=2, dash='dot'),
                name='Channel Resistance',
                showlegend=True
            ))
            
            # Lower trend line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=[trend_channel.get('lower_line', 0)] * len(x_values),
                mode='lines',
                line=dict(color='orange', width=2, dash='dot'),
                name='Channel Support',
                showlegend=True
            ))
            
            # Middle line
            fig.add_trace(go.Scatter(
                x=x_values,
                y=[trend_channel.get('middle_line', 0)] * len(x_values),
                mode='lines',
                line=dict(color='gray', width=1, dash='dash'),
                name='Channel Middle',
                showlegend=True,
                opacity=0.7
            ))
    
    # Update layout
    fig.update_layout(
        title=f"{asset_name} Elliott Wave Analysis",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark",
        height=600,
        showlegend=True,
        xaxis_rangeslider_visible=False
    )
    
    return fig

def display_wave_analysis(wave_analysis):
    """Display Elliott Wave analysis results"""
    
    if not wave_analysis:
        return
    
    st.subheader("🌊 Wave Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Current Wave",
            wave_analysis.get('current_wave', 'Unknown'),
            delta=wave_analysis.get('wave_confidence', 0)
        )
    
    with col2:
        st.metric(
            "Next Expected Wave",
            wave_analysis.get('next_wave', 'Unknown'),
            delta=wave_analysis.get('trend_strength', 0)
        )
    
    with col3:
        st.metric(
            "Pattern Completion",
            f"{wave_analysis.get('completion_percentage', 0):.1f}%",
            delta=wave_analysis.get('pattern_reliability', 0)
        )
    
    # Wave details
    if 'wave_details' in wave_analysis:
        with st.expander("📊 Detailed Wave Information"):
            for wave, details in wave_analysis['wave_details'].items():
                st.write(f"**{wave}**: {details}")

def display_trading_signals(signals, asset_name):
    """Display trading recommendations"""
    
    if not signals:
        return
    
    # Current signal
    current_signal = signals.get('current_signal', {})
    signal_type = current_signal.get('type', 'HOLD')
    signal_strength = current_signal.get('strength', 0)
    
    # Signal display with color coding
    if signal_type == 'BUY':
        st.success(f"🟢 **BUY SIGNAL**")
        st.write(f"**Strength**: {signal_strength:.1f}/10")
    elif signal_type == 'SELL':
        st.error(f"🔴 **SELL SIGNAL**")
        st.write(f"**Strength**: {signal_strength:.1f}/10")
    else:
        st.info(f"🟡 **HOLD**")
        st.write(f"**Strength**: {signal_strength:.1f}/10")
    
    # Entry and exit levels
    if 'entry_level' in current_signal and current_signal['entry_level'] is not None:
        st.write(f"**Entry Level**: {current_signal['entry_level']:.2f}")
    if 'stop_loss' in current_signal and current_signal['stop_loss'] is not None:
        st.write(f"**Stop Loss**: {current_signal['stop_loss']:.2f}")
    if 'take_profit' in current_signal and current_signal['take_profit'] is not None:
        st.write(f"**Take Profit**: {current_signal['take_profit']:.2f}")
    
    # Signal reasoning
    if 'reasoning' in current_signal:
        with st.expander("📝 Signal Analysis"):
            st.write(current_signal['reasoning'])

def display_current_wave_info(wave_analysis):
    """Display current wave information with enhanced precision"""
    
    st.subheader("📍 Current Wave Analysis")
    
    if wave_analysis and 'current_wave_info' in wave_analysis:
        info = wave_analysis['current_wave_info']
        
        # Current wave details
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Current Wave", info.get('label', 'Unknown'), 
                     delta=f"{info.get('confidence', 0):.0%} confidence")
            st.write(f"**Type**: {info.get('type', 'Unknown')}")
            st.write(f"**Status**: {info.get('wave_status', 'Unknown')}")
        
        with col2:
            st.metric("Wave Progress", f"{info.get('progress', 0):.1f}%")
            st.write(f"**Direction**: {info.get('direction', 'Unknown')}")
            st.write(f"**Target Price**: ${info.get('price_level', 0):.2f}")
        
        # Next wave prediction
        if 'next_wave_info' in wave_analysis:
            next_info = wave_analysis['next_wave_info']
            st.subheader("🎯 Next Wave Prediction")
            
            col3, col4 = st.columns(2)
            with col3:
                st.metric("Next Wave", next_info.get('label', 'Unknown'),
                         delta=f"{next_info.get('probability', 0):.0%} probability")
            with col4:
                st.write(f"**Expected Direction**: {next_info.get('predicted_direction', 'Unknown')}")
                if 'target_levels' in next_info:
                    targets = next_info['target_levels']
                    if 'conservative' in targets:
                        st.write(f"**Target**: ${targets['conservative']:.2f}")
        
        # Wave characteristics
        if 'characteristics' in info:
            with st.expander("🔍 Elliott Wave Theory Analysis"):
                for char in info['characteristics']:
                    st.write(f"• {char}")
                
                # Add theoretical expectations
                st.write("\n**Elliott Wave Rules Applied:**")
                current_label = info.get('label', '')
                if current_label in ['1', '3', '5']:
                    st.write("• This is an impulse wave - expect strong trending movement")
                elif current_label in ['2', '4']:
                    st.write("• This is a corrective wave - expect counter-trend movement")
                elif current_label in ['A', 'B', 'C']:
                    st.write("• This is part of a corrective sequence")
    else:
        st.info("Wave analysis in progress...")

def display_fibonacci_levels(fibonacci_levels):
    """Display Fibonacci retracement levels"""
    
    st.subheader("📊 Fibonacci Levels")
    
    if fibonacci_levels and 'levels' in fibonacci_levels:
        for level in fibonacci_levels['levels']:
            percentage = level['percentage']
            price = level['price']
            distance = level.get('distance_from_current', 0)
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{percentage:.1f}%**")
            with col2:
                st.write(f"{price:.2f}")
            
            # Distance indicator
            if distance > 0:
                st.caption(f"↑ {distance:.2f} points above")
            elif distance < 0:
                st.caption(f"↓ {abs(distance):.2f} points below")
            else:
                st.caption("← Current level")

def display_channel_analysis(channel_analysis):
    """Display price channel analysis"""
    
    st.subheader("📊 القنوات السعرية")
    
    if not channel_analysis:
        st.info("جاري حساب تحليل القنوات...")
        return
    
    # Channel position
    channel_position = channel_analysis.get('channel_position', {})
    if channel_position:
        col1, col2 = st.columns(2)
        
        with col1:
            position = channel_position.get('position', 'Unknown')
            percentage = channel_position.get('percentage', 0)
            st.metric("موقع السعر في القناة", position, 
                     delta=f"{percentage:.1f}%")
        
        with col2:
            zone = channel_position.get('zone', 'Neutral')
            if 'Overbought' in zone:
                st.error("منطقة تشبع شرائي")
            elif 'Oversold' in zone:
                st.success("منطقة تشبع بيعي")
            else:
                st.info("منطقة محايدة")
    
    # Channel strength
    channel_strength = channel_analysis.get('channel_strength', {})
    if channel_strength:
        reliability = channel_strength.get('reliability', 'Unknown')
        strength_score = channel_strength.get('strength_score', 0)
        
        st.write(f"**قوة القناة**: {reliability}")
        st.progress(strength_score)
        
        total_touches = channel_strength.get('total_touches', 0)
        st.caption(f"تم اختبار القناة {total_touches} مرة")
    
    # Breakout signals
    breakout_signals = channel_analysis.get('breakout_signals', {})
    if breakout_signals and breakout_signals.get('active_signals'):
        st.subheader("⚡ إشارات الاختراق")
        
        for signal in breakout_signals['active_signals']:
            signal_type = signal.get('type', '')
            description = signal.get('description', '')
            strength = signal.get('strength', 'Medium')
            
            if 'Bullish' in signal_type:
                st.success(f"**اختراق صاعد**: {description} ({strength})")
            elif 'Bearish' in signal_type:
                st.error(f"**اختراق هابط**: {description} ({strength})")
            else:
                st.info(f"**{signal_type}**: {description} ({strength})")
    
    # Support and resistance levels from channels
    support_resistance = channel_analysis.get('support_resistance', {})
    if support_resistance:
        with st.expander("📋 مستويات الدعم والمقاومة"):
            
            resistance_levels = support_resistance.get('resistance_levels', [])
            if resistance_levels:
                st.write("**مستويات المقاومة:**")
                for level in resistance_levels[:3]:
                    st.write(f"• ${level['price']:.2f} ({level['type']}) - {level['strength']}")
            
            support_levels = support_resistance.get('support_levels', [])
            if support_levels:
                st.write("**مستويات الدعم:**")
                for level in support_levels[:3]:
                    st.write(f"• ${level['price']:.2f} ({level['type']}) - {level['strength']}")
    
    # Trend channel info
    trend_channel = channel_analysis.get('trend_channel', {})
    if trend_channel:
        trend_direction = trend_channel.get('trend_direction', 'Unknown')
        channel_width = trend_channel.get('channel_width', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("اتجاه القناة", "صاعد" if trend_direction == "Upward" else "هابط" if trend_direction == "Downward" else "غير محدد")
        with col2:
            st.metric("عرض القناة", f"${channel_width:.2f}")

def display_news_analysis(news_analysis, asset_name):
    """Display news sentiment analysis"""
    
    st.subheader("📰 تحليل الأخبار والمشاعر")
    
    if not news_analysis or news_analysis.get('news_count', 0) == 0:
        st.info("لا توجد أخبار حديثة للتحليل")
        return
    
    # News summary
    news_count = news_analysis.get('news_count', 0)
    sentiment_analysis = news_analysis.get('sentiment_analysis', {})
    market_impact = news_analysis.get('market_impact', {})
    news_signals = news_analysis.get('news_signals', {})
    
    # Overall sentiment metrics
    col1, col2 = st.columns(2)
    
    with col1:
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 0)
        trend = sentiment_analysis.get('trend', 'Neutral')
        
        if trend == 'Bullish':
            st.success(f"**المشاعر العامة**: إيجابية")
            st.metric("درجة المشاعر", f"{overall_sentiment:.2f}", delta="إيجابي")
        elif trend == 'Bearish':
            st.error(f"**المشاعر العامة**: سلبية")
            st.metric("درجة المشاعر", f"{overall_sentiment:.2f}", delta="سلبي")
        else:
            st.info(f"**المشاعر العامة**: محايدة")
            st.metric("درجة المشاعر", f"{overall_sentiment:.2f}", delta="محايد")
    
    with col2:
        confidence = sentiment_analysis.get('confidence', 0)
        impact_level = market_impact.get('overall_impact', 'Low')
        
        st.metric("مستوى الثقة", f"{confidence:.0%}")
        
        impact_arabic = {"High": "عالي", "Medium": "متوسط", "Low": "منخفض"}
        if impact_level == 'High':
            st.error(f"**التأثير المتوقع**: {impact_arabic[impact_level]}")
        elif impact_level == 'Medium':
            st.warning(f"**التأثير المتوقع**: {impact_arabic[impact_level]}")
        else:
            st.info(f"**التأثير المتوقع**: {impact_arabic[impact_level]}")
    
    # News breakdown
    positive_news = sentiment_analysis.get('positive_news', 0)
    negative_news = sentiment_analysis.get('negative_news', 0)
    neutral_news = news_count - positive_news - negative_news
    
    st.write("**توزيع الأخبار:**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("أخبار إيجابية", positive_news, delta=f"{positive_news/news_count:.0%}" if news_count > 0 else "0%")
    with col2:
        st.metric("أخبار سلبية", negative_news, delta=f"{negative_news/news_count:.0%}" if news_count > 0 else "0%")
    with col3:
        st.metric("أخبار محايدة", neutral_news, delta=f"{neutral_news/news_count:.0%}" if news_count > 0 else "0%")
    
    # Trading signal from news
    if news_signals:
        signal_type = news_signals.get('signal', 'HOLD')
        signal_strength = news_signals.get('strength', 0)
        signal_confidence = news_signals.get('confidence', 0)
        time_horizon = news_signals.get('time_horizon', 'Unknown')
        
        st.subheader("📊 إشارة التداول من الأخبار")
        
        col1, col2 = st.columns(2)
        with col1:
            if signal_type == 'BUY':
                st.success(f"**الإشارة**: شراء")
            elif signal_type == 'SELL':
                st.error(f"**الإشارة**: بيع")
            else:
                st.info(f"**الإشارة**: انتظار")
            
            st.metric("قوة الإشارة", f"{signal_strength}/10")
        
        with col2:
            st.metric("مستوى الثقة", f"{signal_confidence:.0%}")
            st.write(f"**الإطار الزمني**: {time_horizon}")
        
        # Signal reasoning
        reasoning = news_signals.get('reasoning', '')
        if reasoning:
            with st.expander("📝 تفسير الإشارة"):
                st.write(reasoning)
    
    # Recent headlines
    recent_headlines = news_analysis.get('recent_headlines', [])
    if recent_headlines:
        with st.expander("📋 العناوين الحديثة"):
            for i, headline in enumerate(recent_headlines[:5], 1):
                st.write(f"{i}. {headline}")
    
    # Key themes
    key_themes = news_analysis.get('key_themes', [])
    if key_themes:
        with st.expander("🏷️ المواضيع الرئيسية"):
            for theme in key_themes:
                st.write(f"• {theme}")
    
    # Risk factors
    risk_factors = news_analysis.get('risk_factors', [])
    if risk_factors:
        with st.expander("⚠️ عوامل المخاطر"):
            for risk in risk_factors:
                st.warning(f"• {risk}")
    
    # Market volatility expectation
    volatility = market_impact.get('volatility_expectation', 'Low')
    volatility_arabic = {"High": "عالية", "Medium": "متوسطة", "Low": "منخفضة"}
    
    st.caption(f"📈 التذبذب المتوقع: {volatility_arabic.get(volatility, 'غير معروف')}")

if __name__ == "__main__":
    main()
