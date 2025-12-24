#!/usr/bin/env python3
"""
Price Channel Analysis for Elliott Wave Trading
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class PriceChannelAnalyzer:
    """
    Price channel analysis to complement Elliott Wave patterns
    """
    
    def __init__(self):
        self.channels = {}
    
    def analyze_channels(self, price_data: pd.DataFrame, wave_analysis: Dict) -> Dict:
        """
        Analyze price channels for Elliott Wave patterns
        
        Args:
            price_data: OHLC price data
            wave_analysis: Elliott Wave analysis results
            
        Returns:
            Dictionary containing channel analysis
        """
        
        current_price = price_data['close'].iloc[-1]
        
        # Calculate different types of channels
        trend_channel = self._calculate_trend_channel(price_data)
        regression_channel = self._calculate_regression_channel(price_data)
        elliott_channel = self._calculate_elliott_wave_channel(price_data, wave_analysis)
        
        # Combine channel analysis
        channel_analysis = {
            'trend_channel': trend_channel,
            'regression_channel': regression_channel,
            'elliott_channel': elliott_channel,
            'current_price': current_price,
            'channel_position': self._determine_channel_position(current_price, trend_channel),
            'support_resistance': self._identify_support_resistance(price_data, trend_channel),
            'breakout_signals': self._detect_breakout_signals(price_data, trend_channel, regression_channel),
            'channel_strength': self._calculate_channel_strength(price_data, trend_channel)
        }
        
        return channel_analysis
    
    def _calculate_trend_channel(self, price_data: pd.DataFrame, period: int = 50) -> Dict:
        """Calculate trend channel using recent highs and lows"""
        
        recent_data = price_data.tail(period)
        highs = recent_data['high']
        lows = recent_data['low']
        times = np.arange(len(recent_data))
        
        # Calculate trend lines for highs and lows
        high_slope, high_intercept = np.polyfit(times, highs, 1)
        low_slope, low_intercept = np.polyfit(times, lows, 1)
        
        # Calculate current channel levels
        current_time = len(recent_data) - 1
        upper_line = high_slope * current_time + high_intercept
        lower_line = low_slope * current_time + low_intercept
        
        # Calculate channel width and middle line
        channel_width = upper_line - lower_line
        middle_line = (upper_line + lower_line) / 2
        
        return {
            'upper_line': upper_line,
            'lower_line': lower_line,
            'middle_line': middle_line,
            'channel_width': channel_width,
            'upper_slope': high_slope,
            'lower_slope': low_slope,
            'trend_direction': 'Upward' if (high_slope + low_slope) / 2 > 0 else 'Downward'
        }
    
    def _calculate_regression_channel(self, price_data: pd.DataFrame, period: int = 30) -> Dict:
        """Calculate linear regression channel"""
        
        recent_data = price_data.tail(period)
        prices = recent_data['close']
        times = np.arange(len(recent_data))
        
        # Linear regression
        prices_values = prices.values
        slope, intercept = np.polyfit(times, prices_values, 1)
        regression_line = slope * times + intercept
        
        # Calculate standard deviation for channel width  
        residuals = prices_values - regression_line
        std_dev = np.std(residuals)
        
        current_time = len(recent_data) - 1
        current_regression = slope * current_time + intercept
        
        return {
            'center_line': current_regression,
            'upper_band': current_regression + (2 * std_dev),
            'lower_band': current_regression - (2 * std_dev),
            'slope': slope,
            'r_squared': self._calculate_r_squared(prices.values, regression_line),
            'trend_strength': abs(slope) / std_dev if std_dev > 0 else 0
        }
    
    def _calculate_elliott_wave_channel(self, price_data: pd.DataFrame, wave_analysis: Dict) -> Dict:
        """Calculate Elliott Wave specific channels"""
        
        if not wave_analysis or 'wave_points' not in wave_analysis:
            return {'available': False}
        
        wave_points = wave_analysis['wave_points']
        
        if len(wave_points) < 3:
            return {'available': False}
        
        # Get the last three wave points
        recent_waves = wave_points[-3:]
        wave_prices = [point['price'] for point in recent_waves]
        wave_times = [point.get('index', i) for i, point in enumerate(recent_waves)]
        
        # Calculate Elliott Wave channel
        if len(wave_prices) >= 2:
            slope = (wave_prices[-1] - wave_prices[0]) / (wave_times[-1] - wave_times[0])
            
            # Parallel channel lines
            upper_channel = max(wave_prices) + (slope * 5)  # Project 5 periods ahead
            lower_channel = min(wave_prices) + (slope * 5)
            
            return {
                'available': True,
                'upper_channel': upper_channel,
                'lower_channel': lower_channel,
                'wave_trend': 'Bullish' if slope > 0 else 'Bearish',
                'channel_slope': slope,
                'wave_count': len(wave_points)
            }
        
        return {'available': False}
    
    def _determine_channel_position(self, current_price: float, trend_channel: Dict) -> Dict:
        """Determine where current price is within the channel"""
        
        upper = trend_channel['upper_line']
        lower = trend_channel['lower_line']
        middle = trend_channel['middle_line']
        
        if current_price > upper:
            position = "Above channel"
            percentage = 100 + ((current_price - upper) / (upper - lower)) * 100
        elif current_price < lower:
            position = "Below channel"
            percentage = ((current_price - lower) / (upper - lower)) * 100
        else:
            position = "Within channel"
            percentage = ((current_price - lower) / (upper - lower)) * 100
        
        # Determine zone
        if percentage > 80:
            zone = "Overbought zone"
        elif percentage < 20:
            zone = "Oversold zone"
        else:
            zone = "Neutral zone"
        
        return {
            'position': position,
            'percentage': round(percentage, 1),
            'zone': zone,
            'distance_to_upper': upper - current_price,
            'distance_to_lower': current_price - lower
        }
    
    def _identify_support_resistance(self, price_data: pd.DataFrame, trend_channel: Dict) -> Dict:
        """Identify key support and resistance levels"""
        
        recent_data = price_data.tail(100)
        
        # Calculate pivot points
        highs = recent_data['high']
        lows = recent_data['low']
        
        # Find significant levels
        resistance_levels = []
        support_levels = []
        
        # Channel-based levels
        resistance_levels.append({
            'price': trend_channel['upper_line'],
            'type': 'Channel Resistance',
            'strength': 'High'
        })
        
        support_levels.append({
            'price': trend_channel['lower_line'],
            'type': 'Channel Support',
            'strength': 'High'
        })
        
        # Historical pivot levels
        for i in range(5, len(recent_data) - 5):
            # Check for resistance (local high)
            if highs.iloc[i] == highs.iloc[i-5:i+6].max():
                resistance_levels.append({
                    'price': highs.iloc[i],
                    'type': 'Historical Resistance',
                    'strength': 'Medium'
                })
            
            # Check for support (local low)
            if lows.iloc[i] == lows.iloc[i-5:i+6].min():
                support_levels.append({
                    'price': lows.iloc[i],
                    'type': 'Historical Support',
                    'strength': 'Medium'
                })
        
        # Sort and limit to most relevant levels
        resistance_levels.sort(key=lambda x: x['price'], reverse=True)
        support_levels.sort(key=lambda x: x['price'], reverse=True)
        
        return {
            'resistance_levels': resistance_levels[:5],
            'support_levels': support_levels[:5],
            'key_resistance': resistance_levels[0]['price'] if resistance_levels else None,
            'key_support': support_levels[-1]['price'] if support_levels else None
        }
    
    def _detect_breakout_signals(self, price_data: pd.DataFrame, trend_channel: Dict, regression_channel: Dict) -> Dict:
        """Detect potential breakout signals"""
        
        current_price = price_data['close'].iloc[-1]
        
        # Handle volume data safely
        if 'volume' in price_data.columns:
            recent_volume = price_data['volume'].tail(10).mean()
        else:
            recent_volume = 1
        
        signals = []
        
        # Channel breakout detection
        if current_price > trend_channel['upper_line']:
            signals.append({
                'type': 'Bullish Breakout',
                'description': 'Price broke above trend channel',
                'strength': 'High' if recent_volume > 1 else 'Medium'
            })
        elif current_price < trend_channel['lower_line']:
            signals.append({
                'type': 'Bearish Breakout',
                'description': 'Price broke below trend channel',
                'strength': 'High' if recent_volume > 1 else 'Medium'
            })
        
        # Regression channel signals
        if current_price > regression_channel['upper_band']:
            signals.append({
                'type': 'Overbought Signal',
                'description': 'Price above 2-sigma regression band',
                'strength': 'Medium'
            })
        elif current_price < regression_channel['lower_band']:
            signals.append({
                'type': 'Oversold Signal',
                'description': 'Price below 2-sigma regression band',
                'strength': 'Medium'
            })
        
        return {
            'active_signals': signals,
            'breakout_probability': len(signals) * 0.3,
            'trend_continuation': trend_channel['trend_direction']
        }
    
    def _calculate_channel_strength(self, price_data: pd.DataFrame, trend_channel: Dict) -> Dict:
        """Calculate the strength and reliability of the channel"""
        
        recent_data = price_data.tail(50)
        
        # Count touches to channel lines
        upper_touches = 0
        lower_touches = 0
        tolerance = trend_channel['channel_width'] * 0.05  # 5% tolerance
        
        for _, row in recent_data.iterrows():
            if abs(row['high'] - trend_channel['upper_line']) < tolerance:
                upper_touches += 1
            if abs(row['low'] - trend_channel['lower_line']) < tolerance:
                lower_touches += 1
        
        # Calculate strength score
        total_touches = upper_touches + lower_touches
        strength_score = min(total_touches / 10, 1.0)  # Normalize to 0-1
        
        # Determine reliability
        if strength_score > 0.7:
            reliability = "Very Strong"
        elif strength_score > 0.5:
            reliability = "Strong"
        elif strength_score > 0.3:
            reliability = "Moderate"
        else:
            reliability = "Weak"
        
        return {
            'strength_score': strength_score,
            'reliability': reliability,
            'upper_touches': upper_touches,
            'lower_touches': lower_touches,
            'total_touches': total_touches
        }
    
    def _calculate_r_squared(self, actual, predicted: np.ndarray) -> float:
        """Calculate R-squared for regression analysis"""
        
        actual_values = np.array(actual)
        ss_res = np.sum((actual_values - predicted) ** 2)
        ss_tot = np.sum((actual_values - np.mean(actual_values)) ** 2)
        
        if ss_tot == 0:
            return 0
        
        return 1 - (ss_res / ss_tot)
    
    def get_channel_trading_levels(self, channel_analysis: Dict, current_price: float) -> Dict:
        """Get specific trading levels based on channel analysis"""
        
        trend_channel = channel_analysis['trend_channel']
        
        # Calculate trading levels
        entry_long = trend_channel['lower_line'] + (trend_channel['channel_width'] * 0.1)
        entry_short = trend_channel['upper_line'] - (trend_channel['channel_width'] * 0.1)
        
        stop_loss_long = trend_channel['lower_line'] - (trend_channel['channel_width'] * 0.1)
        stop_loss_short = trend_channel['upper_line'] + (trend_channel['channel_width'] * 0.1)
        
        target_long = trend_channel['upper_line']
        target_short = trend_channel['lower_line']
        
        return {
            'long_entry': entry_long,
            'short_entry': entry_short,
            'long_stop': stop_loss_long,
            'short_stop': stop_loss_short,
            'long_target': target_long,
            'short_target': target_short,
            'risk_reward_long': (target_long - entry_long) / (entry_long - stop_loss_long) if entry_long > stop_loss_long else 0,
            'risk_reward_short': (entry_short - target_short) / (stop_loss_short - entry_short) if stop_loss_short > entry_short else 0
        }