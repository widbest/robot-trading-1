import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

class TradingSignalGenerator:
    """
    Generate trading signals based on Elliott Wave analysis and Fibonacci levels
    """
    
    def __init__(self):
        self.signal_weights = {
            'elliott_wave': 0.4,
            'fibonacci': 0.3,
            'trend': 0.2,
            'momentum': 0.1
        }
        
        self.risk_levels = {
            'conservative': 0.5,
            'moderate': 1.0,
            'aggressive': 2.0
        }
    
    def generate_signals(self, price_data: pd.DataFrame, wave_analysis: Dict, 
                        fibonacci_levels: Dict, channel_analysis: Dict = None, news_analysis: Dict = None, risk_level: str = 'moderate') -> Dict:
        """
        Generate comprehensive trading signals
        
        Args:
            price_data: OHLC price data
            wave_analysis: Elliott Wave analysis results
            fibonacci_levels: Fibonacci level calculations
            risk_level: Risk tolerance level
            
        Returns:
            Dictionary containing trading signals and recommendations
        """
        
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Analyze each signal component
            wave_signal = self._analyze_elliott_wave_signal(wave_analysis, current_price)
            fibonacci_signal = self._analyze_fibonacci_signal(fibonacci_levels, current_price)
            trend_signal = self._analyze_trend_signal(price_data)
            momentum_signal = self._analyze_momentum_signal(price_data)
            
            # Add channel analysis if available
            channel_signal = {}
            if channel_analysis:
                channel_signal = self._analyze_channel_signal(channel_analysis, current_price)
            
            # Add news analysis if available
            news_signal = {}
            if news_analysis:
                news_signal = self._analyze_news_signal(news_analysis, current_price)
            
            # Combine signals with weights
            combined_signal = self._combine_signals(
                wave_signal, fibonacci_signal, trend_signal, momentum_signal, channel_signal, news_signal
            )
            
            # Generate entry and exit levels
            entry_exit_levels = self._calculate_entry_exit_levels(
                current_price, combined_signal, fibonacci_levels, wave_analysis
            )
            
            # Calculate position sizing and risk management
            risk_management = self._calculate_risk_management(
                current_price, entry_exit_levels, risk_level
            )
            
            # Generate signal reasoning
            signal_reasoning = self._generate_signal_reasoning(
                wave_signal, fibonacci_signal, trend_signal, momentum_signal
            )
            
            return {
                'current_signal': {
                    'type': combined_signal['type'],
                    'strength': combined_signal['strength'],
                    'confidence': combined_signal['confidence'],
                    'entry_level': entry_exit_levels.get('entry'),
                    'stop_loss': entry_exit_levels.get('stop_loss'),
                    'take_profit': entry_exit_levels.get('take_profit'),
                    'reasoning': signal_reasoning
                },
                'component_signals': {
                    'elliott_wave': wave_signal,
                    'fibonacci': fibonacci_signal,
                    'trend': trend_signal,
                    'momentum': momentum_signal
                },
                'risk_management': risk_management,
                'signal_history': self._get_recent_signals(price_data, wave_analysis),
                'market_context': self._analyze_market_context(price_data, wave_analysis)
            }
            
        except Exception as e:
            print(f"Error generating trading signals: {e}")
            return self._generate_default_signal()
    
    def _analyze_elliott_wave_signal(self, wave_analysis: Dict, current_price: float) -> Dict:
        """Analyze Elliott Wave patterns for trading signals"""
        
        if not wave_analysis or not wave_analysis.get('wave_points'):
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'No wave data'}
        
        current_wave = wave_analysis.get('current_wave', '1')
        next_wave = wave_analysis.get('next_wave', '2')
        wave_confidence = wave_analysis.get('wave_confidence', 0)
        
        signal_type = 'HOLD'
        signal_strength = 0
        details = []
        
        # Elliott Wave trading rules
        if current_wave == '2' and next_wave == '3':
            # End of wave 2 correction, expecting strong wave 3
            signal_type = 'BUY'
            signal_strength = 8
            details.append("Wave 2 correction completing, strong Wave 3 expected")
            
        elif current_wave == '4' and next_wave == '5':
            # End of wave 4 correction, expecting final wave 5
            signal_type = 'BUY'
            signal_strength = 6
            details.append("Wave 4 correction ending, final Wave 5 anticipated")
            
        elif current_wave == '5' and next_wave == 'A':
            # End of 5-wave impulse, expecting correction
            signal_type = 'SELL'
            signal_strength = 7
            details.append("Wave 5 completing, corrective sequence beginning")
            
        elif current_wave == 'B' and next_wave == 'C':
            # End of corrective wave B, expecting final wave C down
            signal_type = 'SELL'
            signal_strength = 6
            details.append("Wave B rally ending, Wave C decline expected")
            
        elif current_wave == 'C' and next_wave == '1':
            # End of correction, new impulse wave starting
            signal_type = 'BUY'
            signal_strength = 7
            details.append("Corrective pattern completing, new impulse cycle starting")
            
        elif current_wave in ['1', '3', '5']:
            # Impulse waves - trend continuation
            trend_direction = wave_analysis.get('current_wave_info', {}).get('direction', 'Unknown')
            if trend_direction == 'Upward':
                signal_type = 'BUY'
                signal_strength = 5
                details.append(f"Impulse Wave {current_wave} in uptrend")
            elif trend_direction == 'Downward':
                signal_type = 'SELL'
                signal_strength = 5
                details.append(f"Impulse Wave {current_wave} in downtrend")
        
        # Adjust strength based on wave confidence
        signal_strength = signal_strength * wave_confidence
        
        return {
            'type': signal_type,
            'strength': round(signal_strength, 1),
            'confidence': wave_confidence,
            'details': details,
            'current_wave': current_wave,
            'next_wave': next_wave
        }
    
    def _analyze_fibonacci_signal(self, fibonacci_levels: Dict, current_price: float) -> Dict:
        """Analyze Fibonacci levels for trading signals"""
        
        if not fibonacci_levels or not fibonacci_levels.get('levels'):
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'No Fibonacci data'}
        
        levels = fibonacci_levels['levels']
        key_levels = fibonacci_levels.get('key_levels', {})
        
        signal_type = 'HOLD'
        signal_strength = 0
        details = []
        
        # Find nearby Fibonacci levels
        nearby_levels = [l for l in levels if abs(l['price'] - current_price) / current_price < 0.02]  # Within 2%
        
        if nearby_levels:
            # Near key Fibonacci level
            for level in nearby_levels:
                if level['type'] == 'retracement':
                    if level['percentage'] in [38.2, 50.0, 61.8]:  # Key retracement levels
                        if current_price < level['price']:  # Price below resistance
                            signal_type = 'SELL'
                            signal_strength = 4
                            details.append(f"Near Fibonacci resistance at {level['percentage']:.1f}%")
                        else:  # Price above support
                            signal_type = 'BUY'
                            signal_strength = 4
                            details.append(f"Above Fibonacci support at {level['percentage']:.1f}%")
        
        # Check key levels
        nearest_support = key_levels.get('nearest_support')
        nearest_resistance = key_levels.get('nearest_resistance')
        
        if nearest_support and nearest_resistance:
            support_distance = abs(current_price - nearest_support['price']) / current_price
            resistance_distance = abs(current_price - nearest_resistance['price']) / current_price
            
            if support_distance < 0.01:  # Very close to support
                signal_type = 'BUY'
                signal_strength = max(signal_strength, 5)
                details.append("Price near strong Fibonacci support")
                
            elif resistance_distance < 0.01:  # Very close to resistance
                signal_type = 'SELL'
                signal_strength = max(signal_strength, 5)
                details.append("Price near strong Fibonacci resistance")
        
        confidence = min(len(details) * 0.3, 1.0)
        
        return {
            'type': signal_type,
            'strength': round(signal_strength, 1),
            'confidence': confidence,
            'details': details,
            'nearby_levels': nearby_levels
        }
    
    def _analyze_trend_signal(self, price_data: pd.DataFrame) -> Dict:
        """Analyze overall trend for trading signals"""
        
        if len(price_data) < 20:
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'Insufficient data'}
        
        # Calculate multiple timeframe trends
        short_trend = self._calculate_trend(price_data.tail(10))
        medium_trend = self._calculate_trend(price_data.tail(20))
        long_trend = self._calculate_trend(price_data.tail(50)) if len(price_data) >= 50 else medium_trend
        
        # Determine signal based on trend alignment
        trends = [short_trend, medium_trend, long_trend]
        bullish_trends = sum(1 for t in trends if t > 0)
        bearish_trends = sum(1 for t in trends if t < 0)
        
        signal_type = 'HOLD'
        signal_strength = 0
        details = []
        
        if bullish_trends >= 2:
            signal_type = 'BUY'
            signal_strength = bullish_trends * 2
            details.append(f"Bullish trend alignment ({bullish_trends}/3 timeframes)")
            
        elif bearish_trends >= 2:
            signal_type = 'SELL'
            signal_strength = bearish_trends * 2
            details.append(f"Bearish trend alignment ({bearish_trends}/3 timeframes)")
        
        # Add trend strength
        avg_trend_strength = np.mean([abs(t) for t in trends])
        signal_strength = signal_strength * avg_trend_strength
        
        confidence = max(bullish_trends, bearish_trends) / 3
        
        return {
            'type': signal_type,
            'strength': round(signal_strength, 1),
            'confidence': confidence,
            'details': details,
            'trend_values': {
                'short': round(short_trend, 3),
                'medium': round(medium_trend, 3),
                'long': round(long_trend, 3)
            }
        }
    
    def _analyze_momentum_signal(self, price_data: pd.DataFrame) -> Dict:
        """Analyze momentum indicators for trading signals"""
        
        if len(price_data) < 14:
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'Insufficient data'}
        
        # Calculate RSI
        rsi = self._calculate_rsi(price_data['close'])
        
        # Calculate price momentum
        price_momentum = self._calculate_price_momentum(price_data['close'])
        
        signal_type = 'HOLD'
        signal_strength = 0
        details = []
        
        # RSI signals
        if rsi < 30:
            signal_type = 'BUY'
            signal_strength = 3
            details.append(f"RSI oversold ({rsi:.1f})")
        elif rsi > 70:
            signal_type = 'SELL'
            signal_strength = 3
            details.append(f"RSI overbought ({rsi:.1f})")
        
        # Momentum signals
        if price_momentum > 0.02:  # Strong positive momentum
            if signal_type == 'BUY':
                signal_strength += 2
            details.append("Strong bullish momentum")
        elif price_momentum < -0.02:  # Strong negative momentum
            if signal_type == 'SELL':
                signal_strength += 2
            details.append("Strong bearish momentum")
        
        confidence = 0.5 if details else 0.2
        
        return {
            'type': signal_type,
            'strength': round(signal_strength, 1),
            'confidence': confidence,
            'details': details,
            'rsi': round(rsi, 1),
            'momentum': round(price_momentum, 4)
        }
    
    def _combine_signals(self, wave_signal: Dict, fibonacci_signal: Dict, 
                        trend_signal: Dict, momentum_signal: Dict, channel_signal: Dict = None, news_signal: Dict = None) -> Dict:
        """Combine all signal components into final signal"""
        
        # Calculate weighted scores
        buy_score = 0
        sell_score = 0
        
        signals = [
            (wave_signal, self.signal_weights['elliott_wave']),
            (fibonacci_signal, self.signal_weights['fibonacci']),
            (trend_signal, self.signal_weights['trend']),
            (momentum_signal, self.signal_weights['momentum'])
        ]
        
        # Add channel signal if available
        if channel_signal:
            signals.append((channel_signal, 0.15))
        
        # Add news signal if available
        if news_signal:
            signals.append((news_signal, 0.20))
        
        for signal, weight in signals:
            strength = signal.get('strength', 0) * signal.get('confidence', 0)
            
            if signal.get('type') == 'BUY':
                buy_score += strength * weight
            elif signal.get('type') == 'SELL':
                sell_score += strength * weight
        
        # Determine final signal
        if buy_score > sell_score and buy_score > 2:
            signal_type = 'BUY'
            signal_strength = buy_score
        elif sell_score > buy_score and sell_score > 2:
            signal_type = 'SELL'
            signal_strength = sell_score
        else:
            signal_type = 'HOLD'
            signal_strength = max(buy_score, sell_score)
        
        # Calculate overall confidence
        confidence_scores = [s.get('confidence', 0) for s, _ in signals]
        overall_confidence = np.mean(confidence_scores)
        
        return {
            'type': signal_type,
            'strength': round(min(signal_strength, 10), 1),
            'confidence': round(overall_confidence, 2),
            'buy_score': round(buy_score, 2),
            'sell_score': round(sell_score, 2)
        }
    
    def _analyze_channel_signal(self, channel_analysis: Dict, current_price: float) -> Dict:
        """Analyze price channel patterns for trading signals"""
        
        if not channel_analysis:
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'No channel data'}
        
        channel_position = channel_analysis.get('channel_position', {})
        breakout_signals = channel_analysis.get('breakout_signals', {})
        trend_channel = channel_analysis.get('trend_channel', {})
        
        signal_type = 'HOLD'
        strength = 0
        confidence = 0
        details = []
        
        # Analyze channel position
        position = channel_position.get('position', 'Within channel')
        zone = channel_position.get('zone', 'Neutral zone')
        percentage = channel_position.get('percentage', 50)
        
        if position == 'Above channel':
            signal_type = 'SELL'
            strength = min(8, 5 + (percentage - 100) / 10)
            confidence = 0.75
            details.append('Price broke above channel resistance')
        elif position == 'Below channel':
            signal_type = 'BUY'
            strength = min(8, 5 + abs(percentage) / 10)
            confidence = 0.75
            details.append('Price broke below channel support')
        elif zone == 'Oversold zone':
            signal_type = 'BUY'
            strength = 6
            confidence = 0.6
            details.append('Price in oversold zone of channel')
        elif zone == 'Overbought zone':
            signal_type = 'SELL'
            strength = 6
            confidence = 0.6
            details.append('Price in overbought zone of channel')
        
        # Check for breakout signals
        active_signals = breakout_signals.get('active_signals', [])
        for signal in active_signals:
            if signal['type'] == 'Bullish Breakout':
                signal_type = 'BUY'
                strength = max(strength, 7)
                confidence = max(confidence, 0.8)
                details.append('Bullish breakout detected')
            elif signal['type'] == 'Bearish Breakout':
                signal_type = 'SELL'
                strength = max(strength, 7)
                confidence = max(confidence, 0.8)
                details.append('Bearish breakout detected')
        
        # Consider trend direction
        trend_direction = trend_channel.get('trend_direction', 'Unknown')
        if trend_direction == 'Upward' and signal_type == 'BUY':
            strength += 1
            confidence += 0.1
            details.append('Aligned with upward channel trend')
        elif trend_direction == 'Downward' and signal_type == 'SELL':
            strength += 1
            confidence += 0.1
            details.append('Aligned with downward channel trend')
        
        return {
            'type': signal_type,
            'strength': min(strength, 10),
            'confidence': min(confidence, 1.0),
            'details': '; '.join(details) if details else 'Channel analysis neutral',
            'channel_position': position,
            'zone': zone
        }
    
    def _analyze_news_signal(self, news_analysis: Dict, current_price: float) -> Dict:
        """Analyze news sentiment for trading signals"""
        
        if not news_analysis:
            return {'type': 'HOLD', 'strength': 0, 'confidence': 0, 'details': 'No news data'}
        
        news_signals = news_analysis.get('news_signals', {})
        sentiment_analysis = news_analysis.get('sentiment_analysis', {})
        market_impact = news_analysis.get('market_impact', {})
        
        signal_type = news_signals.get('signal', 'HOLD')
        signal_strength = news_signals.get('strength', 0)
        signal_confidence = news_signals.get('confidence', 0)
        
        # Get sentiment details
        overall_sentiment = sentiment_analysis.get('overall_sentiment', 0)
        sentiment_trend = sentiment_analysis.get('trend', 'Neutral')
        news_volume = sentiment_analysis.get('news_volume', 0)
        
        # Get impact details
        impact_level = market_impact.get('overall_impact', 'Low')
        volatility_expectation = market_impact.get('volatility_expectation', 'Low')
        
        # Adjust signal based on news volume and impact
        if news_volume >= 5 and impact_level == 'High':
            signal_strength *= 1.3
            signal_confidence *= 1.2
        elif news_volume >= 3 and impact_level == 'Medium':
            signal_strength *= 1.1
            signal_confidence *= 1.1
        
        # Generate signal details
        details = []
        if sentiment_trend == 'Bullish':
            details.append(f"Positive news sentiment ({overall_sentiment:.2f})")
        elif sentiment_trend == 'Bearish':
            details.append(f"Negative news sentiment ({overall_sentiment:.2f})")
        else:
            details.append("Neutral news sentiment")
        
        details.append(f"{impact_level} market impact expected")
        if volatility_expectation == 'High':
            details.append("High volatility expected")
        
        return {
            'type': signal_type,
            'strength': min(signal_strength, 10),
            'confidence': min(signal_confidence, 1.0),
            'details': '; '.join(details),
            'sentiment_trend': sentiment_trend,
            'market_impact': impact_level,
            'news_volume': news_volume,
            'time_horizon': news_signals.get('time_horizon', 'Medium-term')
        }
    
    def _calculate_entry_exit_levels(self, current_price: float, signal: Dict, 
                                   fibonacci_levels: Dict, wave_analysis: Dict) -> Dict:
        """Calculate entry, stop loss, and take profit levels"""
        
        levels = {}
        signal_type = signal.get('type', 'HOLD')
        
        if signal_type == 'HOLD':
            return levels
        
        # Base risk percentage
        risk_pct = 0.02  # 2% risk
        reward_ratio = 2.0  # 1:2 risk-reward
        
        if signal_type == 'BUY':
            # Entry at current price or slight pullback
            levels['entry'] = current_price * 0.999  # Slight discount
            
            # Stop loss below recent support or Fibonacci level
            stop_loss = self._find_support_level(current_price, fibonacci_levels)
            if stop_loss:
                levels['stop_loss'] = stop_loss
            else:
                levels['stop_loss'] = current_price * (1 - risk_pct)
            
            # Take profit above resistance or using risk-reward ratio
            take_profit = self._find_resistance_level(current_price, fibonacci_levels)
            if take_profit:
                levels['take_profit'] = take_profit
            else:
                risk_amount = levels['entry'] - levels['stop_loss']
                levels['take_profit'] = levels['entry'] + (risk_amount * reward_ratio)
        
        elif signal_type == 'SELL':
            # Entry at current price or slight bounce
            levels['entry'] = current_price * 1.001  # Slight premium
            
            # Stop loss above recent resistance or Fibonacci level
            stop_loss = self._find_resistance_level(current_price, fibonacci_levels)
            if stop_loss:
                levels['stop_loss'] = stop_loss
            else:
                levels['stop_loss'] = current_price * (1 + risk_pct)
            
            # Take profit below support or using risk-reward ratio
            take_profit = self._find_support_level(current_price, fibonacci_levels)
            if take_profit:
                levels['take_profit'] = take_profit
            else:
                risk_amount = levels['stop_loss'] - levels['entry']
                levels['take_profit'] = levels['entry'] - (risk_amount * reward_ratio)
        
        # Round all levels appropriately
        for key, value in levels.items():
            levels[key] = round(value, 2)
        
        return levels
    
    def _find_support_level(self, current_price: float, fibonacci_levels: Dict) -> Optional[float]:
        """Find nearest support level below current price"""
        
        if not fibonacci_levels or not fibonacci_levels.get('levels'):
            return None
        
        support_levels = [
            l for l in fibonacci_levels['levels'] 
            if l['price'] < current_price and l.get('level_type') == 'support'
        ]
        
        if support_levels:
            # Return the highest support level (nearest to current price)
            return max(support_levels, key=lambda x: x['price'])['price']
        
        return None
    
    def _find_resistance_level(self, current_price: float, fibonacci_levels: Dict) -> Optional[float]:
        """Find nearest resistance level above current price"""
        
        if not fibonacci_levels or not fibonacci_levels.get('levels'):
            return None
        
        resistance_levels = [
            l for l in fibonacci_levels['levels'] 
            if l['price'] > current_price and l.get('level_type') == 'resistance'
        ]
        
        if resistance_levels:
            # Return the lowest resistance level (nearest to current price)
            return min(resistance_levels, key=lambda x: x['price'])['price']
        
        return None
    
    def _calculate_risk_management(self, current_price: float, levels: Dict, 
                                 risk_level: str) -> Dict:
        """Calculate risk management parameters"""
        
        risk_multiplier = self.risk_levels.get(risk_level, 1.0)
        
        if not levels:
            return {'position_size': 0, 'max_risk': 0}
        
        entry = levels.get('entry', current_price)
        stop_loss = levels.get('stop_loss')
        
        if not stop_loss:
            return {'position_size': 0, 'max_risk': 0}
        
        # Calculate risk per unit
        risk_per_unit = abs(entry - stop_loss)
        
        # Standard position sizing (2% account risk)
        account_risk_pct = 0.02 * risk_multiplier
        
        return {
            'position_size': round(account_risk_pct / (risk_per_unit / entry), 4),
            'max_risk': round(risk_per_unit, 2),
            'risk_reward_ratio': round(abs(levels.get('take_profit', entry) - entry) / risk_per_unit, 2) if levels.get('take_profit') else 0,
            'account_risk_pct': account_risk_pct * 100
        }
    
    def _generate_signal_reasoning(self, wave_signal: Dict, fibonacci_signal: Dict, 
                                 trend_signal: Dict, momentum_signal: Dict) -> str:
        """Generate human-readable reasoning for the signal"""
        
        reasoning_parts = []
        
        # Elliott Wave reasoning
        if wave_signal.get('details'):
            reasoning_parts.append(f"Elliott Wave: {', '.join(wave_signal['details'])}")
        
        # Fibonacci reasoning
        if fibonacci_signal.get('details'):
            reasoning_parts.append(f"Fibonacci: {', '.join(fibonacci_signal['details'])}")
        
        # Trend reasoning
        if trend_signal.get('details'):
            reasoning_parts.append(f"Trend: {', '.join(trend_signal['details'])}")
        
        # Momentum reasoning
        if momentum_signal.get('details'):
            reasoning_parts.append(f"Momentum: {', '.join(momentum_signal['details'])}")
        
        if not reasoning_parts:
            return "No clear directional signals detected. Consider waiting for better setup."
        
        return " | ".join(reasoning_parts)
    
    def _calculate_trend(self, price_data: pd.DataFrame) -> float:
        """Calculate trend strength and direction"""
        
        if len(price_data) < 2:
            return 0
        
        closes = price_data['close']
        return (closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0]
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI
        
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gains = gains.rolling(window=period).mean().iloc[-1]
        avg_losses = losses.rolling(window=period).mean().iloc[-1]
        
        if avg_losses == 0:
            return 100.0
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_price_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """Calculate price momentum"""
        
        if len(prices) < period + 1:
            return 0
        
        return (prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]
    
    def _get_recent_signals(self, price_data: pd.DataFrame, wave_analysis: Dict) -> List[Dict]:
        """Get history of recent signals for context"""
        
        # This would typically store and retrieve historical signals
        # For now, return a simple placeholder
        return [
            {
                'timestamp': price_data.index[-1].isoformat() if len(price_data) > 0 else datetime.now().isoformat(),
                'signal': 'Current analysis',
                'wave': wave_analysis.get('current_wave', 'Unknown')
            }
        ]
    
    def _analyze_market_context(self, price_data: pd.DataFrame, wave_analysis: Dict) -> Dict:
        """Analyze broader market context"""
        
        if len(price_data) < 20:
            return {'volatility': 'Unknown', 'trend_strength': 'Unknown'}
        
        # Calculate volatility
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Classify volatility
        if volatility < 0.15:
            vol_description = "Low"
        elif volatility < 0.25:
            vol_description = "Moderate"
        else:
            vol_description = "High"
        
        # Trend strength
        price_change = (price_data['close'].iloc[-1] - price_data['close'].iloc[-20]) / price_data['close'].iloc[-20]
        
        if abs(price_change) < 0.02:
            trend_strength = "Weak"
        elif abs(price_change) < 0.05:
            trend_strength = "Moderate" 
        else:
            trend_strength = "Strong"
        
        return {
            'volatility': f"{vol_description} ({volatility:.1%})",
            'trend_strength': trend_strength,
            'price_change_20d': f"{price_change:.1%}",
            'wave_cycle_position': wave_analysis.get('current_wave', 'Unknown')
        }
    
    def _generate_default_signal(self) -> Dict:
        """Generate default signal when analysis fails"""
        
        return {
            'current_signal': {
                'type': 'HOLD',
                'strength': 0,
                'confidence': 0,
                'reasoning': 'Unable to generate signals due to insufficient data'
            },
            'component_signals': {},
            'risk_management': {},
            'signal_history': [],
            'market_context': {}
        }
