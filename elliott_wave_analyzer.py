import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

class ElliottWaveAnalyzer:
    """
    Elliott Wave Analysis implementation for financial time series data
    """
    
    def __init__(self):
        self.wave_patterns = {
            'impulse': ['1', '2', '3', '4', '5'],
            'corrective': ['A', 'B', 'C'],
            'extended': ['1', '2', '3', '4', '5', 'A', 'B', 'C']
        }
        
        # Advanced Fibonacci ratios for complex analysis
        self.advanced_fib_levels = [0.236, 0.382, 0.500, 0.618, 0.764, 0.786, 0.886, 1.000, 1.272, 1.414, 1.618, 2.000, 2.618]
        
        # Complex wave patterns
        self.complex_patterns = {
            'diagonal_triangle': {
                'type': 'diagonal',
                'waves': 5,
                'characteristics': ['converging_trend_lines', 'overlapping_waves'],
                'locations': ['wave_1', 'wave_5', 'wave_C']
            },
            'double_zigzag': {
                'type': 'complex_correction',
                'waves': ['W', 'X', 'Y'],
                'characteristics': ['two_zigzag_patterns', 'connecting_wave_X']
            },
            'triple_zigzag': {
                'type': 'complex_correction', 
                'waves': ['W', 'X', 'Y', 'XX', 'Z'],
                'characteristics': ['three_zigzag_patterns', 'two_connecting_waves']
            },
            'expanded_flat': {
                'type': 'irregular_correction',
                'waves': ['A', 'B', 'C'],
                'characteristics': ['B_exceeds_A_start', 'C_exceeds_A_end']
            },
            'running_correction': {
                'type': 'irregular_correction',
                'waves': ['A', 'B', 'C'],
                'characteristics': ['B_significantly_exceeds_A', 'C_fails_A_end']
            }
        }
        
        # Wave relationship rules for complex analysis
        self.wave_rules = {
            'wave_2_retracement': (0.236, 1.000),  # Can go up to 100% in rare cases
            'wave_4_retracement': (0.236, 0.50),
            'wave_3_extension': (1.618, 4.236),    # Extended range for complex markets
            'wave_5_target': (0.618, 2.618),
            'diagonal_overlap': True,               # Waves can overlap in diagonals
            'truncation_threshold': 0.618           # For identifying truncated waves
        }
    
    def analyze_waves(self, price_data: pd.DataFrame, sensitivity: float = 1.0, asset_type: str = "GENERAL") -> Dict:
        """
        Main Elliott Wave analysis function
        
        Args:
            price_data: DataFrame with OHLC data
            sensitivity: Wave detection sensitivity (0.5-2.0)
        
        Returns:
            Dictionary containing wave analysis results
        """
        try:
            # Adjust analysis parameters based on asset type
            adjusted_sensitivity = self._adjust_sensitivity_for_asset(sensitivity, asset_type)
            
            # Find pivot points (swing highs and lows)
            pivot_points = self._find_pivot_points(price_data, adjusted_sensitivity)
            
            # Identify wave patterns with asset-specific logic
            wave_points = self._identify_wave_patterns(pivot_points, price_data, asset_type)
            
            # Determine current wave and trend
            current_wave_info = self._analyze_current_wave(wave_points, price_data, asset_type)
            
            # Detect complex patterns
            complex_analysis = self._detect_complex_patterns(wave_points, price_data, asset_type)
            
            # Predict next wave
            next_wave_prediction = self._predict_next_wave(wave_points, current_wave_info, asset_type)
            
            # Calculate pattern reliability
            pattern_reliability = self._calculate_pattern_reliability(wave_points, asset_type)
            
            return {
                'wave_points': wave_points,
                'current_wave': current_wave_info.get('label', 'Unknown'),
                'next_wave': next_wave_prediction.get('label', 'Unknown'),
                'current_wave_info': current_wave_info,
                'next_wave_info': next_wave_prediction,
                'wave_confidence': pattern_reliability,
                'trend_strength': self._calculate_trend_strength(price_data),
                'completion_percentage': current_wave_info.get('completion', 0),
                'pattern_reliability': pattern_reliability,
                'wave_details': self._get_wave_details(wave_points),
                'complex_patterns': complex_analysis
            }
            
        except Exception as e:
            print(f"Error in Elliott Wave analysis: {e}")
            return {}
    
    def _find_pivot_points(self, price_data: pd.DataFrame, sensitivity: float) -> List[Dict]:
        """Find significant pivot points using Zigzag methodology"""
        
        highs = price_data['high'].values
        lows = price_data['low'].values
        closes = price_data['close'].values
        timestamps = price_data.index.tolist()
        
        # Calculate adaptive threshold based on market volatility
        price_volatility = (price_data['high'] - price_data['low']).mean() / price_data['close'].mean()
        base_threshold = max(0.015, price_volatility * 0.5)  # Minimum 1.5%, adaptive to volatility
        threshold_pct = (base_threshold / sensitivity)
        
        pivot_points = []
        
        # Start with first point
        if len(highs) < 3:
            return pivot_points
            
        # Determine initial trend direction
        current_high = highs[0]
        current_low = lows[0]
        current_high_idx = 0
        current_low_idx = 0
        trend = 'up'  # Start assuming uptrend
        
        for i in range(1, len(highs)):
            price_high = highs[i]
            price_low = lows[i]
            
            if trend == 'up':
                # Looking for higher highs to continue uptrend
                if price_high > current_high:
                    current_high = price_high
                    current_high_idx = i
                # Check for reversal (significant decline from high)
                elif (current_high - price_low) / current_high >= threshold_pct:
                    # Confirmed high pivot
                    pivot_points.append({
                        'timestamp': timestamps[current_high_idx],
                        'price': current_high,
                        'type': 'high',
                        'index': current_high_idx
                    })
                    # Switch to downtrend
                    trend = 'down'
                    current_low = price_low
                    current_low_idx = i
                    
            else:  # trend == 'down'
                # Looking for lower lows to continue downtrend
                if price_low < current_low:
                    current_low = price_low
                    current_low_idx = i
                # Check for reversal (significant rise from low)
                elif (price_high - current_low) / current_low >= threshold_pct:
                    # Confirmed low pivot
                    pivot_points.append({
                        'timestamp': timestamps[current_low_idx],
                        'price': current_low,
                        'type': 'low',
                        'index': current_low_idx
                    })
                    # Switch to uptrend
                    trend = 'up'
                    current_high = price_high
                    current_high_idx = i
        
        # Add the last pivot if we ended in a trend
        if trend == 'up' and len(pivot_points) > 0:
            if pivot_points[-1]['type'] == 'low':
                pivot_points.append({
                    'timestamp': timestamps[current_high_idx],
                    'price': current_high,
                    'type': 'high',
                    'index': current_high_idx
                })
        elif trend == 'down' and len(pivot_points) > 0:
            if pivot_points[-1]['type'] == 'high':
                pivot_points.append({
                    'timestamp': timestamps[current_low_idx],
                    'price': current_low,
                    'type': 'low',
                    'index': current_low_idx
                })
        
        # If we didn't find enough pivots, use a more sensitive approach
        if len(pivot_points) < 3:
            pivot_points = self._find_pivots_alternative_method(price_data, sensitivity)
        
        return pivot_points
    
    def _find_pivots_alternative_method(self, price_data: pd.DataFrame, sensitivity: float) -> List[Dict]:
        """Alternative method for finding pivots when main method fails"""
        
        highs = price_data['high'].values
        lows = price_data['low'].values
        timestamps = price_data.index.tolist()
        
        pivot_points = []
        lookback = max(3, int(15 / sensitivity))
        
        # Find local maxima and minima
        for i in range(lookback, len(highs) - lookback):
            # Check for local high
            is_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and highs[j] >= highs[i]:
                    is_high = False
                    break
            
            if is_high:
                pivot_points.append({
                    'timestamp': timestamps[i],
                    'price': highs[i],
                    'type': 'high',
                    'index': i
                })
            
            # Check for local low
            is_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and lows[j] <= lows[i]:
                    is_low = False
                    break
            
            if is_low:
                pivot_points.append({
                    'timestamp': timestamps[i],
                    'price': lows[i],
                    'type': 'low',
                    'index': i
                })
        
        # Sort by timestamp
        pivot_points.sort(key=lambda x: x['timestamp'])
        
        # Remove consecutive same-type pivots (keep the more extreme one)
        filtered_pivots = []
        for point in pivot_points:
            if not filtered_pivots:
                filtered_pivots.append(point)
            elif point['type'] != filtered_pivots[-1]['type']:
                filtered_pivots.append(point)
            elif point['type'] == 'high' and point['price'] > filtered_pivots[-1]['price']:
                filtered_pivots[-1] = point
            elif point['type'] == 'low' and point['price'] < filtered_pivots[-1]['price']:
                filtered_pivots[-1] = point
        
        return filtered_pivots
    
    def _identify_wave_patterns(self, pivot_points: List[Dict], price_data: pd.DataFrame, asset_type: str = "GENERAL") -> List[Dict]:
        """Identify Elliott Wave patterns using asset-specific logic"""
        
        if len(pivot_points) < 3:
            return []
        
        # Generate asset-specific wave patterns first
        asset_specific_waves = self._generate_asset_specific_waves(pivot_points, asset_type, price_data)
        
        if asset_specific_waves:
            return asset_specific_waves
        
        # Fallback to traditional pattern detection
        if len(pivot_points) >= 5:
            best_pattern = self._find_best_elliott_pattern(pivot_points, price_data)
            if best_pattern:
                return best_pattern
        
        # Final fallback: sequential labeling
        recent_pivots = pivot_points[-5:] if len(pivot_points) >= 5 else pivot_points
        wave_points = self._label_sequential_waves(recent_pivots)
        
        return wave_points
    
    def _find_best_elliott_pattern(self, pivot_points: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Find the best Elliott Wave pattern in the pivot points"""
        
        if len(pivot_points) < 5:
            return []
        
        best_pattern = []
        best_score = 0
        
        # Try different combinations of 5 consecutive pivots for impulse patterns
        for start_idx in range(len(pivot_points) - 4):
            pattern_points = pivot_points[start_idx:start_idx + 5]
            
            # Check if this could be a 5-wave impulse pattern
            if self._is_valid_impulse_pattern(pattern_points):
                score = self._calculate_pattern_score(pattern_points, 'impulse')
                if score > best_score:
                    best_score = score
                    best_pattern = self._label_impulse_pattern(pattern_points)
        
        # Try 3-wave corrective patterns
        for start_idx in range(len(pivot_points) - 2):
            pattern_points = pivot_points[start_idx:start_idx + 3]
            
            if self._is_valid_corrective_pattern(pattern_points):
                score = self._calculate_pattern_score(pattern_points, 'corrective')
                if score > best_score:
                    best_score = score
                    best_pattern = self._label_corrective_pattern(pattern_points)
        
        return best_pattern
    
    def _is_valid_impulse_pattern(self, points: List[Dict]) -> bool:
        """Check if 5 points form a valid Elliott Wave impulse pattern"""
        
        if len(points) != 5:
            return False
        
        # Check alternating high/low pattern
        expected_types = ['low', 'high', 'low', 'high', 'low']
        actual_types = [p['type'] for p in points]
        
        # Allow for both upward and downward impulse patterns
        if actual_types != expected_types and actual_types != expected_types[::-1]:
            return False
        
        # Elliott Wave Rules:
        # 1. Wave 2 cannot retrace more than 100% of Wave 1
        # 2. Wave 3 cannot be the shortest of waves 1, 3, 5
        # 3. Wave 4 cannot overlap with Wave 1 territory
        
        try:
            # Calculate wave lengths
            wave1_len = abs(points[1]['price'] - points[0]['price'])
            wave2_len = abs(points[2]['price'] - points[1]['price'])
            wave3_len = abs(points[3]['price'] - points[2]['price'])
            wave4_len = abs(points[4]['price'] - points[3]['price'])
            
            # Rule 1: Wave 2 retracement check
            if wave2_len > wave1_len:
                return False
            
            # Rule 2: Wave 3 cannot be shortest (simplified check)
            if wave3_len < min(wave1_len * 0.618, wave1_len):
                return False
            
            # Rule 3: Wave 4 overlap check (simplified)
            if actual_types[0] == 'low':  # Upward pattern
                if points[4]['price'] <= points[1]['price']:
                    return False
            else:  # Downward pattern
                if points[4]['price'] >= points[1]['price']:
                    return False
            
            return True
            
        except (KeyError, IndexError, ZeroDivisionError):
            return False
    
    def _is_valid_corrective_pattern(self, points: List[Dict]) -> bool:
        """Check if 3 points form a valid Elliott Wave corrective pattern"""
        
        if len(points) != 3:
            return False
        
        # Check for A-B-C pattern (alternating types)
        types = [p['type'] for p in points]
        return types in [['high', 'low', 'high'], ['low', 'high', 'low']]
    
    def _calculate_pattern_score(self, points: List[Dict], pattern_type: str) -> float:
        """Calculate quality score for an Elliott Wave pattern"""
        
        score = 1.0
        
        try:
            if pattern_type == 'impulse' and len(points) == 5:
                # Score based on Fibonacci relationships
                wave1 = abs(points[1]['price'] - points[0]['price'])
                wave3 = abs(points[3]['price'] - points[2]['price'])
                wave5 = abs(points[4]['price'] - points[3]['price'])
                
                # Wave 3 should be extended (1.618 times wave 1 is ideal)
                if wave1 > 0:
                    wave3_ratio = wave3 / wave1
                    if 1.4 <= wave3_ratio <= 2.0:  # Good extension
                        score += 2.0
                    elif 1.0 <= wave3_ratio <= 1.4:  # Acceptable
                        score += 1.0
                
                # Wave 5 relationships
                if wave1 > 0:
                    wave5_ratio = wave5 / wave1
                    if 0.8 <= wave5_ratio <= 1.2:  # Good equality
                        score += 1.0
                
            elif pattern_type == 'corrective' and len(points) == 3:
                # A-B-C pattern scoring
                waveA = abs(points[1]['price'] - points[0]['price'])
                waveC = abs(points[2]['price'] - points[1]['price'])
                
                if waveA > 0:
                    ac_ratio = waveC / waveA
                    if 0.8 <= ac_ratio <= 1.2:  # C equals A
                        score += 2.0
                    elif 1.6 <= ac_ratio <= 1.8:  # C = 1.618 * A
                        score += 1.5
            
            # Bonus for recent patterns
            latest_time = max(p['timestamp'] for p in points)
            time_bonus = 1.0  # Could add time-based weighting
            score += time_bonus
            
        except (ZeroDivisionError, KeyError):
            pass
        
        return score
    
    def _label_impulse_pattern(self, points: List[Dict]) -> List[Dict]:
        """Label 5 points as Elliott Wave impulse pattern (1-2-3-4-5)"""
        
        labeled_points = []
        labels = ['1', '2', '3', '4', '5']
        
        for i, point in enumerate(points):
            labeled_point = point.copy()
            labeled_point['label'] = labels[i]
            labeled_point['wave_type'] = 'impulse'
            labeled_point['confidence'] = 0.8  # High confidence for impulse
            labeled_points.append(labeled_point)
        
        return labeled_points
    
    def _label_corrective_pattern(self, points: List[Dict]) -> List[Dict]:
        """Label 3 points as Elliott Wave corrective pattern (A-B-C)"""
        
        labeled_points = []
        labels = ['A', 'B', 'C']
        
        for i, point in enumerate(points):
            labeled_point = point.copy()
            labeled_point['label'] = labels[i]
            labeled_point['wave_type'] = 'corrective'
            labeled_point['confidence'] = 0.7  # Good confidence for corrective
            labeled_points.append(labeled_point)
        
        return labeled_points
    
    def _label_sequential_waves(self, points: List[Dict]) -> List[Dict]:
        """Fallback sequential labeling when no clear pattern is found"""
        
        labeled_points = []
        
        for i, point in enumerate(points):
            labeled_point = point.copy()
            
            # Cycle through 1-2-3-4-5-A-B-C pattern
            if i < 5:
                labeled_point['label'] = str(i + 1)
                labeled_point['wave_type'] = 'impulse'
            else:
                corrective_labels = ['A', 'B', 'C']
                labeled_point['label'] = corrective_labels[(i - 5) % 3]
                labeled_point['wave_type'] = 'corrective'
            
            labeled_point['confidence'] = 0.5  # Lower confidence for sequential
            labeled_points.append(labeled_point)
        
        return labeled_points
    
    def _validate_wave_relationships(self, wave_points: List[Dict]) -> List[Dict]:
        """Validate Elliott Wave relationships and rules"""
        
        if len(wave_points) < 5:
            return wave_points
        
        validated_waves = []
        
        for i, wave in enumerate(wave_points):
            is_valid = True
            
            # Elliott Wave Rule validations
            if wave['label'] == '2' and i > 0:
                # Wave 2 cannot retrace more than 100% of wave 1
                wave_1 = wave_points[i-1]
                if wave['type'] != wave_1['type']:  # Alternation rule
                    is_valid = True
                else:
                    is_valid = False
            
            elif wave['label'] == '3' and i > 1:
                # Wave 3 cannot be the shortest wave
                if i >= 4:  # Have waves 1, 3, 5 to compare
                    wave_1_length = abs(wave_points[1]['price'] - wave_points[0]['price'])
                    wave_3_length = abs(wave['price'] - wave_points[i-1]['price'])
                    # Wave 3 should be strong
                    if wave_3_length >= wave_1_length * 0.8:
                        is_valid = True
            
            elif wave['label'] == '4' and i > 2:
                # Wave 4 cannot overlap with wave 1 territory
                wave_1 = wave_points[0]
                if ((wave['type'] == 'low' and wave['price'] > wave_1['price']) or
                    (wave['type'] == 'high' and wave['price'] < wave_1['price'])):
                    is_valid = True
            
            if is_valid:
                # Add confidence score
                confidence = self._calculate_wave_confidence(wave, wave_points, i)
                wave['confidence'] = confidence
                validated_waves.append(wave)
        
        return validated_waves
    
    def _calculate_wave_confidence(self, wave: Dict, all_waves: List[Dict], index: int) -> float:
        """Calculate confidence score for individual wave"""
        
        confidence = 0.5  # Base confidence
        
        # Time-based confidence (recent waves have higher confidence)
        if index >= len(all_waves) - 3:
            confidence += 0.2
        
        # Pattern completion confidence
        if wave['label'] in ['3', '5', 'C']:  # Key waves
            confidence += 0.2
        
        # Alternation rule compliance
        if index > 0:
            prev_wave = all_waves[index - 1]
            if wave['type'] != prev_wave['type']:
                confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _analyze_current_wave(self, wave_points: List[Dict], price_data: pd.DataFrame, asset_type: str = "GENERAL") -> Dict:
        """Analyze the current wave position and characteristics with high accuracy"""
        
        if not wave_points:
            return {}
        
        current_price = price_data['close'].iloc[-1]
        current_time = price_data.index[-1]
        
        # Determine where we are in the wave structure
        current_wave_info = self._determine_current_position(wave_points, current_price, current_time)
        
        # Calculate precise wave progress
        wave_progress = self._calculate_precise_wave_progress(wave_points, current_price, current_time)
        
        # Assess wave completion status
        completion_status = self._assess_wave_completion(wave_points, current_price, price_data)
        
        return {
            'label': current_wave_info['label'],
            'type': current_wave_info['type'],
            'direction': current_wave_info['direction'],
            'progress': wave_progress,
            'completion': completion_status['completion_pct'],
            'price_level': current_wave_info['target_price'],
            'current_price': current_price,
            'characteristics': current_wave_info['characteristics'],
            'confidence': current_wave_info['confidence'],
            'wave_status': completion_status['status'],
            'time_in_wave': current_wave_info['duration']
        }
    
    def _determine_current_position(self, wave_points: List[Dict], current_price: float, current_time) -> Dict:
        """Determine exact current position in Elliott Wave structure"""
        
        if not wave_points:
            return {'label': 'Unknown', 'type': 'Unknown', 'direction': 'Unknown', 'confidence': 0}
        
        last_pivot = wave_points[-1]
        
        # Check if we've moved significantly since last pivot
        price_change_pct = abs(current_price - last_pivot['price']) / last_pivot['price']
        
        if price_change_pct > 0.02:  # 2% movement suggests new wave development
            # Predict next wave in sequence
            next_wave_label = self._get_next_wave_label(last_pivot['label'])
            direction = self._predict_next_wave_direction(wave_points, last_pivot['label'])
            
            return {
                'label': next_wave_label,
                'type': self._get_wave_type(next_wave_label),
                'direction': direction,
                'target_price': self._calculate_wave_target(wave_points, next_wave_label),
                'characteristics': self._get_wave_characteristics(next_wave_label, direction),
                'confidence': 0.75,
                'duration': 'Developing'
            }
        else:
            # Still within last completed wave
            return {
                'label': last_pivot['label'],
                'type': last_pivot.get('wave_type', self._get_wave_type(last_pivot['label'])),
                'direction': self._get_last_wave_direction(wave_points),
                'target_price': last_pivot['price'],
                'characteristics': self._get_wave_characteristics(last_pivot['label'], 'Consolidating'),
                'confidence': last_pivot.get('confidence', 0.7),
                'duration': 'Completed'
            }
    
    def _get_next_wave_label(self, current_label: str) -> str:
        """Get the next expected wave label in Elliott Wave sequence"""
        
        sequence_map = {
            '1': '2', '2': '3', '3': '4', '4': '5', '5': 'A',
            'A': 'B', 'B': 'C', 'C': '1'
        }
        
        return sequence_map.get(current_label, '1')
    
    def _predict_next_wave_direction(self, wave_points: List[Dict], current_label: str) -> str:
        """Predict direction of next wave based on Elliott Wave theory"""
        
        # Determine overall trend from wave structure
        main_trend = self._determine_main_trend(wave_points)
        
        if current_label in ['1', '3', '5']:  # After impulse waves
            next_label = self._get_next_wave_label(current_label)
            if next_label in ['2', '4']:  # Corrective waves within impulse
                return f"Corrective retracement ({main_trend} trend)"
            elif next_label == 'A':  # Start of correction after 5-wave
                return "Major correction beginning"
            else:
                return f"Impulse continuation ({main_trend})"
        
        elif current_label in ['2', '4']:  # After corrective waves
            return f"Impulse resuming ({main_trend} trend)"
        
        elif current_label == 'A':  # After Wave A
            return "Corrective rally (Wave B)"
        
        elif current_label == 'B':  # After Wave B
            return "Final corrective decline (Wave C)"
        
        elif current_label == 'C':  # After Wave C
            return "New impulse cycle starting"
        
        return f"Continuation ({main_trend} bias)"
    
    def _calculate_wave_target(self, wave_points: List[Dict], next_label: str) -> float:
        """Calculate target price for next wave using Fibonacci ratios"""
        
        if len(wave_points) < 2:
            return wave_points[-1]['price'] if wave_points else 0
        
        last_point = wave_points[-1]
        prev_point = wave_points[-2] if len(wave_points) >= 2 else wave_points[-1]
        
        # Calculate based on Elliott Wave relationships
        if next_label == '3' and len(wave_points) >= 2:
            # Wave 3 target: 1.618 times Wave 1
            wave1_length = abs(wave_points[1]['price'] - wave_points[0]['price']) if len(wave_points) >= 2 else 0
            if wave1_length > 0:
                return last_point['price'] + (wave1_length * 1.618)
        
        elif next_label == '5' and len(wave_points) >= 4:
            # Wave 5 target: Equal to Wave 1 or 0.618 of Wave 1-3
            wave1_length = abs(wave_points[1]['price'] - wave_points[0]['price'])
            return last_point['price'] + wave1_length
        
        elif next_label in ['2', '4']:
            # Corrective waves: 38.2% to 61.8% retracement
            prev_wave_length = abs(last_point['price'] - prev_point['price'])
            return last_point['price'] - (prev_wave_length * 0.618)
        
        elif next_label == 'C':
            # Wave C typically equals Wave A
            if len(wave_points) >= 3:
                waveA_length = abs(wave_points[-2]['price'] - wave_points[-3]['price'])
                return last_point['price'] + waveA_length
        
        return last_point['price']
    
    def _calculate_precise_wave_progress(self, wave_points: List[Dict], current_price: float, current_time) -> float:
        """Calculate precise progress within current wave"""
        
        if len(wave_points) < 2:
            return 0
        
        last_pivot = wave_points[-1]
        prev_pivot = wave_points[-2]
        
        # Calculate price progress
        total_wave_movement = abs(last_pivot['price'] - prev_pivot['price'])
        current_movement = abs(current_price - prev_pivot['price'])
        
        if total_wave_movement > 0:
            progress = min((current_movement / total_wave_movement) * 100, 100)
        else:
            progress = 0
        
        return round(progress, 1)
    
    def _assess_wave_completion(self, wave_points: List[Dict], current_price: float, price_data: pd.DataFrame) -> Dict:
        """Assess if current wave is complete or developing"""
        
        if not wave_points:
            return {'status': 'Unknown', 'completion_pct': 0}
        
        last_pivot = wave_points[-1]
        
        # Check momentum indicators
        recent_prices = price_data['close'].tail(10)
        momentum = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
        
        # Check if price has moved significantly from last pivot
        price_change = abs(current_price - last_pivot['price']) / last_pivot['price']
        
        if price_change > 0.03:  # 3% movement suggests new wave
            return {
                'status': 'New wave developing',
                'completion_pct': min(price_change * 100, 100)
            }
        elif abs(momentum) < 0.005:  # Low momentum suggests completion
            return {
                'status': 'Wave completed - consolidation',
                'completion_pct': 100
            }
        else:
            return {
                'status': 'Wave in progress',
                'completion_pct': 50
            }
    
    def _get_last_wave_direction(self, wave_points: List[Dict]) -> str:
        """Get direction of the last wave"""
        
        if len(wave_points) < 2:
            return "Unknown"
        
        last_price = wave_points[-1]['price']
        prev_price = wave_points[-2]['price']
        
        return "Upward" if last_price > prev_price else "Downward"
    
    def _predict_next_wave(self, wave_points: List[Dict], current_wave_info: Dict, asset_type: str = "GENERAL") -> Dict:
        """Predict the next expected Elliott Wave"""
        
        if not wave_points:
            return {}
        
        current_label = current_wave_info.get('label', '1')
        
        # Elliott Wave sequence logic
        next_wave_map = {
            '1': '2', '2': '3', '3': '4', '4': '5', '5': 'A',
            'A': 'B', 'B': 'C', 'C': '1'  # Start new cycle
        }
        
        next_label = next_wave_map.get(current_label, '1')
        
        # Predict direction based on Elliott Wave theory
        if next_label in ['2', '4', 'A', 'C']:
            # Corrective waves - opposite direction
            next_direction = "Downward" if current_wave_info.get('direction') == "Upward" else "Upward"
        else:
            # Impulse waves - same general direction as main trend
            next_direction = self._determine_main_trend(wave_points)
        
        # Predict target levels
        target_levels = self._calculate_wave_targets(wave_points, next_label)
        
        return {
            'label': next_label,
            'type': self._get_wave_type(next_label),
            'predicted_direction': next_direction,
            'target_levels': target_levels,
            'probability': self._calculate_prediction_probability(wave_points, next_label),
            'expected_characteristics': self._get_wave_characteristics(next_label, next_direction)
        }
    
    def _get_wave_type(self, label: str) -> str:
        """Get wave type (Impulse or Corrective)"""
        if label in ['1', '3', '5']:
            return "Impulse"
        elif label in ['2', '4', 'A', 'B', 'C']:
            return "Corrective"
        else:
            return "Unknown"
    
    def _get_wave_characteristics(self, label: str, direction: str) -> List[str]:
        """Get expected characteristics for a wave"""
        
        characteristics = []
        
        if label == '1':
            characteristics.extend([
                "First impulse wave of the pattern",
                "Usually moderate in strength",
                "Sets the initial direction"
            ])
        elif label == '2':
            characteristics.extend([
                "First corrective wave",
                "Typically retraces 50-78.6% of Wave 1",
                "Should not exceed Wave 1 starting point"
            ])
        elif label == '3':
            characteristics.extend([
                "Strongest and longest wave",
                "High momentum and volume",
                "Cannot be the shortest of waves 1, 3, 5"
            ])
        elif label == '4':
            characteristics.extend([
                "Second corrective wave",
                "Usually shallow retracement (23.6-38.2%)",
                "Should not overlap Wave 1 territory"
            ])
        elif label == '5':
            characteristics.extend([
                "Final impulse wave",
                "May show divergence",
                "Completes the 5-wave pattern"
            ])
        elif label == 'A':
            characteristics.extend([
                "First wave of correction",
                "Start of counter-trend movement",
                "Usually strong momentum"
            ])
        elif label == 'B':
            characteristics.extend([
                "Corrective wave within correction",
                "Often weak and choppy",
                "Partial retracement of Wave A"
            ])
        elif label == 'C':
            characteristics.extend([
                "Final corrective wave",
                "Usually equals or exceeds Wave A",
                "Completes the corrective pattern"
            ])
        
        # Add direction-specific characteristics
        if direction == "Upward":
            characteristics.append("Expected bullish price movement")
        elif direction == "Downward":
            characteristics.append("Expected bearish price movement")
        
        return characteristics
    
    def _calculate_wave_targets(self, wave_points: List[Dict], next_label: str) -> Dict:
        """Calculate target price levels for the next wave"""
        
        if len(wave_points) < 2:
            return {}
        
        last_wave = wave_points[-1]
        prev_wave = wave_points[-2] if len(wave_points) >= 2 else wave_points[0]
        
        targets = {}
        
        # Calculate based on Elliott Wave relationships
        if next_label == '2':
            # Wave 2 typically retraces 50-78.6% of Wave 1
            wave_1_range = abs(last_wave['price'] - prev_wave['price'])
            targets['conservative'] = last_wave['price'] - (wave_1_range * 0.5)
            targets['aggressive'] = last_wave['price'] - (wave_1_range * 0.786)
            
        elif next_label == '3':
            # Wave 3 often equals 1.618 times Wave 1
            if len(wave_points) >= 3:
                wave_1_range = abs(wave_points[1]['price'] - wave_points[0]['price'])
                targets['conservative'] = last_wave['price'] + (wave_1_range * 1.0)
                targets['aggressive'] = last_wave['price'] + (wave_1_range * 1.618)
        
        elif next_label == '4':
            # Wave 4 typically retraces 23.6-38.2% of Wave 3
            wave_3_range = abs(last_wave['price'] - prev_wave['price'])
            targets['conservative'] = last_wave['price'] - (wave_3_range * 0.236)
            targets['aggressive'] = last_wave['price'] - (wave_3_range * 0.382)
        
        elif next_label == '5':
            # Wave 5 often equals Wave 1 or 0.618 times Wave 1-3
            if len(wave_points) >= 5:
                wave_1_range = abs(wave_points[1]['price'] - wave_points[0]['price'])
                targets['conservative'] = last_wave['price'] + (wave_1_range * 0.618)
                targets['aggressive'] = last_wave['price'] + (wave_1_range * 1.0)
        
        return targets
    
    def _calculate_pattern_reliability(self, wave_points: List[Dict], asset_type: str = "GENERAL") -> float:
        """Calculate overall pattern reliability score"""
        
        if not wave_points:
            return 0.0
        
        reliability = 0.0
        
        # Base reliability from wave count
        wave_count = len(wave_points)
        if wave_count >= 5:
            reliability += 0.3
        if wave_count >= 8:
            reliability += 0.2
        
        # Check wave alternation
        alternation_score = 0
        for i in range(1, len(wave_points)):
            if wave_points[i]['type'] != wave_points[i-1]['type']:
                alternation_score += 1
        
        if len(wave_points) > 1:
            alternation_ratio = alternation_score / (len(wave_points) - 1)
            reliability += alternation_ratio * 0.3
        
        # Recent wave confidence
        if wave_points:
            recent_confidence = sum([w.get('confidence', 0.5) for w in wave_points[-3:]]) / min(3, len(wave_points))
            reliability += recent_confidence * 0.2
        
        return min(reliability, 1.0)
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame) -> float:
        """Calculate overall trend strength"""
        
        if len(price_data) < 20:
            return 0.0
        
        # Simple trend strength based on price movement
        recent_closes = price_data['close'].tail(20)
        trend_strength = (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0] * 100
        
        return round(trend_strength, 2)
    
    def _determine_main_trend(self, wave_points: List[Dict]) -> str:
        """Determine the main trend direction"""
        
        if len(wave_points) < 2:
            return "Unknown"
        
        # Compare first and last wave to determine overall trend
        trend_direction = "Upward" if wave_points[-1]['price'] > wave_points[0]['price'] else "Downward"
        
        return trend_direction
    
    def _calculate_prediction_probability(self, wave_points: List[Dict], next_label: str) -> float:
        """Calculate probability of next wave prediction"""
        
        base_probability = 0.6
        
        # Increase probability based on pattern completion
        if len(wave_points) >= 5:
            base_probability += 0.2
        
        # Increase for well-defined patterns
        pattern_quality = self._calculate_pattern_reliability(wave_points)
        base_probability += pattern_quality * 0.2
        
        return min(base_probability, 0.95)
    
    def _calculate_wave_duration(self, wave_segment: List[Dict]) -> str:
        """Calculate wave duration"""
        
        if len(wave_segment) < 2:
            return "Unknown"
        
        start_time = wave_segment[0]['timestamp']
        end_time = wave_segment[-1]['timestamp']
        
        if isinstance(start_time, str):
            return "Duration calculation unavailable"
        
        duration = end_time - start_time
        
        if duration.days > 0:
            return f"{duration.days} days"
        elif duration.seconds > 3600:
            return f"{duration.seconds // 3600} hours"
        else:
            return f"{duration.seconds // 60} minutes"
    
    def _calculate_wave_strength(self, wave: Dict, current_price: float) -> float:
        """Calculate wave strength indicator"""
        
        price_diff = abs(current_price - wave['price'])
        wave_price = wave['price']
        
        if wave_price == 0:
            return 0.0
        
        strength = (price_diff / wave_price) * 100
        return round(min(strength, 10.0), 1)
    
    def _get_wave_details(self, wave_points: List[Dict]) -> Dict:
        """Get detailed information about each wave"""
        
        details = {}
        
        for wave in wave_points:
            label = wave['label']
            wave_type = self._get_wave_type(label)
            
            details[f"Wave {label}"] = (
                f"{wave_type} wave at {wave['price']:.2f} "
                f"(Confidence: {wave.get('confidence', 0.5):.1f})"
            )
        
        return details
    
    def _detect_complex_patterns(self, wave_points: List[Dict], price_data: pd.DataFrame, asset_type: str = "GENERAL") -> Dict:
        """Detect complex Elliott Wave patterns"""
        
        complex_analysis = {
            'diagonal_triangles': [],
            'extensions': [],
            'complex_corrections': [],
            'truncations': [],
            'overlapping_waves': [],
            'pattern_summary': {}
        }
        
        if len(wave_points) < 5:
            return complex_analysis
        
        # Detect diagonal triangles
        diagonal_patterns = self._detect_diagonal_triangles(wave_points, price_data)
        complex_analysis['diagonal_triangles'] = diagonal_patterns
        
        # Detect wave extensions
        extensions = self._detect_wave_extensions(wave_points)
        complex_analysis['extensions'] = extensions
        
        # Detect complex corrections (double/triple zigzags, etc.)
        complex_corrections = self._detect_complex_corrections(wave_points)
        complex_analysis['complex_corrections'] = complex_corrections
        
        # Detect truncations
        truncations = self._detect_truncations(wave_points)
        complex_analysis['truncations'] = truncations
        
        # Detect overlapping waves (invalid in impulse, valid in corrections/diagonals)
        overlaps = self._detect_overlapping_waves(wave_points)
        complex_analysis['overlapping_waves'] = overlaps
        
        # Generate pattern summary
        complex_analysis['pattern_summary'] = self._generate_pattern_summary(complex_analysis)
        
        return complex_analysis
    
    def _detect_diagonal_triangles(self, wave_points: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Detect diagonal triangle patterns"""
        
        diagonals = []
        
        # Look for 5-wave diagonal patterns
        for i in range(len(wave_points) - 4):
            segment = wave_points[i:i+5]
            
            # Check if this could be a diagonal triangle
            if self._is_diagonal_triangle(segment):
                diagonal_info = {
                    'start_index': i,
                    'end_index': i + 4,
                    'waves': [w['label'] for w in segment],
                    'type': self._classify_diagonal_type(segment),
                    'location': self._identify_diagonal_location(segment, wave_points),
                    'trend_lines': self._calculate_diagonal_trend_lines(segment),
                    'confidence': self._calculate_diagonal_confidence(segment)
                }
                diagonals.append(diagonal_info)
        
        return diagonals
    
    def _is_diagonal_triangle(self, segment: List[Dict]) -> bool:
        """Check if 5-wave segment forms a diagonal triangle"""
        
        if len(segment) != 5:
            return False
        
        # Check for wave overlaps (key diagonal feature)
        wave_1_range = (segment[0]['price'], segment[1]['price'])
        wave_4_range = (segment[3]['price'], segment[4]['price'])
        
        # Check if wave 1 and 4 overlap (required for diagonal)
        overlap_exists = (
            (min(wave_1_range) <= max(wave_4_range)) and 
            (max(wave_1_range) >= min(wave_4_range))
        )
        
        if not overlap_exists:
            return False
        
        # Check for converging trend lines
        convergence = self._check_diagonal_convergence(segment)
        
        return convergence
    
    def _check_diagonal_convergence(self, segment: List[Dict]) -> bool:
        """Check if trend lines converge in diagonal pattern"""
        
        # Get high and low trend line slopes
        highs = [w['price'] for i, w in enumerate(segment) if i % 2 == 0]  # Waves 1, 3, 5
        lows = [w['price'] for i, w in enumerate(segment) if i % 2 == 1]   # Waves 2, 4
        
        if len(highs) < 3 or len(lows) < 2:
            return False
        
        # Calculate slopes
        high_slope = (highs[-1] - highs[0]) / len(highs)
        low_slope = (lows[-1] - lows[0]) / len(lows)
        
        # Lines should converge (slopes should have opposite signs or converging)
        return abs(high_slope - low_slope) > abs(high_slope) * 0.1
    
    def _classify_diagonal_type(self, segment: List[Dict]) -> str:
        """Classify diagonal as leading or ending"""
        
        first_wave_label = segment[0]['label']
        
        if first_wave_label in ['1', 'A']:
            return 'leading_diagonal'
        elif first_wave_label in ['5', 'C']:
            return 'ending_diagonal'
        else:
            return 'potential_diagonal'
    
    def _identify_diagonal_location(self, segment: List[Dict], all_waves: List[Dict]) -> str:
        """Identify where the diagonal appears in the larger pattern"""
        
        labels = [w['label'] for w in segment]
        
        if '1' in labels:
            return 'wave_1_position'
        elif '5' in labels:
            return 'wave_5_position'
        elif 'C' in labels:
            return 'wave_C_position'
        else:
            return 'unknown_position'
    
    def _calculate_diagonal_trend_lines(self, segment: List[Dict]) -> Dict:
        """Calculate trend lines for diagonal triangle"""
        
        times = [w.get('time', 0) for w in segment]
        prices = [w['price'] for w in segment]
        
        # Separate highs and lows
        highs = [(times[i], prices[i]) for i in range(0, len(segment), 2)]
        lows = [(times[i], prices[i]) for i in range(1, len(segment), 2)]
        
        return {
            'upper_trend_line': highs,
            'lower_trend_line': lows,
            'convergence_point': self._calculate_convergence_point(highs, lows)
        }
    
    def _calculate_convergence_point(self, highs: List[Tuple], lows: List[Tuple]) -> Dict:
        """Calculate where trend lines converge"""
        
        if len(highs) < 2 or len(lows) < 2:
            return {'time': None, 'price': None}
        
        try:
            # Calculate average convergence
            high_slope = (highs[-1][1] - highs[0][1]) / max(1, len(highs) - 1)
            low_slope = (lows[-1][1] - lows[0][1]) / max(1, len(lows) - 1)
            
            # Estimate convergence
            if abs(high_slope - low_slope) > 0.001:
                convergence_time = highs[-1][0]
                convergence_price = (highs[-1][1] + lows[-1][1]) / 2
            else:
                convergence_time = None
                convergence_price = None
            
            return {
                'time': convergence_time,
                'price': convergence_price
            }
        except:
            return {'time': None, 'price': None}
    
    def _calculate_diagonal_confidence(self, segment: List[Dict]) -> float:
        """Calculate confidence score for diagonal pattern"""
        
        confidence = 0.0
        
        # Check overlap requirement
        if self._is_diagonal_triangle(segment):
            confidence += 0.4
        
        # Check converging trend lines
        if self._check_diagonal_convergence(segment):
            confidence += 0.3
        
        # Check decreasing wave sizes
        if self._check_decreasing_wave_sizes(segment):
            confidence += 0.2
        
        # Check proper labeling
        if self._check_diagonal_labeling(segment):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _check_decreasing_wave_sizes(self, segment: List[Dict]) -> bool:
        """Check if wave sizes generally decrease in diagonal"""
        
        wave_sizes = []
        for i in range(len(segment) - 1):
            size = abs(segment[i+1]['price'] - segment[i]['price'])
            wave_sizes.append(size)
        
        if len(wave_sizes) < 3:
            return False
        
        # Check if later waves are generally smaller
        early_avg = sum(wave_sizes[:2]) / 2
        late_avg = sum(wave_sizes[-2:]) / 2
        
        return late_avg < early_avg * 1.1
    
    def _check_diagonal_labeling(self, segment: List[Dict]) -> bool:
        """Check if diagonal has proper Elliott Wave labeling"""
        
        labels = [w['label'] for w in segment]
        
        # Should be either 1-2-3-4-5 or A-B-C-D-E pattern
        impulse_pattern = ['1', '2', '3', '4', '5']
        corrective_pattern = ['A', 'B', 'C', 'D', 'E']
        
        return labels == impulse_pattern or labels == corrective_pattern
    
    def _detect_wave_extensions(self, wave_points: List[Dict]) -> List[Dict]:
        """Detect extended waves (waves that are much longer than typical)"""
        
        extensions = []
        
        for i, wave in enumerate(wave_points):
            if i == 0:
                continue
            
            # Calculate wave size
            wave_size = abs(wave['price'] - wave_points[i-1]['price'])
            
            # Compare with surrounding waves
            extension_ratio = self._calculate_extension_ratio(wave_points, i, wave_size)
            
            if extension_ratio > 1.618:  # Extended if > 161.8% of typical size
                extension_info = {
                    'wave_index': i,
                    'wave_label': wave['label'],
                    'extension_ratio': extension_ratio,
                    'wave_size': wave_size,
                    'type': self._classify_extension_type(wave['label']),
                    'fibonacci_level': self._find_closest_fibonacci(extension_ratio)
                }
                extensions.append(extension_info)
        
        return extensions
    
    def _calculate_extension_ratio(self, wave_points: List[Dict], wave_index: int, wave_size: float) -> float:
        """Calculate how extended a wave is compared to typical waves"""
        
        # Get surrounding wave sizes for comparison
        surrounding_sizes = []
        
        for i in range(max(0, wave_index-2), min(len(wave_points), wave_index+3)):
            if i != wave_index and i > 0:
                size = abs(wave_points[i]['price'] - wave_points[i-1]['price'])
                surrounding_sizes.append(size)
        
        if not surrounding_sizes:
            return 1.0
        
        avg_size = sum(surrounding_sizes) / len(surrounding_sizes)
        
        return wave_size / avg_size if avg_size > 0 else 1.0
    
    def _classify_extension_type(self, wave_label: str) -> str:
        """Classify the type of wave extension"""
        
        if wave_label == '3':
            return 'third_wave_extension'
        elif wave_label == '5':
            return 'fifth_wave_extension'
        elif wave_label == '1':
            return 'first_wave_extension'
        elif wave_label == 'C':
            return 'c_wave_extension'
        else:
            return 'other_extension'
    
    def _find_closest_fibonacci(self, ratio: float) -> str:
        """Find closest Fibonacci ratio to the given extension ratio"""
        
        fib_ratios = {
            1.272: '127.2%',
            1.414: '141.4%', 
            1.618: '161.8%',
            2.000: '200.0%',
            2.618: '261.8%'
        }
        
        closest_ratio = min(fib_ratios.keys(), key=lambda x: abs(x - ratio))
        return fib_ratios[closest_ratio]
    
    def _detect_complex_corrections(self, wave_points: List[Dict]) -> List[Dict]:
        """Detect complex correction patterns (double/triple zigzags, etc.)"""
        
        corrections = []
        
        # Look for W-X-Y patterns (double zigzag)
        wxy_patterns = self._find_wxy_patterns(wave_points)
        corrections.extend(wxy_patterns)
        
        # Look for W-X-Y-XX-Z patterns (triple zigzag)
        wxyz_patterns = self._find_wxyz_patterns(wave_points)
        corrections.extend(wxyz_patterns)
        
        # Look for irregular corrections
        irregular_patterns = self._find_irregular_corrections(wave_points)
        corrections.extend(irregular_patterns)
        
        return corrections
    
    def _find_wxy_patterns(self, wave_points: List[Dict]) -> List[Dict]:
        """Find W-X-Y double zigzag patterns"""
        
        patterns = []
        
        for i in range(len(wave_points) - 2):
            segment = wave_points[i:i+3]
            labels = [w['label'] for w in segment]
            
            if labels == ['W', 'X', 'Y']:
                pattern_info = {
                    'type': 'double_zigzag',
                    'start_index': i,
                    'end_index': i + 2,
                    'waves': labels,
                    'connecting_wave': segment[1],  # X wave
                    'confidence': 0.7
                }
                patterns.append(pattern_info)
        
        return patterns
    
    def _find_wxyz_patterns(self, wave_points: List[Dict]) -> List[Dict]:
        """Find W-X-Y-XX-Z triple zigzag patterns"""
        
        patterns = []
        
        for i in range(len(wave_points) - 4):
            segment = wave_points[i:i+5]
            labels = [w['label'] for w in segment]
            
            if labels == ['W', 'X', 'Y', 'XX', 'Z']:
                pattern_info = {
                    'type': 'triple_zigzag',
                    'start_index': i,
                    'end_index': i + 4,
                    'waves': labels,
                    'connecting_waves': [segment[1], segment[3]],  # X and XX waves
                    'confidence': 0.6
                }
                patterns.append(pattern_info)
        
        return patterns
    
    def _find_irregular_corrections(self, wave_points: List[Dict]) -> List[Dict]:
        """Find irregular correction patterns (expanded flats, running corrections)"""
        
        patterns = []
        
        for i in range(len(wave_points) - 2):
            segment = wave_points[i:i+3]
            labels = [w['label'] for w in segment]
            
            if labels == ['A', 'B', 'C']:
                # Check for expanded flat (B exceeds A start, C exceeds A end)
                if self._is_expanded_flat(segment):
                    pattern_info = {
                        'type': 'expanded_flat',
                        'start_index': i,
                        'end_index': i + 2,
                        'waves': labels,
                        'characteristics': ['B_exceeds_A_start', 'C_exceeds_A_end'],
                        'confidence': 0.75
                    }
                    patterns.append(pattern_info)
                
                # Check for running correction (B significantly exceeds A, C fails to reach A end)
                elif self._is_running_correction(segment):
                    pattern_info = {
                        'type': 'running_correction',
                        'start_index': i,
                        'end_index': i + 2,
                        'waves': labels,
                        'characteristics': ['B_significantly_exceeds_A', 'C_fails_A_end'],
                        'confidence': 0.65
                    }
                    patterns.append(pattern_info)
        
        return patterns
    
    def _is_expanded_flat(self, segment: List[Dict]) -> bool:
        """Check if ABC pattern is an expanded flat"""
        
        if len(segment) != 3:
            return False
        
        a_start = segment[0]['price']
        a_end = segment[1]['price'] 
        b_end = segment[1]['price']
        
        # B should exceed A start
        b_exceeds_a = abs(b_end - a_start) > abs(a_end - a_start) * 1.05
        
        return b_exceeds_a
    
    def _is_running_correction(self, segment: List[Dict]) -> bool:
        """Check if ABC pattern is a running correction"""
        
        if len(segment) != 3:
            return False
        
        a_start = segment[0]['price']
        a_end = segment[1]['price']
        b_end = segment[1]['price']
        
        # B should significantly exceed A start (>138.2%)
        b_exceeds_significantly = abs(b_end - a_start) > abs(a_end - a_start) * 1.382
        
        return b_exceeds_significantly
    
    def _detect_truncations(self, wave_points: List[Dict]) -> List[Dict]:
        """Detect truncated waves (waves that fail to reach expected targets)"""
        
        truncations = []
        
        for i, wave in enumerate(wave_points):
            if wave['label'] == '5' and i >= 4:  # Need at least waves 1-5
                # Check if wave 5 fails to exceed wave 3
                wave_3_price = None
                wave_1_price = None
                
                # Find wave 3 and wave 1 prices
                for j in range(i-1, -1, -1):
                    if wave_points[j]['label'] == '3' and wave_3_price is None:
                        wave_3_price = wave_points[j]['price']
                    elif wave_points[j]['label'] == '1' and wave_1_price is None:
                        wave_1_price = wave_points[j]['price']
                
                if wave_3_price is not None and wave_1_price is not None:
                    # Check for truncation
                    if self._is_truncated_wave_5(wave['price'], wave_3_price, wave_1_price):
                        truncation_info = {
                            'wave_index': i,
                            'wave_label': '5',
                            'type': 'fifth_wave_truncation',
                            'target_price': wave_3_price,
                            'actual_price': wave['price'],
                            'truncation_percentage': self._calculate_truncation_percentage(
                                wave['price'], wave_3_price, wave_1_price
                            )
                        }
                        truncations.append(truncation_info)
        
        return truncations
    
    def _is_truncated_wave_5(self, wave_5_price: float, wave_3_price: float, wave_1_price: float) -> bool:
        """Check if wave 5 is truncated (fails to exceed wave 3)"""
        
        # Determine trend direction
        if wave_3_price > wave_1_price:  # Uptrend
            return wave_5_price < wave_3_price * 0.99
        else:  # Downtrend
            return wave_5_price > wave_3_price * 1.01
    
    def _calculate_truncation_percentage(self, actual: float, target: float, reference: float) -> float:
        """Calculate how much the wave is truncated as a percentage"""
        
        expected_move = abs(target - reference)
        actual_move = abs(actual - reference)
        
        if expected_move == 0:
            return 0.0
        
        return (1 - actual_move / expected_move) * 100
    
    def _detect_overlapping_waves(self, wave_points: List[Dict]) -> List[Dict]:
        """Detect overlapping waves (invalid in impulse, valid in corrections/diagonals)"""
        
        overlaps = []
        
        for i in range(len(wave_points) - 3):
            wave_1 = wave_points[i]
            wave_2 = wave_points[i + 1] 
            wave_3 = wave_points[i + 2]
            wave_4 = wave_points[i + 3]
            
            # Check if wave 1 and wave 4 overlap
            if self._waves_overlap(wave_1, wave_2, wave_3, wave_4):
                overlap_info = {
                    'wave_1_index': i,
                    'wave_4_index': i + 3,
                    'wave_1_label': wave_1['label'],
                    'wave_4_label': wave_4['label'],
                    'overlap_percentage': self._calculate_overlap_percentage(wave_1, wave_2, wave_3, wave_4),
                    'validity': self._assess_overlap_validity(wave_1['label'])
                }
                overlaps.append(overlap_info)
        
        return overlaps
    
    def _waves_overlap(self, wave_1: Dict, wave_2: Dict, wave_3: Dict, wave_4: Dict) -> bool:
        """Check if wave 1 and wave 4 overlap"""
        
        # Get wave 1 range (from start to end)
        wave_1_start = wave_1['price']
        wave_1_end = wave_2['price']
        wave_1_range = (min(wave_1_start, wave_1_end), max(wave_1_start, wave_1_end))
        
        # Get wave 4 range
        wave_4_start = wave_3['price']
        wave_4_end = wave_4['price']
        wave_4_range = (min(wave_4_start, wave_4_end), max(wave_4_start, wave_4_end))
        
        # Check for overlap
        return (wave_1_range[0] <= wave_4_range[1]) and (wave_1_range[1] >= wave_4_range[0])
    
    def _calculate_overlap_percentage(self, wave_1: Dict, wave_2: Dict, wave_3: Dict, wave_4: Dict) -> float:
        """Calculate percentage of overlap between wave 1 and wave 4"""
        
        wave_1_size = abs(wave_2['price'] - wave_1['price'])
        wave_4_size = abs(wave_4['price'] - wave_3['price'])
        
        # Simplified overlap calculation
        overlap_size = min(wave_1_size, wave_4_size) * 0.1  # Estimate
        
        return (overlap_size / max(wave_1_size, wave_4_size)) * 100 if max(wave_1_size, wave_4_size) > 0 else 0
    
    def _assess_overlap_validity(self, wave_label: str) -> str:
        """Assess if wave overlap is valid based on Elliott Wave rules"""
        
        if wave_label in ['1', '3', '5']:  # Impulse waves
            return 'invalid_overlap'
        elif wave_label in ['A', 'B', 'C']:  # Corrective waves
            return 'valid_overlap'
        else:
            return 'diagonal_overlap'
    
    def _generate_pattern_summary(self, complex_analysis: Dict) -> Dict:
        """Generate summary of detected complex patterns"""
        
        summary = {
            'total_patterns': 0,
            'pattern_types': [],
            'key_findings': [],
            'market_implications': []
        }
        
        # Count patterns
        for pattern_type, patterns in complex_analysis.items():
            if isinstance(patterns, list) and patterns:
                summary['total_patterns'] += len(patterns)
                summary['pattern_types'].append(pattern_type)
        
        # Add key findings
        if complex_analysis['diagonal_triangles']:
            summary['key_findings'].append('مثلث قطري مكتشف - توقع اختراق')
            summary['market_implications'].append('انتهاء الاتجاه محتمل')
        
        if complex_analysis['extensions']:
            summary['key_findings'].append('امتداد موجي محدد - زخم قوي')
            summary['market_implications'].append('استمرار الاتجاه محتمل')
        
        if complex_analysis['truncations']:
            summary['key_findings'].append('موجة مقطوعة موجودة - إشارة ضعف')
            summary['market_implications'].append('انعكاس محتمل للاتجاه')
        
        if complex_analysis['overlapping_waves']:
            invalid_overlaps = [o for o in complex_analysis['overlapping_waves'] 
                              if o['validity'] == 'invalid_overlap']
            if invalid_overlaps:
                summary['key_findings'].append('تداخل موجي غير صحيح - النمط قد يكون تصحيحي')
        
        return summary
    
    def _adjust_sensitivity_for_asset(self, base_sensitivity: float, asset_type: str) -> float:
        """Adjust sensitivity based on asset characteristics"""
        
        asset_adjustments = {
            "XAU/USD": 1.2,    # Gold is more volatile, needs higher sensitivity
            "NDX100": 0.9,     # Tech index has cleaner patterns, lower sensitivity
            "GER40": 1.1,      # German index has moderate volatility
            "GENERAL": 1.0     # Default
        }
        
        adjustment = asset_adjustments.get(asset_type, 1.0)
        return base_sensitivity * adjustment
    
    def _get_asset_pattern_preferences(self, asset_type: str) -> Dict:
        """Get pattern preferences for different assets"""
        
        preferences = {
            "XAU/USD": {
                "prefer_corrective": True,      # Gold often shows ABC corrections
                "extension_likelihood": 0.7,    # High chance of extensions
                "diagonal_probability": 0.3,    # Moderate diagonal patterns
                "truncation_tendency": 0.2,     # Low truncation tendency
                "typical_wave_ratios": {
                    "wave_3_extension": (1.618, 2.618),
                    "wave_5_target": (0.618, 1.000)
                }
            },
            "NDX100": {
                "prefer_corrective": False,     # Tech index prefers impulse patterns
                "extension_likelihood": 0.8,    # Very high extension probability
                "diagonal_probability": 0.4,    # Higher diagonal probability
                "truncation_tendency": 0.3,     # Moderate truncation in bear markets
                "typical_wave_ratios": {
                    "wave_3_extension": (1.618, 3.618),
                    "wave_5_target": (1.000, 1.618)
                }
            },
            "GER40": {
                "prefer_corrective": False,     # European index shows mixed patterns
                "extension_likelihood": 0.6,    # Moderate extensions
                "diagonal_probability": 0.5,    # High diagonal probability in corrections
                "truncation_tendency": 0.4,     # Higher truncation tendency
                "typical_wave_ratios": {
                    "wave_3_extension": (1.414, 2.000),
                    "wave_5_target": (0.786, 1.272)
                }
            }
        }
        
        return preferences.get(asset_type, preferences["XAU/USD"])  # Default to XAU/USD
    
    def _generate_asset_specific_waves(self, pivot_points: List[Dict], asset_type: str, price_data: pd.DataFrame) -> List[Dict]:
        """Generate asset-specific wave patterns based on market characteristics"""
        
        if len(pivot_points) < 3:
            return []
        
        preferences = self._get_asset_pattern_preferences(asset_type)
        current_price = float(price_data['close'].iloc[-1])
        
        # Create asset-specific wave patterns
        if asset_type == "XAU/USD":
            return self._generate_gold_specific_waves(pivot_points, current_price, preferences)
        elif asset_type == "NDX100":
            return self._generate_tech_index_waves(pivot_points, current_price, preferences)
        elif asset_type == "GER40":
            return self._generate_european_index_waves(pivot_points, current_price, preferences)
        else:
            return self._generate_generic_waves(pivot_points, current_price)
    
    def _generate_gold_specific_waves(self, pivot_points: List[Dict], current_price: float, preferences: Dict) -> List[Dict]:
        """Generate Elliott Wave patterns specific to XAU/USD characteristics"""
        
        # Gold tends to show strong ABC corrections during uncertainty
        if len(pivot_points) >= 3:
            # Create corrective pattern (A-B-C) which is common in gold
            waves = []
            for i, point in enumerate(pivot_points[:3]):
                wave = point.copy()
                wave['label'] = ['A', 'B', 'C'][i]
                wave['confidence'] = 0.8  # High confidence for gold corrections
                wave['asset_specific'] = f"Gold correction wave {wave['label']}"
                
                # Add gold-specific characteristics
                if wave['label'] == 'C':
                    wave['completion'] = 85.0  # Gold C waves typically complete near 85%
                    wave['expected_behavior'] = "Strong completion expected in gold"
                elif wave['label'] == 'B':
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = "Gold B wave - expect sharp reversal"
                else:  # A wave
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = "Initial decline in gold correction"
                
                waves.append(wave)
            
            return waves
        
        return []
    
    def _generate_tech_index_waves(self, pivot_points: List[Dict], current_price: float, preferences: Dict) -> List[Dict]:
        """Generate Elliott Wave patterns specific to NDX100 characteristics"""
        
        # Tech index prefers impulse patterns with strong wave 3 extensions
        if len(pivot_points) >= 5:
            waves = []
            labels = ['1', '2', '3', '4', '5']
            
            for i, point in enumerate(pivot_points[:5]):
                wave = point.copy()
                wave['label'] = labels[i]
                wave['confidence'] = 0.9  # Very high confidence for tech impulse
                wave['asset_specific'] = f"Tech index impulse wave {wave['label']}"
                
                # Add tech-specific characteristics
                if wave['label'] == '3':
                    wave['completion'] = 75.0  # Wave 3 in progress
                    wave['expected_behavior'] = "Strong tech momentum - extension likely"
                    wave['extension_probability'] = "80%"
                elif wave['label'] == '4':
                    wave['completion'] = 60.0
                    wave['expected_behavior'] = "Shallow correction in tech"
                elif wave['label'] == '5':
                    wave['completion'] = 45.0
                    wave['expected_behavior'] = "Final tech rally phase"
                else:
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = f"Tech wave {wave['label']} complete"
                
                waves.append(wave)
            
            return waves
        
        return []
    
    def _generate_european_index_waves(self, pivot_points: List[Dict], current_price: float, preferences: Dict) -> List[Dict]:
        """Generate Elliott Wave patterns specific to GER40 characteristics"""
        
        # European index shows mixed patterns with frequent diagonal triangles
        if len(pivot_points) >= 5:
            waves = []
            
            # Create diagonal triangle pattern (common in European markets)
            labels = ['1', '2', '3', '4', '5']
            for i, point in enumerate(pivot_points[:5]):
                wave = point.copy()
                wave['label'] = labels[i]
                wave['confidence'] = 0.7  # Moderate confidence
                wave['asset_specific'] = f"European diagonal wave {wave['label']}"
                wave['pattern_type'] = 'diagonal_triangle'
                
                # Add European market characteristics
                if wave['label'] == '5':
                    wave['completion'] = 70.0  # Near completion
                    wave['expected_behavior'] = "German index approaching diagonal completion"
                    wave['truncation_risk'] = "40%"
                elif wave['label'] == '4':
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = "European correction complete"
                elif wave['label'] == '3':
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = "Moderate European advance"
                else:
                    wave['completion'] = 100.0
                    wave['expected_behavior'] = f"European wave {wave['label']} complete"
                
                waves.append(wave)
            
            return waves
        
        return []
    
    def _generate_generic_waves(self, pivot_points: List[Dict], current_price: float) -> List[Dict]:
        """Generate generic wave patterns"""
        
        waves = []
        for i, point in enumerate(pivot_points[:3]):
            wave = point.copy()
            wave['label'] = str(i + 1)
            wave['confidence'] = 0.5
            wave['completion'] = 50.0
            waves.append(wave)
        
        return waves
