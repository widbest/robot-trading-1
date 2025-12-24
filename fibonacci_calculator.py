import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

class FibonacciCalculator:
    """
    Fibonacci retracement and extension calculations for Elliott Wave analysis
    """
    
    def __init__(self):
        # Standard Fibonacci ratios
        self.retracement_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        self.extension_levels = [1.272, 1.414, 1.618, 2.0, 2.618]
        
        # Advanced Fibonacci levels for complex analysis
        self.advanced_levels = [0.146, 0.236, 0.382, 0.5, 0.618, 0.707, 0.764, 0.786, 0.854, 0.886, 1.0]
        
        # Time-based Fibonacci ratios for temporal analysis
        self.time_ratios = [1.618, 2.618, 4.236, 6.854, 11.09, 17.944]
        
        # Complex correction Fibonacci levels
        self.complex_correction_levels = [0.382, 0.5, 0.618, 0.764, 0.786, 0.886]
        
        # Color mapping for different Fibonacci levels
        self.level_colors = {
            14.6: '#FF9999',  # Light pink
            23.6: '#FF6B6B',  # Light red
            38.2: '#4ECDC4',  # Teal
            50.0: '#45B7D1',  # Blue  
            61.8: '#96CEB4',  # Green
            70.7: '#90EE90',  # Light green
            76.4: '#FFCC99',  # Peach - Advanced level
            78.6: '#FFEAA7',  # Yellow
            85.4: '#FFE4B5',  # Moccasin
            88.6: '#FFA500',  # Orange - Advanced level
            100.0: '#DDA0DD', # Plum
            127.2: '#FFB347', # Orange
            141.4: '#87CEEB', # Sky blue
            161.8: '#98FB98', # Pale green
            200.0: '#F0E68C', # Khaki
            261.8: '#DEB887', # Burlywood
            361.8: '#D2B48C', # Tan
            423.6: '#CD853F'  # Peru
        }
    
    def calculate_levels(self, price_data: pd.DataFrame, wave_analysis: Dict, 
                        levels: List[float] = None) -> Dict:
        """
        Calculate Fibonacci retracement and extension levels
        
        Args:
            price_data: OHLC price data
            wave_analysis: Elliott wave analysis results
            levels: Custom Fibonacci levels to calculate
        
        Returns:
            Dictionary containing Fibonacci level data
        """
        
        if levels is None:
            levels = [23.6, 38.2, 50.0, 61.8, 78.6]
        
        try:
            current_price = price_data['close'].iloc[-1]
            wave_points = wave_analysis.get('wave_points', [])
            
            if len(wave_points) < 2:
                return self._generate_default_levels(current_price, levels)
            
            # Calculate retracement levels based on recent wave
            retracement_levels = self._calculate_retracement_levels(
                wave_points, levels, current_price
            )
            
            # Calculate extension levels for projections
            extension_levels = self._calculate_extension_levels(
                wave_points, current_price
            )
            
            # Combine and sort all levels
            all_levels = retracement_levels + extension_levels
            all_levels.sort(key=lambda x: x['price'])
            
            # Add support/resistance strength
            levels_with_strength = self._add_support_resistance_strength(
                all_levels, price_data
            )
            
            # Calculate time-based Fibonacci analysis
            time_analysis = self._calculate_time_fibonacci(wave_points, price_data)
            
            return {
                'levels': levels_with_strength,
                'retracement_levels': retracement_levels,
                'extension_levels': extension_levels,
                'current_price': current_price,
                'calculation_base': self._get_calculation_base(wave_points),
                'key_levels': self._identify_key_levels(levels_with_strength, current_price),
                'time_analysis': time_analysis
            }
            
        except Exception as e:
            print(f"Error calculating Fibonacci levels: {e}")
            return self._generate_default_levels(current_price if 'current_price' in locals() else 0, levels)
    
    def _calculate_retracement_levels(self, wave_points: List[Dict], 
                                    levels: List[float], current_price: float) -> List[Dict]:
        """Calculate Fibonacci retracement levels"""
        
        # Find the most recent significant wave for retracement calculation
        if len(wave_points) >= 2:
            # Use last two significant points
            high_point = wave_points[-1] if wave_points[-1]['price'] > wave_points[-2]['price'] else wave_points[-2]
            low_point = wave_points[-2] if wave_points[-1]['price'] > wave_points[-2]['price'] else wave_points[-1]
        else:
            return []
        
        swing_high = high_point['price']
        swing_low = low_point['price']
        swing_range = swing_high - swing_low
        
        retracement_levels = []
        
        for level_pct in levels:
            level_decimal = level_pct / 100.0
            
            # Calculate retracement price
            retracement_price = swing_high - (swing_range * level_decimal)
            
            # Determine distance from current price
            distance_from_current = retracement_price - current_price
            
            retracement_levels.append({
                'percentage': level_pct,
                'price': round(retracement_price, 2),
                'type': 'retracement',
                'color': self.level_colors.get(level_pct, '#CCCCCC'),
                'distance_from_current': round(distance_from_current, 2),
                'swing_high': swing_high,
                'swing_low': swing_low,
                'importance': self._calculate_level_importance(level_pct, distance_from_current)
            })
        
        return retracement_levels
    
    def _calculate_extension_levels(self, wave_points: List[Dict], current_price: float) -> List[Dict]:
        """Calculate Fibonacci extension levels for wave projections"""
        
        extension_levels = []
        
        if len(wave_points) < 3:
            return extension_levels
        
        # Use the last three points for extension calculation
        point_a = wave_points[-3]
        point_b = wave_points[-2] 
        point_c = wave_points[-1]
        
        # Calculate AB and BC ranges
        ab_range = abs(point_b['price'] - point_a['price'])
        bc_range = abs(point_c['price'] - point_b['price'])
        
        # Determine trend direction
        trend_up = point_c['price'] > point_a['price']
        
        # Calculate extension levels
        extension_ratios = [127.2, 161.8, 200.0, 261.8]
        
        for ratio in extension_ratios:
            ratio_decimal = ratio / 100.0
            
            if trend_up:
                extension_price = point_c['price'] + (ab_range * (ratio_decimal - 1.0))
            else:
                extension_price = point_c['price'] - (ab_range * (ratio_decimal - 1.0))
            
            distance_from_current = extension_price - current_price
            
            extension_levels.append({
                'percentage': ratio,
                'price': round(extension_price, 2),
                'type': 'extension',
                'color': self.level_colors.get(ratio, '#CCCCCC'),
                'distance_from_current': round(distance_from_current, 2),
                'base_wave_range': ab_range,
                'importance': self._calculate_level_importance(ratio, distance_from_current, is_extension=True)
            })
        
        return extension_levels
    
    def _add_support_resistance_strength(self, levels: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Add support/resistance strength to Fibonacci levels"""
        
        levels_with_strength = []
        
        for level in levels:
            level_price = level['price']
            
            # Calculate how many times price has tested this level
            test_count = self._count_level_tests(level_price, price_data)
            
            # Calculate bounce strength
            bounce_strength = self._calculate_bounce_strength(level_price, price_data)
            
            # Add strength indicators
            level_copy = level.copy()
            level_copy.update({
                'test_count': test_count,
                'bounce_strength': bounce_strength,
                'support_resistance_strength': min(test_count * bounce_strength, 10.0),
                'level_type': self._determine_level_type(level_price, price_data)
            })
            
            levels_with_strength.append(level_copy)
        
        return levels_with_strength
    
    def _count_level_tests(self, level_price: float, price_data: pd.DataFrame, 
                          tolerance_pct: float = 0.5) -> int:
        """Count how many times price has tested a Fibonacci level"""
        
        tolerance = level_price * (tolerance_pct / 100.0)
        test_count = 0
        
        # Check both highs and lows
        for _, row in price_data.iterrows():
            if (abs(row['high'] - level_price) <= tolerance or 
                abs(row['low'] - level_price) <= tolerance):
                test_count += 1
        
        return min(test_count, 10)  # Cap at 10 for scoring
    
    def _calculate_bounce_strength(self, level_price: float, price_data: pd.DataFrame) -> float:
        """Calculate average bounce strength from a level"""
        
        bounces = []
        tolerance = level_price * 0.01  # 1% tolerance
        
        for i in range(1, len(price_data)):
            prev_close = price_data['close'].iloc[i-1]
            curr_low = price_data['low'].iloc[i]
            curr_high = price_data['high'].iloc[i]
            curr_close = price_data['close'].iloc[i]
            
            # Check for bounce from support
            if (abs(curr_low - level_price) <= tolerance and 
                curr_close > curr_low and prev_close > level_price):
                bounce_strength = (curr_high - curr_low) / curr_low * 100
                bounces.append(bounce_strength)
            
            # Check for rejection from resistance  
            elif (abs(curr_high - level_price) <= tolerance and 
                  curr_close < curr_high and prev_close < level_price):
                bounce_strength = (curr_high - curr_low) / curr_high * 100
                bounces.append(bounce_strength)
        
        return np.mean(bounces) if bounces else 1.0
    
    def _determine_level_type(self, level_price: float, price_data: pd.DataFrame) -> str:
        """Determine if level acts as support or resistance"""
        
        current_price = price_data['close'].iloc[-1]
        
        if level_price < current_price:
            return 'support'
        elif level_price > current_price:
            return 'resistance'
        else:
            return 'current'
    
    def _calculate_level_importance(self, percentage: float, distance: float, 
                                  is_extension: bool = False) -> float:
        """Calculate importance score for a Fibonacci level"""
        
        importance = 1.0
        
        # Key Fibonacci ratios have higher importance
        key_ratios = [38.2, 50.0, 61.8, 161.8]
        if percentage in key_ratios:
            importance += 2.0
        
        # Golden ratio (61.8%) gets extra importance
        if abs(percentage - 61.8) < 0.1:
            importance += 1.0
        
        # Closer levels to current price are more immediately relevant
        distance_factor = max(0.1, 1.0 / (1.0 + abs(distance) / 100.0))
        importance *= distance_factor
        
        # Extensions vs retracements
        if is_extension:
            importance *= 0.8  # Extensions slightly less important than retracements
        
        return round(importance, 2)
    
    def _identify_key_levels(self, levels: List[Dict], current_price: float) -> Dict:
        """Identify the most important Fibonacci levels"""
        
        key_levels = {
            'nearest_support': None,
            'nearest_resistance': None,
            'strongest_support': None,
            'strongest_resistance': None
        }
        
        supports = [l for l in levels if l['price'] < current_price]
        resistances = [l for l in levels if l['price'] > current_price]
        
        if supports:
            # Nearest support
            nearest_support = max(supports, key=lambda x: x['price'])
            key_levels['nearest_support'] = nearest_support
            
            # Strongest support
            strongest_support = max(supports, key=lambda x: x.get('support_resistance_strength', 0))
            key_levels['strongest_support'] = strongest_support
        
        if resistances:
            # Nearest resistance
            nearest_resistance = min(resistances, key=lambda x: x['price'])
            key_levels['nearest_resistance'] = nearest_resistance
            
            # Strongest resistance
            strongest_resistance = max(resistances, key=lambda x: x.get('support_resistance_strength', 0))
            key_levels['strongest_resistance'] = strongest_resistance
        
        return key_levels
    
    def _get_calculation_base(self, wave_points: List[Dict]) -> Dict:
        """Get information about the base points used for calculations"""
        
        if len(wave_points) < 2:
            return {}
        
        return {
            'high_point': wave_points[-1] if wave_points[-1]['price'] > wave_points[-2]['price'] else wave_points[-2],
            'low_point': wave_points[-2] if wave_points[-1]['price'] > wave_points[-2]['price'] else wave_points[-1],
            'calculation_method': 'Recent swing high/low',
            'wave_count_used': len(wave_points)
        }
    
    def _generate_default_levels(self, current_price: float, levels: List[float]) -> Dict:
        """Generate default Fibonacci levels when wave analysis is insufficient"""
        
        if current_price == 0:
            current_price = 1.0  # Fallback value
        
        # Use 10% range as default swing
        default_range = current_price * 0.1
        swing_high = current_price + default_range
        swing_low = current_price - default_range
        
        default_levels = []
        
        for level_pct in levels:
            level_decimal = level_pct / 100.0
            level_price = swing_high - ((swing_high - swing_low) * level_decimal)
            
            default_levels.append({
                'percentage': level_pct,
                'price': round(level_price, 2),
                'type': 'retracement',
                'color': self.level_colors.get(level_pct, '#CCCCCC'),
                'distance_from_current': round(level_price - current_price, 2),
                'importance': 1.0,
                'is_default': True
            })
        
        return {
            'levels': default_levels,
            'retracement_levels': default_levels,
            'extension_levels': [],
            'current_price': current_price,
            'calculation_base': {'method': 'default_range'},
            'key_levels': {}
        }
    
    def get_fibonacci_confluences(self, levels_data: Dict, tolerance_pct: float = 1.0) -> List[Dict]:
        """Find price levels where multiple Fibonacci ratios converge"""
        
        confluences = []
        levels = levels_data.get('levels', [])
        
        if len(levels) < 2:
            return confluences
        
        # Check for price confluences
        for i, level1 in enumerate(levels):
            confluent_levels = [level1]
            
            for j, level2 in enumerate(levels[i+1:], i+1):
                price_diff_pct = abs(level1['price'] - level2['price']) / level1['price'] * 100
                
                if price_diff_pct <= tolerance_pct:
                    confluent_levels.append(level2)
            
            if len(confluent_levels) > 1:
                avg_price = np.mean([l['price'] for l in confluent_levels])
                total_importance = sum([l.get('importance', 1.0) for l in confluent_levels])
                
                confluences.append({
                    'price': round(avg_price, 2),
                    'levels': confluent_levels,
                    'confluence_strength': len(confluent_levels),
                    'total_importance': total_importance,
                    'ratios': [l['percentage'] for l in confluent_levels]
                })
        
        # Sort by confluence strength
        confluences.sort(key=lambda x: x['confluence_strength'], reverse=True)
        
        return confluences
    
    def _calculate_time_fibonacci(self, wave_points: List[Dict], price_data: pd.DataFrame) -> Dict:
        """Calculate time-based Fibonacci analysis for wave duration relationships"""
        
        time_analysis = {
            'wave_durations': [],
            'time_ratios': [],
            'time_projections': [],
            'fibonacci_time_zones': [],
            'temporal_confluence': []
        }
        
        if len(wave_points) < 3:
            return time_analysis
        
        # Calculate wave durations
        wave_durations = self._calculate_wave_durations(wave_points, price_data)
        time_analysis['wave_durations'] = wave_durations
        
        # Calculate time ratios between waves
        time_ratios = self._calculate_time_ratios(wave_durations)
        time_analysis['time_ratios'] = time_ratios
        
        # Project future time targets
        time_projections = self._project_time_targets(wave_durations, price_data)
        time_analysis['time_projections'] = time_projections
        
        # Calculate Fibonacci time zones
        fib_time_zones = self._calculate_fibonacci_time_zones(wave_points, price_data)
        time_analysis['fibonacci_time_zones'] = fib_time_zones
        
        # Find temporal confluence points
        temporal_confluence = self._find_temporal_confluence(time_projections, fib_time_zones)
        time_analysis['temporal_confluence'] = temporal_confluence
        
        return time_analysis
    
    def _calculate_wave_durations(self, wave_points: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Calculate duration of each Elliott Wave"""
        
        durations = []
        
        for i in range(len(wave_points) - 1):
            current_wave = wave_points[i]
            next_wave = wave_points[i + 1]
            
            # Get timestamps
            if 'timestamp' in current_wave and 'timestamp' in next_wave:
                start_time = current_wave['timestamp']
                end_time = next_wave['timestamp']
                
                if hasattr(start_time, 'total_seconds') and hasattr(end_time, 'total_seconds'):
                    duration = abs(end_time - start_time)
                    duration_hours = duration.total_seconds() / 3600
                else:
                    # Fallback: estimate from data index
                    duration_hours = abs(next_wave.get('index', i+1) - current_wave.get('index', i))
            else:
                # Estimate duration from price data index
                duration_hours = 1  # Default assumption
            
            duration_info = {
                'wave_label': current_wave['label'],
                'start_price': current_wave['price'],
                'end_price': next_wave['price'],
                'duration_hours': duration_hours,
                'wave_index': i
            }
            durations.append(duration_info)
        
        return durations
    
    def _calculate_time_ratios(self, wave_durations: List[Dict]) -> List[Dict]:
        """Calculate Fibonacci ratios between wave durations"""
        
        ratios = []
        
        for i in range(len(wave_durations)):
            for j in range(i + 1, len(wave_durations)):
                wave_1 = wave_durations[i]
                wave_2 = wave_durations[j]
                
                if wave_1['duration_hours'] > 0 and wave_2['duration_hours'] > 0:
                    ratio = wave_2['duration_hours'] / wave_1['duration_hours']
                    
                    # Check if ratio is close to Fibonacci levels
                    closest_fib = self._find_closest_time_fibonacci(ratio)
                    
                    if closest_fib:
                        ratio_info = {
                            'wave_1': wave_1['wave_label'],
                            'wave_2': wave_2['wave_label'],
                            'actual_ratio': ratio,
                            'fibonacci_ratio': closest_fib['ratio'],
                            'fibonacci_name': closest_fib['name'],
                            'accuracy': closest_fib['accuracy']
                        }
                        ratios.append(ratio_info)
        
        return ratios
    
    def _find_closest_time_fibonacci(self, ratio: float) -> Optional[Dict]:
        """Find closest Fibonacci ratio for time analysis"""
        
        fib_ratios = {
            0.618: 'Golden Ratio (61.8%)',
            1.000: 'Equality (100%)',
            1.618: 'Golden Ratio (161.8%)',
            2.618: 'Fibonacci Extension (261.8%)',
            4.236: 'Fibonacci Extension (423.6%)',
            6.854: 'Fibonacci Extension (685.4%)'
        }
        
        tolerance = 0.1  # 10% tolerance
        
        for fib_ratio, name in fib_ratios.items():
            if abs(ratio - fib_ratio) / fib_ratio < tolerance:
                accuracy = 1 - (abs(ratio - fib_ratio) / fib_ratio)
                return {
                    'ratio': fib_ratio,
                    'name': name,
                    'accuracy': accuracy
                }
        
        return None
    
    def _project_time_targets(self, wave_durations: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Project future time targets based on Fibonacci ratios"""
        
        projections = []
        
        if not wave_durations:
            return projections
        
        # Use the most recent completed wave as base
        base_wave = wave_durations[-1]
        base_duration = base_wave['duration_hours']
        
        # Get current time (last data point)
        current_time = price_data.index[-1]
        
        # Project using Fibonacci time ratios
        for ratio in self.time_ratios:
            projected_duration = base_duration * ratio
            
            if hasattr(current_time, 'to_pydatetime'):
                projected_time = current_time + pd.Timedelta(hours=projected_duration)
            else:
                projected_time = current_time
            
            projection = {
                'base_wave': base_wave['wave_label'],
                'fibonacci_ratio': ratio,
                'projected_duration': projected_duration,
                'projected_time': projected_time,
                'confidence': self._calculate_time_projection_confidence(ratio, wave_durations)
            }
            projections.append(projection)
        
        return projections
    
    def _calculate_fibonacci_time_zones(self, wave_points: List[Dict], price_data: pd.DataFrame) -> List[Dict]:
        """Calculate Fibonacci time zones from significant wave points"""
        
        time_zones = []
        
        if len(wave_points) < 2:
            return time_zones
        
        # Find significant wave start (usually wave 1 or A)
        start_wave = wave_points[0]
        
        # Calculate time zones from this point
        base_timestamp = start_wave.get('timestamp', price_data.index[0])
        
        if hasattr(base_timestamp, 'to_pydatetime'):
            base_time = base_timestamp
        else:
            base_time = price_data.index[0]
        
        # Calculate Fibonacci time intervals
        fib_sequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
        timeframe_hours = self._estimate_timeframe_hours(price_data)
        
        for fib_num in fib_sequence:
            time_offset = fib_num * timeframe_hours
            
            if hasattr(base_time, 'to_pydatetime'):
                zone_time = base_time + pd.Timedelta(hours=time_offset)
            else:
                zone_time = base_time
            
            zone = {
                'fibonacci_number': fib_num,
                'time_offset_hours': time_offset,
                'zone_time': zone_time,
                'importance': self._calculate_time_zone_importance(fib_num)
            }
            time_zones.append(zone)
        
        return time_zones
    
    def _estimate_timeframe_hours(self, price_data: pd.DataFrame) -> float:
        """Estimate the timeframe of the data in hours"""
        
        if len(price_data) < 2:
            return 1.0
        
        try:
            time_diff = price_data.index[1] - price_data.index[0]
            if hasattr(time_diff, 'total_seconds'):
                return time_diff.total_seconds() / 3600
            else:
                return 1.0  # Default to 1 hour
        except:
            return 1.0
    
    def _calculate_time_zone_importance(self, fib_number: int) -> str:
        """Calculate importance of Fibonacci time zone"""
        
        if fib_number in [8, 13, 21, 34]:
            return 'High'
        elif fib_number in [5, 55, 89]:
            return 'Medium'
        else:
            return 'Low'
    
    def _calculate_time_projection_confidence(self, ratio: float, wave_durations: List[Dict]) -> float:
        """Calculate confidence for time projections"""
        
        base_confidence = 0.5
        
        # Higher confidence for common Fibonacci ratios
        if ratio in [1.618, 2.618]:
            base_confidence += 0.2
        elif ratio in [4.236]:
            base_confidence += 0.1
        
        # Adjust based on historical accuracy
        if len(wave_durations) >= 3:
            base_confidence += 0.1
        
        return min(base_confidence, 0.9)
    
    def _find_temporal_confluence(self, time_projections: List[Dict], time_zones: List[Dict]) -> List[Dict]:
        """Find confluence points where multiple time analyses converge"""
        
        confluences = []
        tolerance_hours = 24  # 1 day tolerance
        
        for projection in time_projections:
            proj_time = projection['projected_time']
            
            # Find matching time zones
            matching_zones = []
            for zone in time_zones:
                zone_time = zone['zone_time']
                
                if hasattr(proj_time, 'to_pydatetime') and hasattr(zone_time, 'to_pydatetime'):
                    time_diff = abs((proj_time - zone_time).total_seconds() / 3600)
                    
                    if time_diff <= tolerance_hours:
                        matching_zones.append(zone)
            
            if matching_zones:
                confluence = {
                    'projected_time': proj_time,
                    'projection_source': projection,
                    'matching_zones': matching_zones,
                    'confluence_strength': len(matching_zones),
                    'importance': 'High' if len(matching_zones) >= 2 else 'Medium'
                }
                confluences.append(confluence)
        
        return confluences
