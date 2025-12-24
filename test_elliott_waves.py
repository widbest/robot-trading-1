#!/usr/bin/env python3
"""
Test Elliott Wave detection accuracy
"""

from elliott_wave_analyzer import ElliottWaveAnalyzer
from data_provider import DataProvider
import pandas as pd

def test_elliott_wave_detection():
    """Test Elliott Wave pattern detection"""
    
    provider = DataProvider()
    analyzer = ElliottWaveAnalyzer()
    
    print("Testing Elliott Wave Detection Accuracy")
    print("=" * 50)
    
    for instrument in ["XAU/USD", "NDX100", "GER40"]:
        print(f"\nAnalyzing {instrument}:")
        
        # Get 1H data for analysis
        data = provider.get_price_data(instrument, "1H", 100)
        
        if data is not None and not data.empty:
            # Perform Elliott Wave analysis
            wave_analysis = analyzer.analyze_waves(data, sensitivity=1.0)
            
            if wave_analysis:
                print(f"  Current Wave: {wave_analysis.get('current_wave', 'Unknown')}")
                print(f"  Next Wave: {wave_analysis.get('next_wave', 'Unknown')}")
                print(f"  Pattern Confidence: {wave_analysis.get('wave_confidence', 0):.2f}")
                
                # Current wave details
                current_info = wave_analysis.get('current_wave_info', {})
                print(f"  Wave Type: {current_info.get('type', 'Unknown')}")
                print(f"  Direction: {current_info.get('direction', 'Unknown')}")
                print(f"  Progress: {current_info.get('progress', 0):.1f}%")
                print(f"  Confidence: {current_info.get('confidence', 0):.0%}")
                
                # Next wave prediction
                next_info = wave_analysis.get('next_wave_info', {})
                if next_info:
                    print(f"  Next Expected: {next_info.get('label', 'Unknown')}")
                    print(f"  Next Direction: {next_info.get('predicted_direction', 'Unknown')}")
                    print(f"  Probability: {next_info.get('probability', 0):.0%}")
                
                # Wave points found
                wave_points = wave_analysis.get('wave_points', [])
                print(f"  Pivot Points Found: {len(wave_points)}")
                
                if wave_points:
                    print("  Recent Wave Sequence:", end=" ")
                    for point in wave_points[-5:]:
                        print(f"{point['label']}", end=" ")
                    print()
            else:
                print("  No wave pattern detected")
        else:
            print("  No data available")

if __name__ == "__main__":
    test_elliott_wave_detection()