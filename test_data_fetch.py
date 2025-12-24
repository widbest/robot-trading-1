#!/usr/bin/env python3
"""
Test script to verify real market data fetching
"""

import os
from data_provider import DataProvider

def test_current_prices():
    """Test fetching current market prices"""
    
    provider = DataProvider()
    
    instruments = ["XAU/USD", "NDX100", "GER40"]
    
    print("Testing current price fetching...")
    print("=" * 50)
    
    for instrument in instruments:
        print(f"\nTesting {instrument}:")
        
        # Test Alpha Vantage current price
        current_price = provider._get_alpha_vantage_current_price(instrument)
        if current_price:
            print(f"  Alpha Vantage current price: ${current_price:.2f}")
        else:
            print(f"  Alpha Vantage: No data")
        
        # Test overall price fetching
        overall_price = provider._get_current_price_from_web(instrument)
        if overall_price:
            print(f"  Final current price: ${overall_price:.2f}")
        else:
            print(f"  Final: No authentic price available")

def test_historical_data():
    """Test historical data fetching"""
    
    provider = DataProvider()
    
    print("\n\nTesting historical data fetching...")
    print("=" * 50)
    
    for instrument in ["XAU/USD", "NDX100", "GER40"]:
        print(f"\nTesting {instrument} historical data:")
        
        for timeframe in ["1H", "Daily"]:
            data = provider.get_price_data(instrument, timeframe, 50)
            
            if data is not None and not data.empty:
                print(f"  {timeframe}: Got {len(data)} periods")
                print(f"    Latest price: ${data['close'].iloc[-1]:.2f}")
                print(f"    Price range: ${data['low'].min():.2f} - ${data['high'].max():.2f}")
            else:
                print(f"  {timeframe}: No data available")

if __name__ == "__main__":
    test_current_prices()
    test_historical_data()