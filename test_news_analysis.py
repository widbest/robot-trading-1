#!/usr/bin/env python3
"""
Test news analysis functionality
"""

from news_analyzer import NewsAnalyzer

def test_news_analysis():
    """Test news analysis for different assets"""
    
    analyzer = NewsAnalyzer()
    
    print("Testing News Analysis System")
    print("=" * 50)
    
    for asset in ["XAU/USD", "NDX100", "GER40"]:
        print(f"\nAnalyzing news for {asset}:")
        
        # Get news analysis
        analysis = analyzer.analyze_market_news(asset, hours_back=24)
        
        print(f"  News Count: {analysis['news_count']}")
        
        sentiment = analysis['sentiment_analysis']
        print(f"  Overall Sentiment: {sentiment['overall_sentiment']:.3f}")
        print(f"  Trend: {sentiment['trend']}")
        print(f"  Confidence: {sentiment['confidence']:.2f}")
        
        market_impact = analysis['market_impact']
        print(f"  Market Impact: {market_impact['overall_impact']}")
        print(f"  Volatility Expectation: {market_impact['volatility_expectation']}")
        
        news_signals = analysis['news_signals']
        print(f"  News Signal: {news_signals['signal']}")
        print(f"  Signal Strength: {news_signals['strength']}/10")
        print(f"  Signal Confidence: {news_signals['confidence']:.2f}")
        
        # Recent headlines
        headlines = analysis['recent_headlines']
        if headlines:
            print(f"  Recent Headlines ({len(headlines)}):")
            for i, headline in enumerate(headlines[:3], 1):
                print(f"    {i}. {headline}")
        
        # Key themes
        themes = analysis['key_themes']
        if themes:
            print(f"  Key Themes: {', '.join(themes[:3])}")

if __name__ == "__main__":
    test_news_analysis()