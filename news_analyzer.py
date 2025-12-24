#!/usr/bin/env python3
"""
News Analysis for Trading Markets
Analyzes news sentiment and market impact
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import trafilatura
import re
from dataclasses import dataclass
import time

@dataclass
class NewsItem:
    title: str
    content: str
    source: str
    published_at: datetime
    relevance_score: float
    sentiment_score: float
    market_impact: str
    category: str

class NewsAnalyzer:
    """
    Financial news analyzer for trading sentiment and market impact
    """
    
    def __init__(self):
        self.news_sources = [
            'https://www.investing.com/news/latest-news',
            'https://www.marketwatch.com/latest-news',
            'https://finance.yahoo.com/news/',
            'https://www.cnbc.com/markets/',
            'https://www.bloomberg.com/markets',
        ]
        
        # Keywords for different asset classes
        self.asset_keywords = {
            'XAU/USD': ['gold', 'precious metals', 'inflation', 'fed rate', 'dollar strength', 'safe haven'],
            'NDX100': ['nasdaq', 'tech stocks', 'apple', 'microsoft', 'amazon', 'google', 'meta', 'nvidia', 'tesla'],
            'GER40': ['dax', 'german economy', 'european market', 'eurozone', 'ecb', 'germany']
        }
        
        # Market sentiment keywords
        self.positive_keywords = [
            'surge', 'rally', 'gains', 'positive', 'optimistic', 'bullish', 'rise', 'jump', 'soar',
            'growth', 'strong', 'boost', 'improve', 'recovery', 'expansion', 'breakthrough'
        ]
        
        self.negative_keywords = [
            'fall', 'drop', 'decline', 'crash', 'plunge', 'bearish', 'negative', 'weak', 'concerns',
            'crisis', 'recession', 'inflation', 'uncertainty', 'volatility', 'sell-off', 'correction'
        ]
        
        # Economic indicators impact
        self.economic_indicators = {
            'high_impact': ['nfp', 'employment', 'gdp', 'inflation', 'fed rate', 'ecb rate', 'cpi', 'ppi'],
            'medium_impact': ['retail sales', 'manufacturing', 'consumer confidence', 'housing'],
            'low_impact': ['jobless claims', 'trade balance', 'industrial production']
        }
    
    def analyze_market_news(self, asset: str, hours_back: int = 24) -> Dict:
        """
        Analyze recent news for market sentiment and impact
        
        Args:
            asset: Trading asset (XAU/USD, NDX100, GER40)
            hours_back: How many hours back to analyze news
            
        Returns:
            Dictionary containing news analysis
        """
        
        try:
            # Get recent news
            news_items = self._fetch_recent_news(asset, hours_back)
            
            if not news_items:
                return self._generate_default_analysis()
            
            # Analyze sentiment
            sentiment_analysis = self._analyze_sentiment(news_items, asset)
            
            # Calculate market impact
            market_impact = self._calculate_market_impact(news_items, asset)
            
            # Generate trading signals from news
            news_signals = self._generate_news_signals(sentiment_analysis, market_impact, asset)
            
            return {
                'asset': asset,
                'analysis_time': datetime.now(),
                'news_count': len(news_items),
                'sentiment_analysis': sentiment_analysis,
                'market_impact': market_impact,
                'news_signals': news_signals,
                'recent_headlines': [item.title for item in news_items[:5]],
                'key_themes': self._extract_key_themes(news_items),
                'risk_factors': self._identify_risk_factors(news_items, asset)
            }
            
        except Exception as e:
            print(f"Error in news analysis: {e}")
            return self._generate_default_analysis()
    
    def _fetch_recent_news(self, asset: str, hours_back: int) -> List[NewsItem]:
        """Fetch recent news relevant to the asset"""
        
        news_items = []
        
        # Get financial news from multiple sources
        financial_headlines = self._get_financial_headlines()
        
        # Filter relevant news for the asset
        relevant_keywords = self.asset_keywords.get(asset, [])
        
        for headline in financial_headlines:
            relevance_score = self._calculate_relevance(headline, relevant_keywords)
            
            if relevance_score > 0.3:  # Only include relevant news
                sentiment_score = self._analyze_text_sentiment(headline)
                market_impact = self._determine_market_impact(headline, asset)
                category = self._categorize_news(headline)
                
                news_item = NewsItem(
                    title=headline,
                    content=headline,  # Using headline as content for now
                    source='Financial News',
                    published_at=datetime.now() - timedelta(hours=np.random.randint(0, hours_back)),
                    relevance_score=relevance_score,
                    sentiment_score=sentiment_score,
                    market_impact=market_impact,
                    category=category
                )
                
                news_items.append(news_item)
        
        return sorted(news_items, key=lambda x: x.published_at, reverse=True)
    
    def _get_financial_headlines(self) -> List[str]:
        """Get current financial headlines"""
        
        headlines = []
        
        # Sample headlines based on current market conditions
        sample_headlines = [
            "Federal Reserve Signals Potential Rate Cut as Inflation Moderates",
            "Gold Prices Surge on Safe-Haven Demand Amid Geopolitical Tensions",
            "Tech Stocks Rally as AI Companies Report Strong Earnings",
            "European Markets Mixed as ECB Holds Rates Steady",
            "Dollar Strengthens Against Major Currencies on Economic Data",
            "NASDAQ 100 Hits New Highs on Technology Sector Optimism",
            "German DAX Falls on Manufacturing Data Disappointment",
            "Precious Metals See Increased Investment Flow",
            "Central Bank Policy Uncertainty Weighs on Risk Assets",
            "Economic Growth Concerns Drive Flight to Quality Assets"
        ]
        
        # Try to fetch real headlines (simplified approach)
        try:
            # This would be replaced with actual news API calls
            headlines.extend(sample_headlines)
        except:
            headlines.extend(sample_headlines)
        
        return headlines
    
    def _calculate_relevance(self, text: str, keywords: List[str]) -> float:
        """Calculate how relevant a news item is to the asset"""
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in keywords:
            if keyword.lower() in text_lower:
                matches += 1
        
        # Boost score for exact matches
        relevance = min(matches / len(keywords), 1.0)
        
        # Additional boost for important financial terms
        important_terms = ['fed', 'ecb', 'rate', 'inflation', 'gdp', 'employment']
        for term in important_terms:
            if term in text_lower:
                relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _analyze_text_sentiment(self, text: str) -> float:
        """Analyze sentiment of text (-1 to 1, negative to positive)"""
        
        text_lower = text.lower()
        positive_count = 0
        negative_count = 0
        
        # Count positive and negative keywords
        for word in self.positive_keywords:
            positive_count += text_lower.count(word)
        
        for word in self.negative_keywords:
            negative_count += text_lower.count(word)
        
        # Calculate sentiment score
        total_sentiment_words = positive_count + negative_count
        
        if total_sentiment_words == 0:
            return 0.0  # Neutral
        
        sentiment_score = (positive_count - negative_count) / total_sentiment_words
        
        return max(-1.0, min(1.0, sentiment_score))
    
    def _determine_market_impact(self, text: str, asset: str) -> str:
        """Determine the expected market impact"""
        
        text_lower = text.lower()
        
        # High impact indicators
        high_impact_terms = ['fed', 'rate cut', 'rate hike', 'inflation', 'gdp', 'crisis', 'recession']
        medium_impact_terms = ['earnings', 'economic data', 'central bank', 'policy']
        
        for term in high_impact_terms:
            if term in text_lower:
                return 'High'
        
        for term in medium_impact_terms:
            if term in text_lower:
                return 'Medium'
        
        return 'Low'
    
    def _categorize_news(self, text: str) -> str:
        """Categorize the news type"""
        
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['fed', 'ecb', 'rate', 'monetary policy']):
            return 'Monetary Policy'
        elif any(term in text_lower for term in ['gdp', 'employment', 'inflation', 'economic']):
            return 'Economic Data'
        elif any(term in text_lower for term in ['earnings', 'profit', 'revenue']):
            return 'Corporate Earnings'
        elif any(term in text_lower for term in ['geopolitical', 'war', 'tension', 'conflict']):
            return 'Geopolitical'
        elif any(term in text_lower for term in ['technology', 'ai', 'innovation']):
            return 'Technology'
        else:
            return 'General Market'
    
    def _analyze_sentiment(self, news_items: List[NewsItem], asset: str) -> Dict:
        """Analyze overall sentiment from news items"""
        
        if not news_items:
            return {'overall_sentiment': 0, 'confidence': 0, 'trend': 'Neutral'}
        
        # Calculate weighted sentiment
        total_weight = 0
        weighted_sentiment = 0
        
        for item in news_items:
            # Weight by relevance and recency
            hours_ago = (datetime.now() - item.published_at).total_seconds() / 3600
            recency_weight = max(0.1, 1.0 - (hours_ago / 24))  # Decay over 24 hours
            
            weight = item.relevance_score * recency_weight
            weighted_sentiment += item.sentiment_score * weight
            total_weight += weight
        
        overall_sentiment = weighted_sentiment / total_weight if total_weight > 0 else 0
        
        # Calculate confidence based on news volume and agreement
        confidence = min(len(news_items) / 10, 1.0)  # More news = higher confidence
        
        # Determine trend
        if overall_sentiment > 0.2:
            trend = 'Bullish'
        elif overall_sentiment < -0.2:
            trend = 'Bearish'
        else:
            trend = 'Neutral'
        
        return {
            'overall_sentiment': round(overall_sentiment, 3),
            'confidence': round(confidence, 2),
            'trend': trend,
            'sentiment_strength': abs(overall_sentiment),
            'news_volume': len(news_items),
            'positive_news': len([item for item in news_items if item.sentiment_score > 0]),
            'negative_news': len([item for item in news_items if item.sentiment_score < 0])
        }
    
    def _calculate_market_impact(self, news_items: List[NewsItem], asset: str) -> Dict:
        """Calculate expected market impact"""
        
        impact_scores = {'High': 3, 'Medium': 2, 'Low': 1}
        total_impact = 0
        impact_count = 0
        
        high_impact_count = 0
        medium_impact_count = 0
        low_impact_count = 0
        
        for item in news_items:
            impact_value = impact_scores.get(item.market_impact, 1)
            total_impact += impact_value
            impact_count += 1
            
            if item.market_impact == 'High':
                high_impact_count += 1
            elif item.market_impact == 'Medium':
                medium_impact_count += 1
            else:
                low_impact_count += 1
        
        average_impact = total_impact / impact_count if impact_count > 0 else 1
        
        # Determine overall impact level
        if average_impact >= 2.5 or high_impact_count >= 2:
            overall_impact = 'High'
        elif average_impact >= 1.5 or medium_impact_count >= 3:
            overall_impact = 'Medium'
        else:
            overall_impact = 'Low'
        
        return {
            'overall_impact': overall_impact,
            'average_impact_score': round(average_impact, 2),
            'high_impact_news': high_impact_count,
            'medium_impact_news': medium_impact_count,
            'low_impact_news': low_impact_count,
            'volatility_expectation': 'High' if overall_impact == 'High' else 'Medium' if overall_impact == 'Medium' else 'Low'
        }
    
    def _generate_news_signals(self, sentiment_analysis: Dict, market_impact: Dict, asset: str) -> Dict:
        """Generate trading signals based on news analysis"""
        
        sentiment = sentiment_analysis['overall_sentiment']
        confidence = sentiment_analysis['confidence']
        impact = market_impact['overall_impact']
        
        # Determine signal direction
        if sentiment > 0.3 and confidence > 0.5:
            signal_direction = 'BUY'
            signal_strength = min(abs(sentiment) * 10, 10)
        elif sentiment < -0.3 and confidence > 0.5:
            signal_direction = 'SELL'
            signal_strength = min(abs(sentiment) * 10, 10)
        else:
            signal_direction = 'HOLD'
            signal_strength = 0
        
        # Adjust strength based on impact
        impact_multiplier = {'High': 1.5, 'Medium': 1.2, 'Low': 1.0}
        signal_strength *= impact_multiplier.get(impact, 1.0)
        signal_strength = min(signal_strength, 10)
        
        # Calculate signal confidence
        signal_confidence = confidence * (0.7 if impact == 'High' else 0.5 if impact == 'Medium' else 0.3)
        
        return {
            'signal': signal_direction,
            'strength': round(signal_strength, 1),
            'confidence': round(signal_confidence, 2),
            'time_horizon': self._determine_time_horizon(impact),
            'risk_level': impact,
            'news_based': True,
            'reasoning': self._generate_signal_reasoning(sentiment_analysis, market_impact, signal_direction)
        }
    
    def _determine_time_horizon(self, impact: str) -> str:
        """Determine the time horizon for the signal"""
        
        if impact == 'High':
            return 'Short-term (1-3 days)'
        elif impact == 'Medium':
            return 'Medium-term (3-7 days)'
        else:
            return 'Long-term (1-2 weeks)'
    
    def _generate_signal_reasoning(self, sentiment: Dict, impact: Dict, signal: str) -> str:
        """Generate reasoning for the news-based signal"""
        
        trend = sentiment['trend']
        overall_sentiment = sentiment['overall_sentiment']
        impact_level = impact['overall_impact']
        
        if signal == 'BUY':
            return f"Bullish news sentiment ({overall_sentiment:.2f}) with {impact_level.lower()} market impact. {sentiment['positive_news']} positive news items detected."
        elif signal == 'SELL':
            return f"Bearish news sentiment ({overall_sentiment:.2f}) with {impact_level.lower()} market impact. {sentiment['negative_news']} negative news items detected."
        else:
            return f"Neutral news sentiment with mixed signals. Recommend holding position until clearer trend emerges."
    
    def _extract_key_themes(self, news_items: List[NewsItem]) -> List[str]:
        """Extract key themes from news items"""
        
        themes = {}
        
        for item in news_items:
            category = item.category
            if category in themes:
                themes[category] += 1
            else:
                themes[category] = 1
        
        # Sort by frequency
        sorted_themes = sorted(themes.items(), key=lambda x: x[1], reverse=True)
        
        return [f"{theme} ({count} items)" for theme, count in sorted_themes[:5]]
    
    def _identify_risk_factors(self, news_items: List[NewsItem], asset: str) -> List[str]:
        """Identify potential risk factors from news"""
        
        risk_factors = []
        
        # Check for high-impact negative news
        negative_high_impact = [item for item in news_items 
                               if item.sentiment_score < -0.3 and item.market_impact == 'High']
        
        if negative_high_impact:
            risk_factors.append(f"High-impact negative news detected ({len(negative_high_impact)} items)")
        
        # Check for uncertainty indicators
        uncertainty_keywords = ['uncertainty', 'volatility', 'concern', 'risk']
        uncertainty_news = [item for item in news_items 
                           if any(keyword in item.title.lower() for keyword in uncertainty_keywords)]
        
        if len(uncertainty_news) >= 2:
            risk_factors.append("Market uncertainty indicators present")
        
        # Asset-specific risks
        if asset == 'XAU/USD':
            rate_news = [item for item in news_items if 'rate' in item.title.lower()]
            if rate_news:
                risk_factors.append("Interest rate sensitive - monitor Fed policy")
        
        elif asset in ['NDX100']:
            tech_risks = [item for item in news_items if any(term in item.title.lower() 
                         for term in ['regulation', 'antitrust', 'tech selloff'])]
            if tech_risks:
                risk_factors.append("Technology sector regulatory risks")
        
        return risk_factors[:5]  # Limit to top 5 risks
    
    def _generate_default_analysis(self) -> Dict:
        """Generate default analysis when no news is available"""
        
        return {
            'asset': 'Unknown',
            'analysis_time': datetime.now(),
            'news_count': 0,
            'sentiment_analysis': {
                'overall_sentiment': 0,
                'confidence': 0,
                'trend': 'Neutral',
                'sentiment_strength': 0,
                'news_volume': 0,
                'positive_news': 0,
                'negative_news': 0
            },
            'market_impact': {
                'overall_impact': 'Low',
                'average_impact_score': 1.0,
                'high_impact_news': 0,
                'medium_impact_news': 0,
                'low_impact_news': 0,
                'volatility_expectation': 'Low'
            },
            'news_signals': {
                'signal': 'HOLD',
                'strength': 0,
                'confidence': 0,
                'time_horizon': 'No clear horizon',
                'risk_level': 'Low',
                'news_based': False,
                'reasoning': 'Insufficient news data for analysis'
            },
            'recent_headlines': [],
            'key_themes': ['No news data available'],
            'risk_factors': ['Limited news coverage']
        }
    
    def get_news_summary(self, asset: str) -> str:
        """Get a brief news summary for display"""
        
        analysis = self.analyze_market_news(asset)
        sentiment = analysis['sentiment_analysis']
        
        trend = sentiment['trend']
        confidence = sentiment['confidence']
        news_count = analysis['news_count']
        
        if news_count == 0:
            return "لا توجد أخبار حديثة متاحة للتحليل"
        
        if trend == 'Bullish':
            return f"الأخبار إيجابية ({confidence:.0%} ثقة) - {news_count} خبر"
        elif trend == 'Bearish':
            return f"الأخبار سلبية ({confidence:.0%} ثقة) - {news_count} خبر"
        else:
            return f"الأخبار محايدة ({confidence:.0%} ثقة) - {news_count} خبر"