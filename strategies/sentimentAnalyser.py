"""
Sentiment Analysis Utilities for Agent Intelligence Layer.

This module provides sentiment analysis functions for agents to apply
to raw headline data returned by StockExchange.getNewsForStock().

Separates concerns:
- StockExchange: Data layer (queries headlines, handles on-demand FinBERT computation & caching)
- SentimentAnalyser: Utility layer (aggregates pre-computed sentiment scores into metrics)
- Agent: Intelligence layer (applies aggregated metrics to make trading decisions)

Usage:
    headlines = exchange.getNewsForStock('AAPL', 'XNAS', '2015-01-15')
    # Note: headlines may have been computed on-demand via FinBERT and cached to DB
    
    metrics = analyseSentiment(headlines)
    
    if metrics['avgSentiment'] > 0.5:
        exchange.placeLong(...)
"""
from typing import List, Dict, Any
import numpy as np

def analyseSentiment(headlines):
    """
    Aggregate sentiment analysis from raw headline data.
    
    Applies agent intelligence to raw headlines returned by stockExchange.getNewsForStock().
    
    Args:
        headlines: List of headline dicts from getNewsForStock():
            [
                {'headline': str, 'sentiment': str, 'score': float, 'url': str, 'date': str, 'publisher': str},
                ...
            ]
    
    Returns:
        Dict with aggregated sentiment metrics:
        {
            'total_headlines': int,         # Total number of headlines
            'positive_count': int,          # Number of positive headlines
            'neutral_count': int,           # Number of neutral headlines
            'negative_count': int,          # Number of negative headlines
            'avg_sentiment': float,         # Average sentiment score [-1.0, +1.0]
            'sentiment_std': float,         # Standard deviation of sentiment scores
            'confidence': float,            # Confidence metric (proportion of non-neutral)
            'positive_ratio': float,        # Positive headlines / total
            'negative_ratio': float,        # Negative headlines / total
            'headlines': list               # Original headline data (includes sentiment/score)
        }
    
    Example:
        >>> headlines = [
        ...     {'headline': 'Stock rises', 'sentiment': 'positive', 'score': 0.8, ...},
        ...     {'headline': 'Mixed signals', 'sentiment': 'neutral', 'score': 0.0, ...}
        ... ]
        >>> metrics = analyse_sentiment(headlines)
        >>> print(metrics['avg_sentiment'])  # 0.4
        >>> print(metrics['confidence'])      # 0.5
    """
    if not headlines:
        return {'totalHeadlines': 0, 'positiveCount': 0, 'neutralCount': 0, 'negativeCount': 0, 'avgSentiment': 0.0, 'sentimentStd': 0.0, 'confidence': 0.0, 'positiveRatio': 0.0, 'negativeRatio': 0.0, 'headlines': []}
    total = len(headlines)
    scores = []
    positive_count = 0
    neutral_count = 0
    negative_count = 0
    for headline in headlines:
        sentiment = headline['sentiment']
        score = float(headline['score'])
        scores.append(score)
        if sentiment == 'positive':
            positive_count += 1
        elif sentiment == 'negative':
            negative_count += 1
        else:
            neutral_count += 1
    avg_sentiment = np.mean(scores) if scores else 0.0
    sentiment_std = np.std(scores) if len(scores) > 1 else 0.0
    non_neutral = positive_count + negative_count
    confidence = non_neutral / total if total > 0 else 0.0
    positive_ratio = positive_count / total if total > 0 else 0.0
    negative_ratio = negative_count / total if total > 0 else 0.0
    return {'totalHeadlines': total, 'positiveCount': positive_count, 'neutralCount': neutral_count, 'negativeCount': negative_count, 'avgSentiment': round(avg_sentiment, 4), 'sentimentStd': round(sentiment_std, 4), 'confidence': round(confidence, 4), 'positiveRatio': round(positive_ratio, 4), 'negativeRatio': round(negative_ratio, 4), 'headlines': headlines}

def filterBySentiment(headlines, sentiment):
    """
    Filter headlines by specific sentiment.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
        sentiment: 'positive', 'neutral', or 'negative'
    
    Returns:
        Filtered list of headlines with matching sentiment
    
    Example:
        >>> headlines = [...]
        >>> positiveOnly = filterBySentiment(headlines, 'positive')
    """
    return [h for h in headlines if h['sentiment'] == sentiment]

def filterByConfidence(headlines, minScore):
    """
    Filter headlines by minimum confidence score.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
        minScore: Minimum sentiment score [-1.0, +1.0]
    
    Returns:
        Filtered list of headlines with |score| >= minScore
    
    Example:
        >>> headlines = [...]
        >>> highConfidence = filterByConfidence(headlines, 0.7)
    """
    return [h for h in headlines if abs(h['score']) >= minScore]

def summariseSentiment(headlines):
    """
    Generate human-readable sentiment summary.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
    
    Returns:
        String summary (e.g., "Strongly positive (8 headlines, avg: 0.75)")
    
    Example:
        >>> headlines = [...]
        >>> print(summariseSentiment(headlines))
        Positive (5 positive, 2 neutral, 1 negative; avg: 0.45)
    """
    metrics = analyseSentiment(headlines)
    if metrics['totalHeadlines'] == 0:
        return 'No headlines'
    pos = metrics['positiveCount']
    neu = metrics['neutralCount']
    neg = metrics['negativeCount']
    avg = metrics['avgSentiment']
    if avg > 0.5:
        category = 'Strongly positive'
    elif avg > 0.2:
        category = 'Positive'
    elif avg > -0.2:
        category = 'Neutral'
    elif avg > -0.5:
        category = 'Negative'
    else:
        category = 'Strongly negative'
    return f'{category} ({pos} positive, {neu} neutral, {neg} negative; avg: {avg:.2f})'

def getBullishThreshold(sentimentMetrics, strict=False):
    """
    Determine if sentiment is bullish (positive signal).
    
    Args:
        sentimentMetrics: Dict returned by analyseSentiment()
        strict: If True, use stricter threshold (avg > 0.5); if False, use avg > 0.3
    
    Returns:
        Boolean indicating bullish sentiment
    
    Example:
        >>> metrics = analyseSentiment(headlines)
        >>> if getBullishThreshold(metrics):
        ...     exchange.placeLong(...)
    """
    threshold = 0.5 if strict else 0.3
    return sentimentMetrics['avgSentiment'] > threshold

def getBearishThreshold(sentimentMetrics, strict=False):
    """
    Determine if sentiment is bearish (negative signal).
    
    Args:
        sentimentMetrics: Dict returned by analyseSentiment()
        strict: If True, use stricter threshold (avg < -0.5); if False, use avg < -0.3
    
    Returns:
        Boolean indicating bearish sentiment
    
    Example:
        >>> metrics = analyseSentiment(headlines)
        >>> if getBearishThreshold(metrics):
        ...     exchange.placeShort(...)
    """
    threshold = -0.5 if strict else -0.3
    return sentimentMetrics['avgSentiment'] < threshold

class SentimentAnalyser:
    """
    Convenience wrapper for sentiment analysis in agents.
    
    Usage:
        analyser = SentimentAnalyser()
        
        headlines = exchange.getNewsForStock('AAPL', 'XNAS', simDate)
        metrics = analyser.analyse(headlines)
        
        if analyser.is_bullish(metrics):
            exchange.placeLong(...)
    """

    def __init__(self, bullishThreshold=0.3, bearishThreshold=-0.3):
        """
        Initialise analyser with custom thresholds.
        
        Args:
            bullishThreshold: Score above which sentiment is bullish (default: 0.3)
            bearishThreshold: Score below which sentiment is bearish (default: -0.3)
        """
        self.bullishThreshold = bullishThreshold
        self.bearishThreshold = bearishThreshold

    def analyse(self, headlines):
        """Analyse sentiment of headlines."""
        return analyseSentiment(headlines)

    def isBullish(self, metrics):
        """Check if metrics indicate bullish sentiment."""
        return metrics['avgSentiment'] > self.bullishThreshold

    def isBearish(self, metrics):
        """Check if metrics indicate bearish sentiment."""
        return metrics['avgSentiment'] < self.bearishThreshold

    def isNeutral(self, metrics):
        """Check if metrics indicate neutral sentiment."""
        return self.bearishThreshold <= metrics['avgSentiment'] <= self.bullishThreshold

    def getSignal(self, metrics):
        """Get signal: 'long', 'short', or 'hold'."""
        if self.isBullish(metrics):
            return 'long'
        elif self.isBearish(metrics):
            return 'short'
        else:
            return 'hold'

    def summarise(self, headlines):
        """Get human-readable summary."""
        return summariseSentiment(headlines)