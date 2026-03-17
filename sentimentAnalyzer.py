"""
Sentiment Analysis Utilities for Agent Intelligence Layer.

This module provides sentiment analysis functions for agents to apply
to raw headline data returned by StockExchange.getNewsForStock().

Separates concerns:
- StockExchange: Data access layer (returns raw headlines)
- Agent: Intelligence layer (applies sentiment analysis logic)

Usage:
    headlines = exchange.getNewsForStock('AAPL', 'XNAS', '2015-01-15')
    metrics = analyze_sentiment(headlines)
    
    if metrics['avg_sentiment'] > 0.5:
        exchange.placeLong(...)
"""

from typing import List, Dict, Any
import numpy as np


def analyzeSentiment(headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
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
        >>> metrics = analyze_sentiment(headlines)
        >>> print(metrics['avg_sentiment'])  # 0.4
        >>> print(metrics['confidence'])      # 0.5
    """
    if not headlines:
        return {
            'total_headlines': 0,
            'positive_count': 0,
            'neutral_count': 0,
            'negative_count': 0,
            'avg_sentiment': 0.0,
            'sentiment_std': 0.0,
            'confidence': 0.0,
            'positive_ratio': 0.0,
            'negative_ratio': 0.0,
            'headlines': []
        }
    
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
        else:  # neutral
            neutral_count += 1
    
    # Calculate statistics
    avg_sentiment = np.mean(scores) if scores else 0.0
    sentiment_std = np.std(scores) if len(scores) > 1 else 0.0
    
    # Confidence: proportion of non-neutral headlines
    non_neutral = positive_count + negative_count
    confidence = non_neutral / total if total > 0 else 0.0
    
    # Ratios
    positive_ratio = positive_count / total if total > 0 else 0.0
    negative_ratio = negative_count / total if total > 0 else 0.0
    
    return {
        'total_headlines': total,
        'positive_count': positive_count,
        'neutral_count': neutral_count,
        'negative_count': negative_count,
        'avg_sentiment': round(avg_sentiment, 4),
        'sentiment_std': round(sentiment_std, 4),
        'confidence': round(confidence, 4),
        'positive_ratio': round(positive_ratio, 4),
        'negative_ratio': round(negative_ratio, 4),
        'headlines': headlines
    }


def filter_by_sentiment(headlines: List[Dict[str, Any]], sentiment: str) -> List[Dict[str, Any]]:
    """
    Filter headlines by specific sentiment.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
        sentiment: 'positive', 'neutral', or 'negative'
    
    Returns:
        Filtered list of headlines with matching sentiment
    
    Example:
        >>> headlines = [...]
        >>> positive_only = filter_by_sentiment(headlines, 'positive')
    """
    return [h for h in headlines if h['sentiment'] == sentiment]


def filter_by_confidence(headlines: List[Dict[str, Any]], min_score: float) -> List[Dict[str, Any]]:
    """
    Filter headlines by minimum confidence score.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
        min_score: Minimum sentiment score [-1.0, +1.0]
    
    Returns:
        Filtered list of headlines with |score| >= min_score
    
    Example:
        >>> headlines = [...]
        >>> high_confidence = filter_by_confidence(headlines, 0.7)
    """
    return [h for h in headlines if abs(h['score']) >= min_score]


def summarize_sentiment(headlines: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable sentiment summary.
    
    Args:
        headlines: List of headline dicts from getNewsForStock()
    
    Returns:
        String summary (e.g., "Strongly positive (8 headlines, avg: 0.75)")
    
    Example:
        >>> headlines = [...]
        >>> print(summarize_sentiment(headlines))
        Positive (5 positive, 2 neutral, 1 negative; avg: 0.45)
    """
    metrics = analyze_sentiment(headlines)
    
    if metrics['total_headlines'] == 0:
        return "No headlines"
    
    pos = metrics['positive_count']
    neu = metrics['neutral_count']
    neg = metrics['negative_count']
    avg = metrics['avg_sentiment']
    
    # Determine sentiment category
    if avg > 0.5:
        category = "Strongly positive"
    elif avg > 0.2:
        category = "Positive"
    elif avg > -0.2:
        category = "Neutral"
    elif avg > -0.5:
        category = "Negative"
    else:
        category = "Strongly negative"
    
    return f"{category} ({pos} positive, {neu} neutral, {neg} negative; avg: {avg:.2f})"


def get_bullish_threshold(sentiment_metrics: Dict[str, Any], strict: bool = False) -> bool:
    """
    Determine if sentiment is bullish (positive signal).
    
    Args:
        sentiment_metrics: Dict returned by analyze_sentiment()
        strict: If True, use stricter threshold (avg > 0.5); if False, use avg > 0.3
    
    Returns:
        Boolean indicating bullish sentiment
    
    Example:
        >>> metrics = analyze_sentiment(headlines)
        >>> if get_bullish_threshold(metrics):
        ...     exchange.placeLong(...)
    """
    threshold = 0.5 if strict else 0.3
    return sentiment_metrics['avg_sentiment'] > threshold


def get_bearish_threshold(sentiment_metrics: Dict[str, Any], strict: bool = False) -> bool:
    """
    Determine if sentiment is bearish (negative signal).
    
    Args:
        sentiment_metrics: Dict returned by analyze_sentiment()
        strict: If True, use stricter threshold (avg < -0.5); if False, use avg < -0.3
    
    Returns:
        Boolean indicating bearish sentiment
    
    Example:
        >>> metrics = analyze_sentiment(headlines)
        >>> if get_bearish_threshold(metrics):
        ...     exchange.placeShort(...)
    """
    threshold = -0.5 if strict else -0.3
    return sentiment_metrics['avg_sentiment'] < threshold


# Convenience wrapper for common agent use cases
class SentimentAnalyzer:
    """
    Convenience wrapper for sentiment analysis in agents.
    
    Usage:
        analyzer = SentimentAnalyzer()
        
        headlines = exchange.getNewsForStock('AAPL', 'XNAS', simDate)
        metrics = analyzer.analyze(headlines)
        
        if analyzer.is_bullish(metrics):
            exchange.placeLong(...)
    """
    
    def __init__(self, bullish_threshold: float = 0.3, bearish_threshold: float = -0.3):
        """
        Initialize analyzer with custom thresholds.
        
        Args:
            bullish_threshold: Score above which sentiment is bullish (default: 0.3)
            bearish_threshold: Score below which sentiment is bearish (default: -0.3)
        """
        self.bullish_threshold = bullish_threshold
        self.bearish_threshold = bearish_threshold
    
    def analyze(self, headlines: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of headlines."""
        return analyze_sentiment(headlines)
    
    def is_bullish(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics indicate bullish sentiment."""
        return metrics['avg_sentiment'] > self.bullish_threshold
    
    def is_bearish(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics indicate bearish sentiment."""
        return metrics['avg_sentiment'] < self.bearish_threshold
    
    def is_neutral(self, metrics: Dict[str, Any]) -> bool:
        """Check if metrics indicate neutral sentiment."""
        return self.bearish_threshold <= metrics['avg_sentiment'] <= self.bullish_threshold
    
    def get_signal(self, metrics: Dict[str, Any]) -> str:
        """Get signal: 'long', 'short', or 'hold'."""
        if self.is_bullish(metrics):
            return 'long'
        elif self.is_bearish(metrics):
            return 'short'
        else:
            return 'hold'
    
    def summarize(self, headlines: List[Dict[str, Any]]) -> str:
        """Get human-readable summary."""
        return summarize_sentiment(headlines)
