"""
Sentiment Analysis Trading Strategy.

Analyses news sentiment to make trading decisions.
Positive sentiment signals LONG, Negative sentiment signals SELL, Neutral signals HOLD.

Uses SentimentAnalyser.analyse() to aggregate headlines.
"""
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy
from .sentimentAnalyser import SentimentAnalyser
logger = logging.getLogger(__name__)

class SentimentStrategy(TradingStrategy):
    """
    Analyses news sentiment to make trading decisions.
    Uses sentimentAnalyser to aggregate sentiment metrics from raw headlines.
    """

    def __init__(self):
        super().__init__(name='Sentiment', version='1.0')
        self.sentimentAnalyser = SentimentAnalyser()

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=1):
        """
        Analyse news sentiment for the ticker over analysisPeriod days.
        
        simDate is (YYYY-MM-DD) format string.
        exchange is the StockExchange instance from the agent for data access.
        analysisPeriod is the number of days prior to aggregate sentiment from the current simDate.
        
        Returns a dict with action ('long'|'sell'|'hold'), confidence, and reason.
        """
        try:
            logger.info(f'[SentimentStrategy] Analysing {ticker} on {simDate}')
            allHeadlines = []
            headlineCount = 0
            for dayOffset in range(analysisPeriod):
                currentDate = (datetime.strptime(simDate, '%Y-%m-%d') - timedelta(days=dayOffset)).strftime('%Y-%m-%d')
                headlines = exchange.getNewsForStock(ticker, mic, currentDate)
                if headlines:
                    headlines = self.sentimentAnalyser.enrichHeadlines(headlines)
                    inferredUpdates = []
                    for headline in headlines:
                        if headline.get('sentimentInferred') and headline.get('id') is not None:
                            inferredUpdates.append({
                                'id': headline.get('id'),
                                'sentiment': headline.get('sentiment', 'neutral'),
                                'score': headline.get('score', 0.0),
                                'confidence': headline.get('confidence', 0.0),
                            })
                    if inferredUpdates:
                        updatedCount = exchange.cacheHeadlineSentiments(inferredUpdates)
                        logger.info(f'[SentimentStrategy] Cached {updatedCount} inferred headlines for {ticker} on {currentDate}')
                    logger.info(f'[SentimentStrategy] {ticker} on {currentDate}: Found {len(headlines)} headlines')
                    allHeadlines.extend(headlines)
                    headlineCount += len(headlines)
                    for i, h in enumerate(headlines[:3]):
                        sampleSentiment = h.get('sentiment', '?')
                        sampleScore = h.get('score', 0.0)
                        logger.debug(f"  Sample {i + 1}: [{sampleSentiment:8}] score={sampleScore:.3f} | {h.get('headline', '')[:60]}")
                else:
                    logger.debug(f'[SentimentStrategy] {ticker} on {currentDate}: No headlines found')
            if not allHeadlines:
                logger.info(f'[SentimentStrategy] {ticker} on {simDate}: No headlines found - recommending HOLD')
                return {'action': 'hold', 'confidence': 0.3, 'reason': f'No news data available over {analysisPeriod} days', 'targetQuantity': 0}
            sentimentMetrics = self.sentimentAnalyser.analyse(allHeadlines)
            averageScore = sentimentMetrics.get('avgSentiment', 0.0)
            logger.info(f'[SentimentStrategy] {ticker} on {simDate}: {headlineCount} headlines, avg sentiment={averageScore:.4f}')
            if averageScore > 0.01:
                actionInfo = f'LONG (score={averageScore:.4f} > 0.01)'
                logger.info(f'[SentimentStrategy] {ticker}: Recommending {actionInfo}')
                return {'action': 'long', 'confidence': min(0.95, abs(averageScore) * 2), 'reason': f'Positive sentiment ({averageScore:.2f}) from {headlineCount} headlines over {analysisPeriod} days', 'targetQuantity': 1}
            elif averageScore < -0.01:
                actionInfo = f'SELL (score={averageScore:.4f} < -0.01)'
                logger.info(f'[SentimentStrategy] {ticker}: Recommending {actionInfo}')
                return {'action': 'sell', 'confidence': min(0.95, abs(averageScore) * 2), 'reason': f'Negative sentiment ({averageScore:.2f}) - exit long positions', 'targetQuantity': 0}
            else:
                actionInfo = f'HOLD (score={averageScore:.4f} near 0)'
                logger.info(f'[SentimentStrategy] {ticker}: Recommending {actionInfo}')
                return {'action': 'hold', 'confidence': 0.5, 'reason': f'Neutral sentiment ({averageScore:.2f}) from {headlineCount} headlines over {analysisPeriod} days', 'targetQuantity': 0}
        except Exception as e:
            logger.warning(f'SentimentStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}