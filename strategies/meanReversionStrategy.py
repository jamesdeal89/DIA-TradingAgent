"""
Mean Reversion Trading Strategy.
Testing findings of Kim et al. from literature review.

Trades based on price deviation from mean.
If price < mean - 2*std: LONG (buy the dip)
If price > mean + 2*std: SELL (sell the peak)
"""
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy
logger = logging.getLogger(__name__)

class MeanReversionStrategy(TradingStrategy):
    """
    Trades based on price deviation from mean.
    Scales lookback window based on analysisPeriod.
    """

    def __init__(self):
        super().__init__(name='MeanReversion', version='1.0')

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=30):
        """
        Calculate if stock is above/below mean over analysisPeriod.
        simDate is (YYYY-MM-DD) as str.
        exchange is a StockExchange instance for data access.
        analysisPeriod is the indow size for calculating mean and std (min 30, scales up with decision period).
        
        Returns a dict with action, confidence, reason, and targetQuantity
        """
        try:
            lookbackWindow = max(30, analysisPeriod)
            simDateTime = datetime.strptime(simDate, '%Y-%m-%d')
            startDateTime = simDateTime - timedelta(days=lookbackWindow * 6)
            startDateStr = startDateTime.strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=startDateStr, end=simDate)
            if data is None or len(data) < lookbackWindow:
                return {'action': 'hold', 'confidence': 0.2, 'reason': f'Insufficient historical data (< {lookbackWindow} days)', 'targetQuantity': 0}
            closePrices = data['Close'].tail(lookbackWindow*5)
            meanPrice = closePrices.mean()
            stdPrice = closePrices.std()
            currentPrice = closePrices.iloc[-1]
            zScore = (currentPrice - meanPrice) / stdPrice if stdPrice > 0 else 0
            threshold = 0.3
            if zScore < -threshold:
                return {'action': 'long', 'confidence': min(0.95, 0.5 + abs(zScore) * 0.15), 'reason': f'Price {currentPrice:.2f} is {abs(zScore):.2f}sd below {lookbackWindow}d mean {meanPrice:.2f}', 'targetQuantity': 1}
            elif zScore > threshold:
                return {'action': 'sell', 'confidence': min(0.95, 0.5 + abs(zScore) * 0.15), 'reason': f'Price {currentPrice:.2f} ({abs(zScore):.2f}sd above mean) - exit signal', 'targetQuantity': 0}
            else:
                return {'action': 'hold', 'confidence': 0.4, 'reason': f'Price within normal range (z={zScore:.2f}) over {lookbackWindow}d', 'targetQuantity': 0}
        except Exception as e:
            logger.warning(f'MeanReversionStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}