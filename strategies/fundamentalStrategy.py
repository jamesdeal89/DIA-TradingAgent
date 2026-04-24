"""
Fundamental Analysis Trading Strategy.

Uses price-based statistics: volatility and momentum.
"""
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy
logger = logging.getLogger(__name__)

class FundamentalStrategy(TradingStrategy):
    """
    Uses basic price-based statistics:
    Volatility and momentum over the analysis window.
    """

    def __init__(self):
        super().__init__(name='Fundamental', version='1.0')

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod):
        """
        Analyse fundamental signals using price volatility and momentum.
        
        ticker: Stock ticker symbol to examine.
        mic: Market Identifier Code
        simDate: Simulation date (YYYY-MM-DD) as str.
        exchange: StockExchange instance for data access
        analysisPeriod: Window size for calculating volatility/momentum.
        
        Returns a dict with action, confidence, reason, and targetQuantity.
        """
        try:
            lookbackWindow = max(30, analysisPeriod)
            simDateTime = datetime.strptime(simDate, '%Y-%m-%d')
            startDateTime = simDateTime - timedelta(days=lookbackWindow * 2)
            startDateStr = startDateTime.strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=startDateStr, end=simDate)
            if data is None or len(data) < lookbackWindow:
                return {'action': 'hold', 'confidence': 0.3, 'reason': f'Insufficient data for fundamental analysis (< {lookbackWindow} days)', 'targetQuantity': 0}
            returns = data['Close'].pct_change()
            volatility = returns.std()
            priceAgo = data['Close'].iloc[-lookbackWindow]
            priceToday = data['Close'].iloc[-1]
            momentum = (priceToday - priceAgo) / priceAgo
            volatility_scaling = self.calculateVolatilityScaling(volatility)
            if momentum > 0.005:
                base_confidence = min(0.8, 0.4 + momentum)
                scaled_confidence = base_confidence * volatility_scaling
                targetQty = self.confidenceToQuantity(scaled_confidence)
                return {'action': 'long', 'confidence': scaled_confidence, 'reason': f'Positive momentum ({momentum * 100:.1f}%) over {lookbackWindow}d (vol-adjusted: {volatility * 100:.1f}%)', 'targetQuantity': targetQty}
            elif momentum < -0.005:
                base_confidence = min(0.8, 0.4 + abs(momentum))
                scaled_confidence = base_confidence * volatility_scaling
                targetQty = self.confidenceToQuantity(scaled_confidence)
                return {'action': 'sell', 'confidence': scaled_confidence, 'reason': f'Weak momentum ({momentum * 100:.1f}%) - exit long (vol-adjusted: {volatility * 100:.1f}%)', 'targetQuantity': targetQty}
            else:
                return {'action': 'hold', 'confidence': 0.5, 'reason': f'Momentum neutral ({momentum * 100:.1f}%), volatility: {volatility * 100:.1f}%', 'targetQuantity': 0}
        except Exception as e:
            logger.warning(f'FundamentalStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}

    def calculateVolatilityScaling(self, volatility):
        """
        Calculate confidence scaling factor based on volatility.
        Implements literature review finding: high volatility predicts weaker momentum payoffs.
        
        volatility param is the standard deviation of returns (0.0-1.0+).
        
        Returns a scaling factor (0.2-1.2) to multiply confidence by.
        """
        if volatility < 0.05:
            return 1.2
        elif volatility < 0.15:
            return 1.0
        elif volatility < 0.3:
            return 1.0 - (volatility - 0.15) / 0.3
        else:
            return 0.3

    def confidenceToQuantity(self, confidence):
        """
        Convert confidence score to target stock quantity.
        
        Implements position sizing based on signal strength:
        Higher confidence = larger position size.
        confidence param is 0.0-1.0.
        
        Returns a target quantity of stocks (0-8).
        """
        maxQuantity = 8
        quantity = int(round(maxQuantity * confidence))
        return max(0, min(quantity, maxQuantity))