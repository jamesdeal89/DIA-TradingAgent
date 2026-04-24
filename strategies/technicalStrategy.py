"""
Technical Indicators Trading Strategy.

RSI > 70: overbought (SELL signal)
RSI < 30: oversold (LONG signal)
MACD crossover: trend change signal.
"""
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy
logger = logging.getLogger(__name__)

class TechnicalStrategy(TradingStrategy):
    """
    Uses technical indicators: RSI, MACD.
    Scales indicator periods with analysis period for larger windows.
    """

    def __init__(self):
        super().__init__(name='Technical', version='1.0')

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=30):
        """
        Calculate RSI and MACD indicators over the analysis period.
        
        simDate is a (YYYY-MM-DD) str.
        exchange is the StockExchange instance for data access from the agent.
        analysisPeriod gives the window size for calculating indicators (default 30).
        
        Returns a dict with action, confidence, reason, and targetQuantity.
        """
        try:
            lookbackWindow = max(30, analysisPeriod)
            rsiPeriod = max(14, lookbackWindow // 2)
            requiredData = lookbackWindow * 2
            simDateTime = datetime.strptime(simDate, '%Y-%m-%d')
            startDateTime = simDateTime - timedelta(days=requiredData)
            startDateStr = startDateTime.strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=startDateStr, end=simDate)
            if data is None or len(data) < rsiPeriod:
                return {'action': 'hold', 'confidence': 0.2, 'reason': f'Insufficient data for technical analysis (< {rsiPeriod} days)', 'targetQuantity': 0}
            rsi = self.calculateRsi(data['Close'], period=rsiPeriod)
            macd, signal, histogram = self.calculateMacd(data['Close'])
            confidence = 0.0
            reason = ''
            action = 'hold'
            if rsi is not None:
                if rsi > 75:
                    action = 'short'
                    confidence = max(confidence, (rsi - 75) / 25)
                    reason = f'RSI {rsi:.1f} overbought'
                elif rsi < 25:
                    action = 'long'
                    confidence = max(confidence, (25 - rsi) / 25)
                    reason = f'RSI {rsi:.1f} oversold'
                elif rsi > 60:
                    confidence = 0.2
                elif rsi < 40:
                    confidence = 0.2
                else:
                    confidence = 0.1
            else:
                confidence = 0.15
            if action == 'hold' and macd is not None and (signal is not None) and (histogram is not None):
                if histogram > 0:
                    macdStrength = abs(macd) / max(abs(signal), 1.0)
                    action = 'long'
                    confidence = max(confidence, 0.5 + min(0.3, macdStrength * 0.1))
                    reason = 'MACD bullish (histogram positive)' if reason == '' else reason
                elif histogram < 0:
                    macdStrength = abs(macd) / max(abs(signal), 1.0)
                    action = 'short'
                    confidence = max(confidence, 0.5 + min(0.3, macdStrength * 0.1))
                    reason = 'MACD bearish (histogram negative)' if reason == '' else reason
            if rsi is not None and rsi > 75:
                action = 'sell'
                confidence = max(confidence, (rsi - 75) / 25)
                reason = f'Selling overbought RSI position ({rsi:.1f})'
            targetQty = self.confidenceToQuantity(confidence) if action != 'sell' else 0
            targetQty = 0 if action == 'hold' else targetQty
            return {'action': action, 'confidence': confidence, 'reason': reason or 'No technical signal', 'targetQuantity': targetQty}
        except Exception as e:
            logger.warning(f'TechnicalStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}

    def calculateRsi(self, prices, period=14):
        """Calculate RSI indicator."""
        try:
            deltas = prices.diff()
            seed = deltas[:period + 1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsiValues = [100.0 - 100.0 / (1.0 + rs)] * period
            for i in range(period, len(deltas)):
                delta = deltas.iloc[i]
                if delta > 0:
                    up = (up * (period - 1) + delta) / period
                    down = down * (period - 1) / period
                else:
                    up = up * (period - 1) / period
                    down = (down * (period - 1) - delta) / period
                rs = up / down if down != 0 else 0
                rsiValues.append(100.0 - 100.0 / (1.0 + rs))
            return rsiValues[-1] if rsiValues else None
        except:
            return None

    def calculateMacd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator with histogram.
        Returns a tuple of (MACD line, Signal line, Histogram) where:
        - MACD: 12-day EMA minus 26-day EMA
        - Signal line: 9-day EMA of MACD line
        - Histogram: MACD minus Signal line (indicates momentum strength)
        """
        try:
            emaFast = prices.ewm(span=fast, adjust=False).mean()
            emaSlow = prices.ewm(span=slow, adjust=False).mean()
            macdLine = emaFast - emaSlow
            signalLine = macdLine.ewm(span=signal, adjust=False).mean()
            histogram = macdLine - signalLine
            return (macdLine.iloc[-1], signalLine.iloc[-1], histogram.iloc[-1])
        except:
            return (None, None, None)

    def confidenceToQuantity(self, confidence):
        """
        Convert confidence score to target stock quantity.
        Implements position sizing based on signal strength:
        Higher confidence = larger position size.
        confidence passed is 0.0-1.0
        
        Returns the target quantity of stocks (0-8).
        """
        maxQuantity = 8
        quantity = int(round(maxQuantity * confidence))
        return max(0, min(quantity, maxQuantity))