"""
Technical Indicators Trading Strategy.

- RSI > 70: overbought (SELL signal)
- RSI < 30: oversold (LONG signal)
- MACD crossover: trend change signal
"""
from typing import Dict, Any, Optional, Tuple
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

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=1):
        """
        Calculate RSI and MACD indicators over the analysis period.
        
        Args:
            ticker: Stock ticker symbol
            mic: Market Identifier Code
            simDate: Simulation date (YYYY-MM-DD)
            exchange: StockExchange instance for data access
            analysisPeriod: Window size for calculating indicators (default 20)
        
        Returns:
            Dict with action, confidence, reason, and targetQuantity
        """
        try:
            rsiPeriod = max(14, analysisPeriod // 2)
            requiredData = max(50, analysisPeriod * 2)
            simDateTime = datetime.strptime(simDate, '%Y-%m-%d')
            startDateTime = simDateTime - timedelta(days=requiredData * 3)
            startDateStr = startDateTime.strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=startDateStr, end=simDate)
            if data is None or len(data) < rsiPeriod:
                return {'action': 'hold', 'confidence': 0.2, 'reason': f'Insufficient data for technical analysis (< {rsiPeriod} days)', 'targetQuantity': 0}
            rsi = self._calculateRsi(data['Close'], period=rsiPeriod)
            macd, signal, histogram = self._calculateMacd(data['Close'])
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
            target_qty = self._confidenceToQuantity(confidence) if action != 'sell' else 0
            target_qty = 0 if action == 'hold' else target_qty
            return {'action': action, 'confidence': confidence, 'reason': reason or 'No technical signal', 'targetQuantity': target_qty}
        except Exception as e:
            logger.warning(f'TechnicalStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}

    def _calculateRsi(self, prices, period=14):
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

    def _calculateMacd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD indicator with histogram.
        
        Returns:
            Tuple of (MACD line, Signal line, Histogram) where:
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

    def _confidenceToQuantity(self, confidence):
        """Convert confidence score to target stock quantity.
        
        Implements position sizing based on signal strength:
        Higher confidence = larger position size.
        
        Args:
            confidence: Confidence value (0.0-1.0)
        
        Returns:
            Target quantity of stocks (0-8):
            - 0.0 confidence → 0 shares
            - 0.25 confidence → 2 shares
            - 0.5 confidence → 4 shares
            - 0.75 confidence → 6 shares
            - 1.0 confidence → 8 shares (max)
        """
        max_quantity = 8
        quantity = int(round(max_quantity * confidence))
        return max(0, min(quantity, max_quantity))