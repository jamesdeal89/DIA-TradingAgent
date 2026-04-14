"""
Fundamental Analysis Trading Strategy.

Uses price-based statistics: volatility and momentum.
"""
from typing import Dict, Any
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy
logger = logging.getLogger(__name__)

class FundamentalStrategy(TradingStrategy):
    """
    Uses basic price-based statistics.
    Volatility and momentum over the analysis window as proxy for fundamental strength.
    """

    def __init__(self):
        super().__init__(name='Fundamental', version='1.0')

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=1):
        """
        Analyse fundamental signals using price volatility and momentum.
        
        Args:
            ticker: Stock ticker symbol
            mic: Market Identifier Code
            simDate: Simulation date (YYYY-MM-DD)
            exchange: StockExchange instance for data access
            analysisPeriod: Window size for calculating volatility/momentum (default 20)
        
        Returns:
            Dict with action, confidence, reason, and targetQuantity
        """
        try:
            lookbackWindow = max(30, analysisPeriod)
            simDateTime = datetime.strptime(simDate, '%Y-%m-%d')
            startDateTime = simDateTime - timedelta(days=lookbackWindow * 3)
            startDateStr = startDateTime.strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=startDateStr, end=simDate)
            if data is None or len(data) < lookbackWindow:
                return {'action': 'hold', 'confidence': 0.3, 'reason': f'Insufficient data for fundamental analysis (< {lookbackWindow} days)', 'targetQuantity': 0}
            returns = data['Close'].pct_change()
            volatility = returns.std()
            priceAgo = data['Close'].iloc[-lookbackWindow]
            priceToday = data['Close'].iloc[-1]
            momentum = (priceToday - priceAgo) / priceAgo
            volatility_scaling = self._calculateVolatilityScaling(volatility)
            if momentum > 0.005:
                base_confidence = min(0.8, 0.4 + momentum)
                scaled_confidence = base_confidence * volatility_scaling
                target_qty = self._confidenceToQuantity(scaled_confidence)
                return {'action': 'long', 'confidence': scaled_confidence, 'reason': f'Positive momentum ({momentum * 100:.1f}%) over {lookbackWindow}d (vol-adjusted: {volatility * 100:.1f}%)', 'targetQuantity': target_qty}
            elif momentum < -0.005:
                base_confidence = min(0.8, 0.4 + abs(momentum))
                scaled_confidence = base_confidence * volatility_scaling
                target_qty = self._confidenceToQuantity(scaled_confidence)
                return {'action': 'sell', 'confidence': scaled_confidence, 'reason': f'Weak momentum ({momentum * 100:.1f}%) - exit long (vol-adjusted: {volatility * 100:.1f}%)', 'targetQuantity': target_qty}
            else:
                return {'action': 'hold', 'confidence': 0.5, 'reason': f'Momentum neutral ({momentum * 100:.1f}%), volatility: {volatility * 100:.1f}%', 'targetQuantity': 0}
        except Exception as e:
            logger.warning(f'FundamentalStrategy.analyse() failed: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'reason': f'Error: {str(e)}', 'targetQuantity': 0}

    def _calculateVolatilityScaling(self, volatility):
        """
        Calculate confidence scaling factor based on volatility.
        
        Implements research finding: high volatility predicts weaker momentum payoffs.
        Uses continuous scaling instead of binary cutoff.
        
        Args:
            volatility: Standard deviation of returns (0.0-1.0+)
        
        Returns:
            Scaling factor (0.2-1.2) to multiply confidence by:
            - 1.0-1.2x at low volatility (<5%): strong momentum signals
            - 1.0x at moderate volatility (5-15%): baseline confidence
            - 0.5-0.8x at high volatility (15-30%): weak momentum signals
            - 0.2-0.5x at extreme volatility (>30%): very weak signals
        """
        if volatility < 0.05:
            return 1.2
        elif volatility < 0.15:
            return 1.0
        elif volatility < 0.3:
            return 1.0 - (volatility - 0.15) / 0.3
        else:
            return 0.3

    def _confidenceToQuantity(self, confidence):
        """
        Convert confidence score to target stock quantity.
        
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