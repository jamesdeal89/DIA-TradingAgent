"""
Fundamental Analysis Trading Strategy.

Uses price-based statistics: volatility and momentum.
"""

from typing import Dict, Any
import logging
from .tradingStrategy import TradingStrategy

logger = logging.getLogger(__name__)


class FundamentalStrategy(TradingStrategy):
    """
    Uses basic price-based statistics.
    Volatility and momentum over the analysis window as proxy for fundamental strength.
    """
    
    def __init__(self):
        super().__init__(name="Fundamental", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
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
            # Scale lookback window based on analysis period
            lookbackWindow = max(30, analysisPeriod)
            
            # Get historical data
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < lookbackWindow:
                return {
                    'action': 'hold',
                    'confidence': 0.3,
                    'reason': f'Insufficient data for fundamental analysis (< {lookbackWindow} days)',
                    'targetQuantity': 0
                }
            
            # Calculate volatility (proxy for risk/strength) over the window
            returns = data['Close'].pct_change()
            volatility = returns.std()
            
            # Calculate price momentum over the analysis period
            priceAgo = data['Close'].iloc[-lookbackWindow]
            priceToday = data['Close'].iloc[-1]
            momentum = (priceToday - priceAgo) / priceAgo
            
            # Decision logic: threshold 0.005 (very sensitive) and volatility tolerance for more trades
            if momentum > 0.005 and volatility < 0.15:
                return {
                    'action': 'long',
                    'confidence': min(0.8, 0.4 + momentum),
                    'reason': f'Positive momentum ({momentum*100:.1f}%) over {lookbackWindow}d',
                    'targetQuantity': 1
                }
            elif momentum < -0.005 and volatility < 0.15:
                # Weak momentum = recommend selling longs (agent checks portfolio)
                return {
                    'action': 'sell',
                    'confidence': min(0.8, 0.4 + abs(momentum)),
                    'reason': f'Weak momentum ({momentum*100:.1f}%) - exit long positions',
                    'targetQuantity': 0
                }
            else:
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'reason': f'Momentum neutral ({momentum*100:.1f}%), high volatility ({volatility*100:.1f}%)',
                    'targetQuantity': 0
                }
        
        except Exception as e:
            logger.warning(f"FundamentalStrategy.analyse() failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'targetQuantity': 0
            }
