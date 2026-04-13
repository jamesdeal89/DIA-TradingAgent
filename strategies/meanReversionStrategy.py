"""
Mean Reversion Trading Strategy.

Trades based on price deviation from mean.
If price < mean - 2*std → LONG (buy the dip)
If price > mean + 2*std → SELL (sell the peak)
"""

from typing import Dict, Any, Optional
import logging
from .tradingStrategy import TradingStrategy

logger = logging.getLogger(__name__)


class MeanReversionStrategy(TradingStrategy):
    """
    Trades based on price deviation from mean.
    Dynamically scales lookback window based on analysisPeriod.
    """
    
    def __init__(self):
        super().__init__(name="MeanReversion", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """
        Calculate if stock is above/below mean over analysisPeriod.
        
        Args:
            ticker: Stock ticker symbol
            mic: Market Identifier Code
            simDate: Simulation date (YYYY-MM-DD)
            exchange: StockExchange instance for data access
            analysisPeriod: Window size for calculating mean and std (default 20, scales up with decision period)
        
        Returns:
            Dict with action, confidence, reason, and targetQuantity
        """
        try:
            # Scale lookback window based on analysis period
            lookbackWindow = max(20, analysisPeriod)
            requiredHistory = lookbackWindow + 5
            
            # Get historical data leading up to simDate
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < lookbackWindow:
                return {
                    'action': 'hold',
                    'confidence': 0.2,
                    'reason': f'Insufficient historical data (< {lookbackWindow} days)',
                    'targetQuantity': 0
                }
            
            # Calculate moving average and std deviation over the scaled window
            closePrices = data['Close'].tail(lookbackWindow)
            meanPrice = closePrices.mean()
            stdPrice = closePrices.std()
            currentPrice = closePrices.iloc[-1]
            
            # Calculate z-score (how many std devs away from mean)
            zScore = (currentPrice - meanPrice) / stdPrice if stdPrice > 0 else 0
            
            # Trading logic (threshold 0.3 generates signals ~24% of time - more aggressive)
            threshold = 0.3
            
            if zScore < -threshold:  # Price is threshold+ std devs below mean
                return {
                    'action': 'long',
                    'confidence': min(0.9, 0.5 + abs(zScore) * 0.15),
                    'reason': f'Price {currentPrice:.2f} is {abs(zScore):.2f}sd below {lookbackWindow}d mean {meanPrice:.2f}',
                    'targetQuantity': 1
                }
            elif zScore > threshold:  # Price is threshold+ std devs above mean
                # Recommend selling (agent will check portfolio before executing)
                return {
                    'action': 'sell',
                    'confidence': min(0.9, 0.5 + abs(zScore) * 0.15),
                    'reason': f'Price {currentPrice:.2f} ({abs(zScore):.2f}sd above mean) - exit signal',
                    'targetQuantity': 0
                }
            else:
                return {
                    'action': 'hold',
                    'confidence': 0.4,
                    'reason': f'Price within normal range (z={zScore:.2f}) over {lookbackWindow}d',
                    'targetQuantity': 0
                }
        
        except Exception as e:
            logger.warning(f"MeanReversionStrategy.analyse() failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'targetQuantity': 0
            }
