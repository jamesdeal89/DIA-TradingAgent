"""
Technical Indicators Trading Strategy.

Uses RSI, MACD, and Bollinger Bands.
- RSI > 70: overbought (SELL signal)
- RSI < 30: oversold (LONG signal)
- MACD crossover: trend change signal
"""

from typing import Dict, Any, Optional, Tuple
import logging
from .tradingStrategy import TradingStrategy

logger = logging.getLogger(__name__)


class TechnicalStrategy(TradingStrategy):
    """
    Uses technical indicators: RSI, MACD, Bollinger Bands.
    Scales indicator periods with analysis period for larger windows.
    """
    
    def __init__(self):
        super().__init__(name="Technical", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
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
            # Scale indicator periods based on analysis window
            rsiPeriod = max(14, analysisPeriod // 2)
            requiredData = max(50, analysisPeriod * 2)
            
            # Get historical data for indicator calculation
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < rsiPeriod:
                return {
                    'action': 'hold',
                    'confidence': 0.2,
                    'reason': f'Insufficient data for technical analysis (< {rsiPeriod} days)',
                    'targetQuantity': 0
                }
            
            # Calculate RSI with scaled period
            rsi = self._calculateRsi(data['Close'], period=rsiPeriod)
            
            # Calculate MACD
            macd, signal = self._calculateMacd(data['Close'])
            
            # Determine action based on indicators
            confidence = 0.0
            reason = ""
            action = 'hold'
            
            # RSI signals (strongest signal) - be more selective to reduce dominance
            if rsi is not None:
                if rsi > 75:  # Raised to 75 - only extreme overbought
                    action = 'short'
                    confidence = max(confidence, (rsi - 75) / 25)  # 0.0-1.0
                    reason = f"RSI {rsi:.1f} overbought"
                elif rsi < 25:  # Raised to 25 - only extreme oversold
                    action = 'long'
                    confidence = max(confidence, (25 - rsi) / 25)  # 0.0-1.0
                    reason = f"RSI {rsi:.1f} oversold"
                else:
                    # Neutral RSI: provide confidence based on proximity to extremes
                    if rsi > 60:
                        confidence = 0.2  # Approaching overbought
                    elif rsi < 40:
                        confidence = 0.2  # Approaching oversold
                    else:
                        confidence = 0.1  # Mid-range, low conviction
            else:
                # RSI calculation failed, use minimal confidence
                confidence = 0.15
            
            # MACD confirmation (use if RSI is hold)
            if action == 'hold' and macd is not None and signal is not None:
                if macd > signal:
                    action = 'long'
                    confidence = max(confidence, 0.6)
                    reason = "MACD bullish crossover" if reason == "" else reason
                elif macd < signal:
                    action = 'short'
                    confidence = max(confidence, 0.6)
                    reason = "MACD bearish crossover" if reason == "" else reason
            
            # If overbought, recommend selling (agent will check portfolio before executing)
            if rsi is not None and rsi > 75:
                action = 'sell'
                confidence = max(confidence, (rsi - 75) / 25)
                reason = f"Selling overbought RSI position ({rsi:.1f})"
            
            return {
                'action': action,
                'confidence': confidence,
                'reason': reason or 'No technical signal',
                'targetQuantity': 0 if action == 'sell' else (1 if action != 'hold' else 0)
            }
        
        except Exception as e:
            logger.warning(f"TechnicalStrategy.analyse() failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'targetQuantity': 0
            }
    
    def _calculateRsi(self, prices, period: int = 14) -> Optional[float]:
        """Calculate RSI indicator."""
        try:
            deltas = prices.diff()
            seed = deltas[:period+1]
            up = seed[seed >= 0].sum() / period
            down = -seed[seed < 0].sum() / period
            rs = up / down if down != 0 else 0
            rsiValues = [100.0 - 100.0 / (1.0 + rs)] * period
            
            for i in range(period, len(deltas)):
                delta = deltas.iloc[i]
                if delta > 0:
                    up = (up * (period - 1) + delta) / period
                    down = (down * (period - 1)) / period
                else:
                    up = (up * (period - 1)) / period
                    down = (down * (period - 1) - delta) / period
                
                rs = up / down if down != 0 else 0
                rsiValues.append(100.0 - 100.0 / (1.0 + rs))
            
            return rsiValues[-1] if rsiValues else None
        except:
            return None
    
    def _calculateMacd(self, prices, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
        """Calculate MACD indicator."""
        try:
            emaFast = prices.ewm(span=fast).mean().iloc[-1]
            emaSlow = prices.ewm(span=slow).mean().iloc[-1]
            macdLine = emaFast - emaSlow
            
            # Simplified signal line (using last 9 MACD values)
            macdValues = (prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()).tail(signal)
            signalLine = macdValues.mean()
            
            return macdLine, signalLine
        except:
            return None, None
