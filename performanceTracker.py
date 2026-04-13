"""
Performance Tracker for Trading Strategies.

Thread-safe tracking of strategy performance using profit factor.
Maintains rolling window metrics and all-time statistics.
"""

import threading
from typing import Dict, List, Any
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks strategy performance using profit factor (sum of profits / sum of losses).
    Thread-safe: uses RLock() to prevent read/write conflicts when GUI queries while agent records trades.
    """
    
    def __init__(self, windowSize: int = 50):
        """
        Initialise performance tracker.
        
        Args:
            windowSize: Number of recent trades to consider for sliding window metrics (default: 50)
        """
        self._lock = threading.RLock()
        self.windowSize = windowSize
        
        # Per-strategy tracking
        # strategyName -> list of trades (executed trades only)
        self._tradeHistory: Dict[str, List[Dict]] = {}  
        # strategyName -> cached metrics for trades
        self._metricsCache: Dict[str, Dict] = {}
        # strategyName -> list of all recommendations (including HOLDs)
        self._recommendationHistory: Dict[str, List[Dict]] = {}  
    
    def recordTrade(self, strategyName: str, entryPrice: float, exitPrice: float, 
                   quantity: int, ticker: str, entryDate: str, exitDate: str, tradeType: str = 'long') -> None:
        """
        Record a closed trade for a strategy.
        
        Args:
            strategyName: Name of strategy that initiated this trade
            entryPrice: Price at trade entry
            exitPrice: Price at trade exit
            quantity: Number of shares traded
            ticker: Stock ticker
            entryDate: Trade entry date (YYYY-MM-DD)
            exitDate: Trade exit date (YYYY-MM-DD)
            tradeType: 'long' or 'short' for correct P&L calculation
        """
        with self._lock:
            if strategyName not in self._tradeHistory:
                self._tradeHistory[strategyName] = []
            
            # Calculate P&L correctly based on trade type
            if tradeType == 'short':
                pnl = (entryPrice - exitPrice) * quantity
            else:  # default to long
                pnl = (exitPrice - entryPrice) * quantity
            
            # Deduplication: check if this EXACT trade was just recorded (same entry date, exit date, prices)
            tradeSignatureTuple = (ticker, quantity, round(entryPrice, 4), round(exitPrice, 4), round(pnl, 4), entryDate)
            recentTrades = self._tradeHistory[strategyName][-3:] if len(self._tradeHistory[strategyName]) > 0 else []
            for recentTrade in recentTrades:
                recentSignature = (recentTrade['ticker'], recentTrade['quantity'],
                                  round(recentTrade['entryPrice'], 4), round(recentTrade['exitPrice'], 4),
                                  round(recentTrade['pnl'], 4), recentTrade['entryDate'])
                if tradeSignatureTuple == recentSignature:
                    logger.warning(f"[{strategyName}] DUPLICATE TRADE REJECTED: {ticker} {quantity}@{entryPrice:.2f}->{exitPrice:.2f} on {entryDate} P&L ${pnl:.2f}")
                    return
            
            trade = {
                'ticker': ticker,
                'entryPrice': entryPrice,
                'exitPrice': exitPrice,
                'quantity': quantity,
                'pnl': pnl,
                'entryDate': entryDate,
                'exitDate': exitDate,
                'returnPct': ((exitPrice - entryPrice) / entryPrice * 100) if entryPrice != 0 else 0
            }
            self._tradeHistory[strategyName].append(trade)
            
            # Invalidate cache so next query recalculates
            if strategyName in self._metricsCache:
                del self._metricsCache[strategyName]
            
            # Log with detailed breakdown
            allTradesCount = len(self._tradeHistory[strategyName])
            allTradesPnL = sum(t['pnl'] for t in self._tradeHistory[strategyName])
            tradeSignature = f"{ticker}_{quantity}_{entryPrice:.2f}_{exitPrice:.2f}_{pnl:.2f}"
            logger.info(f"[{strategyName}] Trade #{allTradesCount} [{tradeSignature}] recorded: {ticker} {quantity}@{entryPrice:.2f}->{exitPrice:.2f}, P&L ${pnl:.2f} | Total P&L all-time: ${allTradesPnL:.2f}")
    
    def getProfitFactor(self, strategyName: str) -> float:
        """
        Get profit factor for a strategy using sliding window.
        Profit Factor = sum(profits) / sum(losses)
        
        Special cases:
        - No trades: return 0.0
        - No losses (all profitable): return 100.0 (capped infinity)
        - No profits (all losing): return 0.0
        - Normal case: return sum(profits) / sum(losses)
        
        Args:
            strategyName: Name of the strategy
            
        Returns:
            Profit factor value (0.0 to 100.0+)
        """
        with self._lock:
            if strategyName not in self._tradeHistory or not self._tradeHistory[strategyName]:
                return 0.0
            
            # Get last windowSize trades
            recentTrades = self._tradeHistory[strategyName][-self.windowSize:]
            
            totalProfit = sum(t['pnl'] for t in recentTrades if t['pnl'] > 0)
            totalLoss = abs(sum(t['pnl'] for t in recentTrades if t['pnl'] < 0))
            
            # Handle edge cases
            if totalLoss < 1e-6:  # No losses or negligible losses
                if totalProfit > 1e-6:  # Strategy has profits but no losses = perfect
                    return 100.0  # Cap at 100 to indicate "exceptional"
                else:  # No trades or only break-even trades
                    return 0.0
            
            # Normal case: profits / losses
            profitFactor = totalProfit / totalLoss
            return profitFactor
    
    def getMetrics(self, strategyName: str) -> Dict[str, Any]:
        """
        Get metrics for a strategy (cached, sliding window).
        
        Args:
            strategyName: Name of the strategy
            
        Returns:
            Dict with profitFactor, totalTrades, winCount, lossCount, avgWin, avgLoss, avgPnL, totalPnL
        """
        with self._lock:
            if strategyName not in self._tradeHistory or not self._tradeHistory[strategyName]:
                return {
                    'profitFactor': 0.0,
                    'totalTrades': 0,
                    'winCount': 0,
                    'lossCount': 0,
                    'avgWin': 0.0,
                    'avgLoss': 0.0,
                    'avgPnL': 0.0,
                    'totalPnL': 0.0
                }
            
            # Get ALL trades for this strategy (not just window)
            allTrades = self._tradeHistory[strategyName]
            allTradesPnL = sum(t['pnl'] for t in allTrades)
            allWins = [t for t in allTrades if t['pnl'] > 0]
            allLosses = [t for t in allTrades if t['pnl'] < 0]
            
            # Get windowed trades for metrics calculation
            recentTrades = allTrades[-self.windowSize:]
            
            wins = [t for t in recentTrades if t['pnl'] > 0]
            losses = [t for t in recentTrades if t['pnl'] < 0]
            
            totalProfit = sum(t['pnl'] for t in wins) if wins else 0
            totalLoss = sum(t['pnl'] for t in losses) if losses else 0
            
            metrics = {
                'profitFactor': self.getProfitFactor(strategyName),
                'totalTrades': len(recentTrades),
                'winCount': len(wins),
                'lossCount': len(losses),
                'avgWin': totalProfit / len(wins) if wins else 0.0,
                'avgLoss': abs(totalLoss / len(losses)) if losses else 0.0,
                'avgPnL': sum(t['pnl'] for t in recentTrades) / len(recentTrades),
                'totalPnL': sum(t['pnl'] for t in recentTrades)
            }
            
            # Log detail if metrics show zero P&L but all-time P&L is non-zero
            if metrics['totalPnL'] < 0.01 and allTradesPnL > 1.0:
                logger.warning(f"[{strategyName}] METRICS MISMATCH: windowPnL=${metrics['totalPnL']:.2f} vs allTimePnL=${allTradesPnL:.2f} ({len(allTrades)} total trades, window={self.windowSize})")
                logger.warning(f"  Window trades: {len(recentTrades)}, Wins={len(wins)} (${totalProfit:.2f}), Losses={len(losses)} (${abs(totalLoss):.2f})")
            
            return metrics
