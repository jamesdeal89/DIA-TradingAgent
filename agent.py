'''
Stock Trading Intelligent Agent with Hyper-Heuristic Delegator.

Multi-strategy agent that selects trading approaches based on past performance (profit factor).
Strategies include: Sentiment Analysis, Mean Reversion, Technical Indicators, Fundamental Analysis.

Each agent runs in a separate thread and has:
- Per-agent isolated performance tracking (thread-safe)
- Epsilon-greedy strategy selection
- Query interface for GUI integration
- Lifecycle controls (pause/resume/stop)

Shared market: All agents trade on single StockExchange/MySQL instance (concurrent writes serialized).
'''

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import threading
import time
import logging
from datetime import datetime, timedelta
from tradingStrategy import TradingStrategy
from qLearningStrategy import DeepQLearningStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# CONCRETE STRATEGY: SENTIMENT ANALYSIS

class SentimentStrategy(TradingStrategy):
    """
    Analyses news sentiment to make trading decisions.
    Positive sentiment -> long, Negative sentiment -> short, Neutral -> hold.
    Aggregates sentiment over the analysis period for more robust signals.
    """
    
    def __init__(self):
        super().__init__(name="Sentiment", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """
        Analyse news sentiment for the ticker over analysisPeriod days.
        
        Args:
            analysisPeriod: Number of days to aggregate sentiment over
        
        Returns:
            - LONG if average sentiment is positive (score > 0.2)
            - SHORT if average sentiment is negative (score < -0.2)
            - HOLD if neutral
        """
        try:
            from datetime import datetime, timedelta
            
            # Gather sentiment data over the analysis period
            all_scores = []
            headline_count = 0
            
            for day_offset in range(analysisPeriod):
                current_date = (datetime.strptime(simDate, '%Y-%m-%d') - timedelta(days=day_offset)).strftime('%Y-%m-%d')
                headlines = exchange.getNewsForStock(ticker, mic, current_date)
                if headlines:
                    scores = [h.get('score', 0.0) for h in headlines]
                    all_scores.extend(scores)
                    headline_count += len(headlines)
            
            if not all_scores:
                return {
                    'action': 'hold',
                    'confidence': 0.3,
                    'reason': f'No news data available over {analysisPeriod} days',
                    'targetQuantity': 0
                }
            
            # Calculate average sentiment score across all collected headlines
            avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0
            
            # Determine action based on sentiment (threshold 0.05 for reasonable signal detection)
            if avg_score > 0.05:
                return {
                    'action': 'long',
                    'confidence': min(0.95, abs(avg_score)),
                    'reason': f'Positive sentiment ({avg_score:.2f}) from {headline_count} headlines over {analysisPeriod} days',
                    'targetQuantity': 1
                }
            elif avg_score < -0.05:
                # Recommend selling longs if sentiment is negative (agent checks portfolio)
                return {
                    'action': 'sell',
                    'confidence': min(0.95, abs(avg_score)),
                    'reason': f'Negative sentiment ({avg_score:.2f}) - exit long positions',
                    'targetQuantity': 0
                }
            else:
                return {
                    'action': 'hold',
                    'confidence': 0.5,
                    'reason': f'Neutral sentiment ({avg_score:.2f}) from {headline_count} headlines over {analysisPeriod} days',
                    'targetQuantity': 0
                }
        
        except Exception as e:
            logger.warning(f"SentimentStrategy.analyse() failed: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Error: {str(e)}',
                'targetQuantity': 0
            }


# CONCRETE STRATEGY: MEAN REVERSION

class MeanReversionStrategy(TradingStrategy):
    """
    Trades based on price deviation from mean.
    Dynamically scales lookback window based on analysisPeriod.
    If price < mean - 2*std LONG (buy the dip)
    If price > mean + 2*std SHORT (sell the peak)
    """
    
    def __init__(self):
        super().__init__(name="MeanReversion", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """
        Calculate if stock is above/below mean over analysisPeriod.
        
        Args:
            analysisPeriod: Window size for calculating mean and std (default 20, scales up with decision period)
        """
        try:
            # Scale lookback window based on analysis period
            lookback_window = max(20, analysisPeriod)
            required_history = lookback_window + 5  # Get extra data for safety
            
            # Get historical data leading up to simDate
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < lookback_window:
                return {
                    'action': 'hold',
                    'confidence': 0.2,
                    'reason': f'Insufficient historical data (< {lookback_window} days)',
                    'targetQuantity': 0
                }
            
            # Calculate moving average and std deviation over the scaled window
            close_prices = data['Close'].tail(lookback_window)
            mean_price = close_prices.mean()
            std_price = close_prices.std()
            current_price = close_prices.iloc[-1]
            
            # Calculate z-score (how many std devs away from mean)
            z_score = (current_price - mean_price) / std_price if std_price > 0 else 0
            
            # Trading logic (lower threshold 1.0 for more frequent trades)
            threshold = 1.0
            
            if z_score < -threshold:  # Price is threshold+ std devs below mean
                return {
                    'action': 'long',
                    'confidence': min(0.9, 0.5 + abs(z_score) * 0.15),
                    'reason': f'Price {current_price:.2f} is {abs(z_score):.2f}sd below {lookback_window}d mean {mean_price:.2f}',
                    'targetQuantity': 1
                }
            elif z_score > threshold:  # Price is threshold+ std devs above mean
                # Recommend selling (agent will check portfolio before executing)
                return {
                    'action': 'sell',
                    'confidence': min(0.9, 0.5 + abs(z_score) * 0.15),
                    'reason': f'Price {current_price:.2f} ({abs(z_score):.2f}sd above mean) - exit signal',
                    'targetQuantity': 0
                }
            else:
                return {
                    'action': 'hold',
                    'confidence': 0.4,
                    'reason': f'Price within normal range (z={z_score:.2f}) over {lookback_window}d',
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


# CONCRETE STRATEGY: TECHNICAL INDICATORS

class TechnicalStrategy(TradingStrategy):
    """
    Uses technical indicators: RSI, MACD, Bollinger Bands.
    - RSI > 70: overbought (SHORT signal)
    - RSI < 30: oversold (LONG signal)
    - MACD crossover: trend change signal
    """
    
    def __init__(self):
        super().__init__(name="Technical", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """
        Calculate RSI and MACD indicators over the analysis period.
        Scales indicator periods with analysis period for larger windows.
        """
        try:
            # Scale indicator periods based on analysis window
            rsi_period = max(14, analysisPeriod // 2)
            required_data = max(50, analysisPeriod * 2)
            
            # Get historical data for indicator calculation
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < rsi_period:
                return {
                    'action': 'hold',
                    'confidence': 0.2,
                    'reason': f'Insufficient data for technical analysis (< {rsi_period} days)',
                    'targetQuantity': 0
                }
            
            # Calculate RSI with scaled period
            rsi = self._calculateRsi(data['Close'], period=rsi_period)
            
            # Calculate MACD
            macd, signal = self._calculateMacd(data['Close'])
            
            # Determine action based on indicators
            confidence = 0.0
            reason = ""
            action = 'hold'
            
            # RSI signals (strongest signal)
            if rsi is not None:
                if rsi > 70:
                    action = 'short'
                    confidence = max(confidence, (rsi - 70) / 30)  # 0.0-1.0
                    reason = f"RSI {rsi:.1f} overbought"
                elif rsi < 30:
                    action = 'long'
                    confidence = max(confidence, (30 - rsi) / 30)  # 0.0-1.0
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
            if rsi is not None and rsi > 70:
                action = 'sell'
                confidence = max(confidence, (rsi - 70) / 30)
                reason = f"Selling overbought position (RSI {rsi:.1f})"
            
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
            rsi_values = [100.0 - 100.0 / (1.0 + rs)] * period
            
            for i in range(period, len(deltas)):
                delta = deltas.iloc[i]
                if delta > 0:
                    up = (up * (period - 1) + delta) / period
                    down = (down * (period - 1)) / period
                else:
                    up = (up * (period - 1)) / period
                    down = (down * (period - 1) - delta) / period
                
                rs = up / down if down != 0 else 0
                rsi_values.append(100.0 - 100.0 / (1.0 + rs))
            
            return rsi_values[-1] if rsi_values else None
        except:
            return None
    
    def _calculateMacd(self, prices, fast: int = 12, slow: int = 26, signal: int = 9):
        """Calculate MACD indicator."""
        try:
            ema_fast = prices.ewm(span=fast).mean().iloc[-1]
            ema_slow = prices.ewm(span=slow).mean().iloc[-1]
            macd_line = ema_fast - ema_slow
            
            # Simplified signal line (using last 9 MACD values)
            macd_values = (prices.ewm(span=fast).mean() - prices.ewm(span=slow).mean()).tail(signal)
            signal_line = macd_values.mean()
            
            return macd_line, signal_line
        except:
            return None, None


# CONCRETE STRATEGY: FUNDAMENTAL ANALYSIS

class FundamentalStrategy(TradingStrategy):
    """
    Uses basic price-based statistics.
    """
    
    def __init__(self):
        super().__init__(name="Fundamental", version="1.0")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """
        Placeholder fundamental analysis scaled by analysis period.
        Uses price volatility and momentum over the analysis window as proxy for fundamental strength.
        """
        try:
            # Scale lookback window based on analysis period
            lookback_window = max(30, analysisPeriod)
            
            # Get historical data
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < lookback_window:
                return {
                    'action': 'hold',
                    'confidence': 0.3,
                    'reason': f'Insufficient data for fundamental analysis (< {lookback_window} days)',
                    'targetQuantity': 0
                }
            
            # Calculate volatility (proxy for risk/strength) over the window
            returns = data['Close'].pct_change()
            volatility = returns.std()
            
            # Calculate price momentum over the analysis period
            price_ago = data['Close'].iloc[-lookback_window]
            price_today = data['Close'].iloc[-1]
            momentum = (price_today - price_ago) / price_ago
            
            # Decision logic: lower momentum threshold (0.03) and higher volatility tolerance for more trades
            if momentum > 0.03 and volatility < 0.06:
                return {
                    'action': 'long',
                    'confidence': min(0.8, 0.4 + momentum),
                    'reason': f'Strong momentum ({momentum*100:.1f}%) over {lookback_window}d with controlled volatility',
                    'targetQuantity': 1
                }
            elif momentum < -0.03 and volatility < 0.06:
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


# PERFORMANCE TRACKER

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
                   quantity: int, ticker: str, entryDate: str, exitDate: str) -> None:
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
        """
        with self._lock:
            if strategyName not in self._tradeHistory:
                self._tradeHistory[strategyName] = []
            
            pnl = (exitPrice - entryPrice) * quantity
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
            
            logger.info(f"[{strategyName}] Trade recorded: {ticker} {quantity}@{entryPrice}->{exitPrice}, P&L=${pnl:.2f}")
    
    def getProfitFactor(self, strategyName: str) -> float:
        """
        Get profit factor for a strategy using sliding window.
        Profit Factor = sum(profits) / sum(losses)
        
        Returns 0.0 if no trades, handles division by zero.
        """
        with self._lock:
            if strategyName not in self._tradeHistory or not self._tradeHistory[strategyName]:
                return 0.0
            
            # Get last windowSize trades
            recentTrades = self._tradeHistory[strategyName][-self.windowSize:]
            
            totalProfit = sum(t['pnl'] for t in recentTrades if t['pnl'] > 0)
            totalLoss = abs(sum(t['pnl'] for t in recentTrades if t['pnl'] < 0))
            
            if totalLoss < 1e-6:  # No losses or negligible losses
                return 1.0 if totalProfit > 0 else 0.0
            
            return totalProfit / totalLoss
    
    def getMetrics(self, strategyName: str) -> Dict[str, Any]:
        """
        Get metrics for a strategy (cached, sliding window).
        
        Returns:
            {
                'profitFactor': float,
                'totalTrades': int,
                'winCount': int,
                'lossCount': int,
                'avgWin': float,
                'avgLoss': float,
                'avgPnl': float,
                'totalPnl': float
            }
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
                    'avgPnl': 0.0,
                    'totalPnl': 0.0
                }
            
            recentTrades = self._tradeHistory[strategyName][-self.windowSize:]
            
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
                'avgPnl': sum(t['pnl'] for t in recentTrades) / len(recentTrades),
                'totalPnl': sum(t['pnl'] for t in recentTrades)
            }
            
            self._metricsCache[strategyName] = metrics
            return metrics
    
    def getAllMetrics(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all strategies (returns copy to prevent external mutation)."""
        with self._lock:
            return {
                strategy: self.getMetrics(strategy)
                for strategy in self._tradeHistory.keys()
            }
    
    def resetStrategy(self, strategyName: str) -> None:
        """Clear performance history for a strategy."""
        with self._lock:
            if strategyName in self._tradeHistory:
                del self._tradeHistory[strategyName]
            if strategyName in self._metricsCache:
                del self._metricsCache[strategyName]
    
    def recordRecommendation(self, strategyName: str, action: str, ticker: str, 
                            price: float, confidence: float, simDate: str) -> None:
        """
        Record a strategy recommendation (LONG, SHORT, or HOLD).
        
        Args:
            strategyName: Name of strategy making recommendation
            action: 'long', 'short', or 'hold'
            ticker: Stock ticker
            price: Price at time of recommendation
            confidence: Strategy confidence in recommendation (0.0-1.0)
            simDate: Simulation date (YYYY-MM-DD)
        """
        with self._lock:
            if strategyName not in self._recommendationHistory:
                self._recommendationHistory[strategyName] = []
            
            recommendation = {
                'action': action,
                'ticker': ticker,
                'price': price,
                'confidence': confidence,
                'date': simDate,
                'outcome': 'PENDING'
            }
            self._recommendationHistory[strategyName].append(recommendation)
            logger.debug(f"[{strategyName}] Recommendation recorded: {action} {ticker} @ {price:.2f}")
    
    def scoreRecommendations(self, exchange, thresholdPct: float = 2.0) -> None:
        """
        Retroactively score HOLD recommendations based on actual price movement.
        
        Args:
            exchange: StockExchange instance to get current prices
            thresholdPct: Percentage threshold for outcome classification (default 2%)
        """
        with self._lock:
            from datetime import datetime, timedelta
            
            for strategyName, trades in self._recommendationHistory.items():
                for rec in trades:
                    if rec['outcome'] != 'PENDING':
                        continue  # Already scored
                    
                    # Only score HOLD recommendations; LONG/SHORT are scored when executed
                    if rec['action'] != 'hold':
                        continue
                    
                    try:
                        # Get current price for the ticker (would need to be called with current date context)
                        # This is a simplified version; in practice you'd pass current price
                        # For now, mark as pending until we integrate with exchange
                        pass
                    except Exception as e:
                        logger.warning(f"Could not score recommendation for {rec['ticker']}: {e}")
    
    def getRecommendationMetrics(self, strategyName: str) -> Dict[str, Any]:
        """
        Get recommendation quality metrics for a strategy.
        
        Returns:
            {
                'totalRecommendations': int,
                'holdCount': int,
                'holdCorrect': int,
                'holdMissedLong': int,
                'holdMissedShort': int,
                'holdAccuracy': float (0.0-1.0),
                'longCount': int,
                'shortCount': int,
                'totalExecuted': int
            }
        """
        with self._lock:
            if strategyName not in self._recommendationHistory:
                return {
                    'totalRecommendations': 0,
                    'holdCount': 0,
                    'holdCorrect': 0,
                    'holdMissedLong': 0,
                    'holdMissedShort': 0,
                    'holdAccuracy': 0.0,
                    'longCount': 0,
                    'shortCount': 0,
                    'totalExecuted': 0
                }
            
            recs = self._recommendationHistory[strategyName]
            
            holds = [r for r in recs if r['action'] == 'hold']
            hold_correct = len([r for r in holds if r['outcome'] == 'CORRECT'])
            hold_missed_long = len([r for r in holds if r['outcome'] == 'MISSED_LONG'])
            hold_missed_short = len([r for r in holds if r['outcome'] in ('MISSED_SHORT', 'SOLD')])
            
            longs = [r for r in recs if r['action'] == 'long']
            shorts = [r for r in recs if r['action'] == 'short']
            
            hold_accuracy = hold_correct / len(holds) if holds else 0.0
            
            return {
                'totalRecommendations': len(recs),
                'holdCount': len(holds),
                'holdCorrect': hold_correct,
                'holdMissedLong': hold_missed_long,
                'holdMissedShort': hold_missed_short,
                'holdAccuracy': hold_accuracy,
                'longCount': len(longs),
                'shortCount': len(shorts),
                'totalExecuted': len(longs) + len(shorts),
                'pendingCount': len([r for r in recs if r['outcome'] == 'PENDING'])
            }


# EPSILON-GREEDY STRATEGY SELECTOR

class StrategySelector:
    """
    Selects which strategy to use at a given timestep using epsilon-greedy sampling.
    Balances exploitation (use top performer) with exploration (try others).
    """
    
    def __init__(self, strategies: Dict[str, TradingStrategy], epsilon: float = 0.2, 
                 minTradesRequired: int = 3):
        """
        Initialise strategy selector.
        
        Args:
            strategies: Dict mapping strategy names to TradingStrategy instances
            epsilon: Probability of random exploration (default: 0.2 = 20% random)
            minTradesRequired: Minimum trades per strategy before ranking (default: 3)
        """
        self.strategies = strategies
        self.epsilon = epsilon
        self.minTradesRequired = minTradesRequired
        self._selectionHistory = []  
    
    def selectStrategy(self, performanceMetrics: Dict[str, Dict], 
                      currentTradeCount: int) -> str:
        """
        Select a strategy using epsilon-greedy approach.
        
        Exploration mode (uniform random): if tradeCount < minTradesRequired
        Exploitation mode: with prob epsilon, random; else highest profitFactor
        
        Args:
            performanceMetrics: Dict of strategyName -> metrics from PerformanceTracker.getAllMetrics()
            currentTradeCount: Total trades executed so far by this agent
        
        Returns:
            Selected strategy name
        """
        import random
        import numpy as np
        
        availableStrategies = list(self.strategies.keys())
        
        # Exploration mode: random until sufficient data
        if currentTradeCount < self.minTradesRequired:
            selected = random.choice(availableStrategies)
            logger.info(f"[StrategySelector] Exploration mode ({currentTradeCount}/{self.minTradesRequired} trades): selected {selected}")
            self._selectionHistory.append(selected)
            return selected
        
        # Exploitation mode with epsilon-greedy
        if random.random() < self.epsilon:
            # Explore: select random strategy
            selected = random.choice(availableStrategies)
            logger.info(f"[StrategySelector] Exploration phase (epsilon={self.epsilon}): selected {selected}")
        else:
            # Exploit: select highest profit factor, with random tiebreaker for equal performers
            bestPf = -np.inf
            bestStrategies = []
            
            for strategyName in availableStrategies:
                metrics = performanceMetrics.get(strategyName, {})
                pf = metrics.get('profitFactor', 0.0)
                
                if pf > bestPf:
                    bestPf = pf
                    bestStrategies = [strategyName]
                elif pf == bestPf:
                    bestStrategies.append(strategyName)
            
            # Random tiebreaker: when multiple strategies have same best PF, pick randomly
            selected = random.choice(bestStrategies) if bestStrategies else random.choice(availableStrategies)
            logger.info(f"[StrategySelector] Exploitation: selected {selected} (PF={bestPf:.2f})")
        
        self._selectionHistory.append(selected)
        return selected
    
    def getWeights(self) -> Dict[str, float]:
        """Get current weight distribution across strategies based on selection history."""
        if not self._selectionHistory:
            # Uniform distribution if no history
            n = len(self.strategies)
            return {name: 1.0/n for name in self.strategies.keys()}
        
        # Count selections
        from collections import Counter
        counts = Counter(self._selectionHistory)
        total = sum(counts.values())
        
        return {
            strategyName: counts.get(strategyName, 0) / total
            for strategyName in self.strategies.keys()
        }


# AGENT

class Agent:
    '''
    The intelligent stock trading agent with hyper-heuristic strategy delegator.
    
    - Runs in background thread
    - Maintains per-agent performance tracking (thread-safe)
    - Selects strategies based on past profit factors
    - Provides query interface for GUI
    - Supports lifecycle control (pause/resume/stop)
    - analyses all stocks in market at each timestep
    '''
    
    # Market Identifier Code to ticker mapping
    MIC_TICKERS = {
        'XNAS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'XLON': ['BP', 'SHEL', 'AZN', 'VOD', 'LLOY'],
        'XHKG': ['0700', '0388', '1113', '0005', '0011'],
        'XJPX': ['7203', '6758', '9984', '6861', '8053']
    }

    def __init__(self, agentId: int, accountId: int, mic: str = 'XLON', preferredStrategy: Optional[str] = None, 
                 bannedStrategies: List[str] = None, simDate: str = None, decisionPeriod: int = 1):
        '''
        Initialise the agent.
        
        Args:
            agentId: Unique agent identifier
            accountId: Trading account ID (for executing trades)
            mic: Market Identifier Code ('XNAS', 'XLON', 'XHKG', 'XJPX')
            preferredStrategy: Optional strategy to preferentially select
            bannedStrategies: List of strategy names to exclude from selection
            simDate: Initial simulation date (YYYY-MM-DD format)
            decisionPeriod: Days between trading decisions (default 1 = daily decisions)
        '''
        self.agentId = agentId
        self.accountId = accountId
        self.mic = mic
        self.preferredStrategy = preferredStrategy
        self.bannedStrategies = bannedStrategies or []
        self.simDate = simDate
        self.decisionPeriod = max(1, decisionPeriod)  # Ensure at least 1
        
        # Performance / strategy management
        self.performanceTracker = PerformanceTracker(windowSize=50)
        self.strategies: Dict[str, TradingStrategy] = {}  # Will be populated by _initialiseStrategies()
        self.strategySelector: Optional[StrategySelector] = None
        
        # Trade tracking
        self.totalTrades = 0
        self._executionLog: List[Dict] = []
        self._timestepCounter = 0  # Tracks timesteps since last decision
        
        # Lifecycle control
        self._pauseFlag = False
        self._stopFlag = False
        self._lock = threading.Lock()
        
        # Initialise strategies
        self._initialiseStrategies()
        
        logger.info(f"Agent {agentId} initialised (accountId={accountId}, MIC={mic}, simDate={simDate}, decisionPeriod={self.decisionPeriod}d)")
    
    def _initialiseStrategies(self) -> None:
        """Initialise all available strategies, respecting banned/preferred."""
        # Instantiate all trading strategies
        all_strategies = {
            'Sentiment': SentimentStrategy(),
            'MeanReversion': MeanReversionStrategy(),
            'Technical': TechnicalStrategy(),
            'Fundamental': FundamentalStrategy(),
            'DeepQL': DeepQLearningStrategy(),
        }
        
        # Apply bans
        self.strategies = {name: strat for name, strat in all_strategies.items() 
                          if name not in self.bannedStrategies}
        
        # If preferredStrategy is set and not banned, prioritise it
        if self.preferredStrategy and self.preferredStrategy in self.strategies:
            logger.info(f"Agent {self.agentId}: Using preferred strategy {self.preferredStrategy}")
        
        logger.info(f"Agent {self.agentId}: Initialised {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        
        self.strategySelector = StrategySelector(self.strategies, epsilon=0.35, minTradesRequired=2)
    
    # LIFECYCLE CONTROL
    
    def pause(self) -> None:
        """Pause agent execution between iterations."""
        with self._lock:
            self._pauseFlag = True
            logger.info(f"Agent {self.agentId} paused")
    
    def resume(self) -> None:
        """Resume agent execution."""
        with self._lock:
            self._pauseFlag = False
            logger.info(f"Agent {self.agentId} resumed")
    
    def stop(self) -> None:
        """Stop agent execution cleanly."""
        with self._lock:
            self._stopFlag = True
            logger.info(f"Agent {self.agentId} stop signal sent")
    
    def isRunning(self) -> bool:
        """Check if agent is running (not stopped)."""
        with self._lock:
            return not self._stopFlag
    
    def isPaused(self) -> bool:
        """Check if agent is paused."""
        with self._lock:
            return self._pauseFlag
    
    # EXECUTION: TIMESTEP (ALL STOCKS)
    
    def runTimestep(self, exchange, simDate: str = None) -> None:
        '''
        Execute one timestep of the trading loop: analyse and trade ALL stocks in the agent's market.
        
        Called by background thread every decision period. Since date advancement is handled
        by the GUI loop (which advances by decisionPeriod days), this method
        should always execute a complete analysis and trade cycle.
        
        Args:
            exchange: StockExchange instance for trade execution and data access
            simDate: Current simulation date (YYYY-MM-DD). If None, uses agent.simDate
        '''
        if simDate is None:
            simDate = self.simDate
        
        # Check lifecycle flags
        with self._lock:
            if self._stopFlag:
                logger.info(f"Agent {self.agentId}: stop_flag set, skipping timestep")
                return
            
            if self._pauseFlag:
                logger.debug(f"Agent {self.agentId}: paused, skipping timestep")
                return
        
        # Increment timestep counter for tracking purposes
        self._timestepCounter += 1
        
        # Get all tickers for this market
        tickers = self.MIC_TICKERS.get(self.mic, [])
        if not tickers:
            logger.warning(f"Agent {self.agentId}: No tickers found for MIC {self.mic}")
            return
        
        print(f"[Agent {self.agentId}] Decision Day {simDate} (period={self.decisionPeriod}d) on {self.mic}: analyzing {len(tickers)} stocks")
        
        # Select ONE strategy for this entire timestep (used for all stocks)
        try:
            metrics = self.performanceTracker.getAllMetrics()
            selected_strategy_name = self.strategySelector.selectStrategy(metrics, self.totalTrades)
            selected_strategy = self.strategies[selected_strategy_name]
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Strategy selection failed - {e}")
            logger.error(f"Agent {self.agentId}: strategy selection failed: {e}")
            return
        
        # Analyse and trade each stock in sequence using the same strategy
        for ticker in tickers:
            try:
                self._analyseAndTrade(exchange, ticker, simDate, selected_strategy_name, selected_strategy)
            except Exception as e:
                print(f"[Agent {self.agentId}] ERROR processing {ticker}: {e}")
                logger.error(f"Agent {self.agentId}: failed to process {ticker}: {e}")
        
        # Log portfolio snapshot at decision point
        exchange.logPortfolioSnapshot(self.accountId, self.agentId, simDate)
    
    def _analyseAndTrade(self, exchange, ticker: str, simDate: str, 
                         selected_strategy_name: str, selected_strategy: TradingStrategy) -> None:
        '''
        Internal method: analyse a single ticker using the given strategy and execute trade if recommended.
        Only called on decision days (every N days based on decisionPeriod).
        
        Args:
            exchange: StockExchange instance
            ticker: Stock ticker symbol
            simDate: Current simulation date (YYYY-MM-DD)
            selected_strategy_name: Name of the strategy to use
            selected_strategy: Strategy instance to use for analysis
        '''
        
        # Get trade recommendation from strategy with decision period as analysis window
        try:
            recommendation = selected_strategy.analyse(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
            action_rec = recommendation.get('action', 'hold')
            confidence_rec = recommendation.get('confidence', 0)
            
            print(f"[Agent {self.agentId}] {ticker}: {selected_strategy_name} -> {action_rec.upper()} ({confidence_rec*100:.0f}%)")
            logger.info(f"Agent {self.agentId}: {ticker} {selected_strategy_name} recommended {action_rec} ({confidence_rec:.1%})")
            
            # Record ALL recommendations (including HOLD) for later quality assessment
            try:
                current_price = exchange.getStockData(exchange.getMicTicker(ticker, self.mic), start=None, end=simDate)
                if current_price is not None and len(current_price) > 0:
                    price_at_rec = float(current_price['Close'].iloc[-1])
                    self.performanceTracker.recordRecommendation(
                        selected_strategy_name, action_rec, ticker, price_at_rec, confidence_rec, simDate
                    )
            except Exception as rec_err:
                logger.debug(f"Could not record recommendation for {ticker}: {rec_err}")
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Strategy analysis failed for {ticker} - {e}")
            logger.error(f"Agent {self.agentId}: strategy.analyse({ticker}) failed: {e}")
            return
        
        action = recommendation.get('action', 'hold')
        if action == 'hold':
            logger.debug(f"Agent {self.agentId}: HOLD on {ticker}")
            return
        
        # Execute buy/short
        try:
            target_quantity = recommendation.get('targetQuantity', 0)
            self.executeAction(exchange, ticker, action, target_quantity, simDate, selected_strategy_name, confidence_rec)
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Trade execution failed for {ticker} - {e}")
            logger.error(f"Agent {self.agentId}: executeAction({ticker}) failed: {e}")
    
    def executeAction(self, exchange, ticker: str, action: str, quantity: int, simDate: str, strategyName: str, confidence: float = 0.0) -> None:
        '''
        Execute a trade action (LONG, SHORT, SELL, or HOLD).
        
        Args:
            exchange: StockExchange instance
            ticker: Stock ticker symbol
            action: Trade action ('long', 'short', 'sell', or 'hold')
            quantity: Number of shares to trade
            simDate: Current simulation date (YYYY-MM-DD)
            strategyName: Name of strategy recommending the trade
            confidence: Confidence score from strategy (0.0-1.0)
        '''
        try:
            if action == 'long':
                print(f"[Agent {self.agentId}] EXECUTING LONG: {ticker} x{quantity}")
                result = exchange.placeLong(ticker, self.mic, quantity, self.accountId, simDate, 
                                 strategyName=strategyName, agentId=self.agentId)
                if result == -1:
                    print(f"[Agent {self.agentId}] TRADE ERROR: long {ticker} failed - insufficient balance or delisted stock")
                    logger.error(f"Agent {self.agentId}: long {ticker} execution failed")
                    return
                print(f"[Agent {self.agentId}] SUCCESS: LONG {ticker} x{quantity} via {strategyName}")
                logger.info(f"Agent {self.agentId}: LONG {ticker} x{quantity} executed via {strategyName}")
            elif action == 'short':
                print(f"[Agent {self.agentId}] EXECUTING SHORT: {ticker} x{quantity}")
                result = exchange.placeShort(ticker, self.mic, quantity, self.accountId, simDate, 
                                  strategyName=strategyName, agentId=self.agentId)
                if result == -1:
                    print(f"[Agent {self.agentId}] TRADE ERROR: short {ticker} failed - check error log")
                    logger.error(f"Agent {self.agentId}: short {ticker} execution failed")
                    return
                print(f"[Agent {self.agentId}] SUCCESS: SHORT {ticker} x{quantity} via {strategyName}")
                logger.info(f"Agent {self.agentId}: SHORT {ticker} x{quantity} executed via {strategyName}")
            elif action == 'sell':
                # Sell existing long position
                portfolio = exchange.checkPortfolio(self.accountId, simDate)
                if portfolio is not None and not portfolio.empty:
                    long_positions = portfolio[(portfolio['ticker'] == ticker) & (portfolio['tradeType'] == 'long')]
                    if not long_positions.empty:
                        pos = long_positions.iloc[0]
                        sell_qty = int(pos['quantity'])
                        entry_price = float(pos.get('entryPrice', 0))
                        entry_date = pos.get('entryDate', simDate)
                        pos_strategy = pos.get('strategyName', strategyName)  
                        
                        print(f"[Agent {self.agentId}] EXECUTING SELL: {ticker} x{sell_qty}")
                        result = exchange.sellLong(self.accountId, ticker, self.mic, sell_qty, simDate,
                                        strategyName=pos_strategy, entryPrice=entry_price, entryDate=entry_date)
                        if result == -1:
                            print(f"[Agent {self.agentId}] TRADE ERROR: sell {ticker} failed - delisted stock")
                            logger.error(f"Agent {self.agentId}: sell {ticker} execution failed")
                            return
                        print(f"[Agent {self.agentId}] SUCCESS: SOLD {ticker} x{sell_qty} via {strategyName}")
                        logger.info(f"Agent {self.agentId}: SOLD {ticker} x{sell_qty} executed via {strategyName}")
                    else:
                        logger.info(f"Agent {self.agentId}: No long position to sell for {ticker}")
                        return
                else:
                    logger.info(f"Agent {self.agentId}: No portfolio data for {ticker}")
                    return
            
            self.totalTrades += 1
            self._executionLog.append({
                'strategy': strategyName,
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'confidence': confidence,
                'timestamp': simDate
            })
        except Exception as e:
            print(f"[Agent {self.agentId}] TRADE ERROR: {action} {ticker} failed - {e}")
            logger.error(f"Agent {self.agentId}: {action} {ticker} execution failed: {e}")
            raise
    
    # QUERY INTERFACE FOR GUI
    
    def getPerformanceMetrics(self, strategyName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get performance metrics for one or all strategies.
        
        Args:
            strategyName: If specified, get metrics for this strategy; else all strategies
        
        Returns:
            Single strategy metrics or Dict[strategyName -> metrics]
        """
        if strategyName:
            return self.performanceTracker.getMetrics(strategyName)
        else:
            return self.performanceTracker.getAllMetrics()
    
    def getRecentTrades(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent closed trades across all strategies.
        
        Returns:
            List of trade dicts (last N trades, most recent first)
        """
        # TODO
        return self._executionLog[-limit:]
    
    def getCurrentOpenPositions(self) -> List[Dict[str, Any]]:
        """
        Get current open positions in portfolio.
        
        Returns:
            List of active long/short positions
        """
        # TODO
        return []
    
    def getStrategyWeights(self) -> Dict[str, float]:
        """Get current epsilon-greedy weights for each strategy."""
        if self.strategySelector:
            return self.strategySelector.getWeights()
        return {}
    
    # MAIN ITERATION LOOP 
    
    def setSimDate(self, simDate: str) -> None:
        '''
        Update the current simulation date for the agent.
        
        Args:
            simDate: New simulation date (YYYY-MM-DD)
        '''
        self.simDate = simDate
        print(f"DEBUG: Agent {self.agentId} simDate updated to {self.simDate}")
    
    def setDecisionPeriod(self, decisionPeriod: int) -> None:
        '''
        Update the decision period for trading (window size in days).
        
        Args:
            decisionPeriod: Number of days between trading decisions (minimum 1)
        '''
        self.decisionPeriod = max(1, decisionPeriod)
        print(f"DEBUG: Agent {self.agentId} decisionPeriod updated to {self.decisionPeriod} days")
        logger.info(f"Agent {self.agentId}: decision period changed to {self.decisionPeriod} days")
    
    def getDecisionPeriod(self) -> int:
        '''
        Get the current decision period.
        
        Returns:
            Current decision period in days
        '''
        return self.decisionPeriod
    
    def getTimestepCounter(self) -> int:
        '''
        Get the current timestep counter (progess in current decision window).
        
        Returns:
            Current timestep in the decision window (0 to decisionPeriod-1)
        '''
        return self._timestepCounter % self.decisionPeriod if self.decisionPeriod > 0 else 0
    
    def runIteration(self, exchange, ticker: str, simDate: str = None) -> None:
        '''
        Execute one iteration of the trading loop.
        
        Called iteratively by a background thread.
        1. Check pause/stop flags
        2. Select strategy based on past performance
        3. Get trade recommendation from strategy
        4. Execute trade if recommended
        5. Log execution
        
        Args:
            exchange: StockExchange instance for trade execution and data access
            ticker: Stock ticker to trade
            simDate: Current simulation date (YYYY-MM-DD). If None, uses agent.simDate
        
        NOTE: yFinance rate-limit: best to use 2s wait between requests.
        For backtesting, use larger date skips (e.g., weekly instead of daily).
        '''
        if simDate is None:
            simDate = self.simDate
        
        # Check lifecycle flags
        with self._lock:
            if self._stopFlag:
                logger.info(f"Agent {self.agentId}: stop_flag set, exiting iteration")
                return
            
            if self._pauseFlag:
                logger.debug(f"Agent {self.agentId}: paused, skipping iteration")
                return
        
        # Initialise strategies if not already done
        if not self.strategies:
            self._initialiseStrategies()
        
        # Select strategy based on performance
        try:
            metrics = self.performanceTracker.getAllMetrics()
            selected_strategy_name = self.strategySelector.selectStrategy(metrics, self.totalTrades)
            selected_strategy = self.strategies[selected_strategy_name]
            print(f"[Agent {self.agentId}] Selected strategy: {selected_strategy_name}")
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Strategy selection failed - {e}")
            logger.error(f"Agent {self.agentId}: strategy selection failed: {e}")
            return
        
        # Get trade recommendation from strategy with decision period as analysis window
        try:
            recommendation = selected_strategy.analyse(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
            action_rec = recommendation.get('action', 'hold')
            confidence_rec = recommendation.get('confidence', 0)
            logger.info(f"Agent {self.agentId}: {selected_strategy_name} recommended {action_rec} ({confidence_rec:.1%} confidence)")
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Strategy analysis failed - {e}")
            logger.error(f"Agent {self.agentId}: strategy.analyse() failed: {e}")
            return
        
        # Execute trade if recommended (non-hold)
        action = recommendation.get('action', 'hold')
        if action == 'hold':
            print(f"[Agent {self.agentId}] HOLD on {ticker}")
            logger.info(f"Agent {self.agentId}: HOLD (no action taken)")
            return
        
        try:
            quantity = recommendation.get('targetQuantity', 1)
            
            if action == 'long':
                print(f"[Agent {self.agentId}] EXECUTING LONG: {ticker} x{quantity}")
                exchange.placeLong(ticker, self.mic, quantity, self.accountId, simDate, 
                                 strategyName=selected_strategy_name, agentId=self.agentId)
                print(f"[Agent {self.agentId}] SUCCESS: LONG {ticker} x{quantity} via {selected_strategy_name}")
                logger.info(f"Agent {self.agentId}: LONG {ticker} x{quantity} executed via {selected_strategy_name}")
            elif action == 'short':
                print(f"[Agent {self.agentId}] EXECUTING SHORT: {ticker} x{quantity}")
                exchange.placeShort(ticker, self.mic, quantity, self.accountId, simDate, 
                                  strategyName=selected_strategy_name, agentId=self.agentId)
                print(f"[Agent {self.agentId}] SUCCESS: SHORT {ticker} x{quantity} via {selected_strategy_name}")
                logger.info(f"Agent {self.agentId}: SHORT {ticker} x{quantity} executed via {selected_strategy_name}")
            
            self.totalTrades += 1
            self._executionLog.append({
                'strategy': selected_strategy_name,
                'ticker': ticker,
                'action': action,
                'quantity': quantity,
                'confidence': recommendation.get('confidence', 0),
                'timestamp': simDate
            })
        
        except Exception as e:
            print(f"[Agent {self.agentId}] ERROR: Trade execution failed - {e}")
            logger.error(f"Agent {self.agentId}: trade execution failed: {e}")


# AGENT MANAGER

class AgentManager:
    """
    Manages multiple trading agents running in parallel.
    Each agent runs in its own background thread.
    Manages simulation loop (date iteration, ticker rotation).
    Thread-safe: all shared state protected by locks.
    """
    
    def __init__(self, exchange, tickers: List[str], mics: Dict[str, str],
                 date_range: tuple, date_step_days: int = 1):
        """
        Initialise AgentManager.
        
        Args:
            exchange: StockExchange instance (shared by all agents)
            tickers: List of stock tickers to trade (e.g., ['AAPL', 'MSFT'])
            mics: Dict mapping ticker -> MIC (e.g., {'AAPL': 'XNAS'})
            date_range: Tuple of (start_date, end_date) in YYYY-MM-DD format
            date_step_days: Number of days to step forward per iteration (default: 1)
        """
        self.exchange = exchange
        self.tickers = tickers
        self.mics = mics
        self.dateStep = date_step_days
        
        # Parse date range
        self.startDate = datetime.strptime(date_range[0], '%Y-%m-%d')
        self.endDate = datetime.strptime(date_range[1], '%Y-%m-%d')
        self.currentDate = self.startDate
        
        # Agent registry
        self.agents: Dict[int, Agent] = {}
        self.agentThreads: Dict[int, threading.Thread] = {}
        
        # Lifecycle control
        self._lock = threading.RLock()
        self._running = False
        self._paused = False
        self.mainLoopThread: Optional[threading.Thread] = None
        
        logger.info(f"AgentManager: Initialised with {len(tickers)} tickers, "
                   f"date range {date_range[0]} to {date_range[1]}")
    
    def addAgent(self, agentId: int, accountId: int, mic: str, preferredStrategy: Optional[str] = None,
                bannedStrategies: Optional[List[str]] = None) -> Agent:
        """
        Create and register a new agent.
        
        Args:
            agentId: Unique agent ID
            accountId: Trading account ID for this agent
            mic: Market Identifier Code (e.g., 'XNAS')
            preferredStrategy: Optional preferred strategy name
            bannedStrategies: Optional list of strategy names to ban
        
        Returns:
            Newly created Agent instance
        """
        with self._lock:
            if agentId in self.agents:
                logger.warning(f"Agent {agentId} already exists, skipping")
                return self.agents[agentId]
            
            agent = Agent(
                agentId=agentId,
                accountId=accountId,
                mic=mic,
                preferredStrategy=preferredStrategy,
                bannedStrategies=bannedStrategies
            )
            self.agents[agentId] = agent
            logger.info(f"AgentManager: Added agent {agentId} (accountId={accountId}, MIC={mic})")
            return agent
    
    def removeAgent(self, agentId: int) -> bool:
        """
        Remove an agent from the manager.
        
        Args:
            agentId: Agent ID to remove
        
        Returns:
            True if successful, False if not found
        """
        with self._lock:
            if agentId not in self.agents:
                logger.warning(f"Agent {agentId} not found")
                return False
            
            # Stop the agent thread if running
            if agentId in self.agentThreads and self.agentThreads[agentId].is_alive():
                self.agents[agentId].stop()
                self.agentThreads[agentId].join(timeout=5.0)
            
            del self.agents[agentId]
            if agentId in self.agentThreads:
                del self.agentThreads[agentId]
            
            logger.info(f"AgentManager: Removed agent {agentId}")
            return True
    
    def start(self) -> None:
        """Start the simulation loop in a background thread."""
        with self._lock:
            if self._running:
                logger.warning("AgentManager already running")
                return
            
            self._running = True
            self._paused = False
        
        # Start main simulation loop thread
        self.mainLoopThread = threading.Thread(target=self._simulationLoop, daemon=False)
        self.mainLoopThread.start()
        logger.info("AgentManager: Simulation started")
    
    def stop(self) -> None:
        """Stop all agents and the simulation loop."""
        with self._lock:
            self._running = False
        
        # Signal all agents to stop
        for agent in self.agents.values():
            agent.stop()
        
        # Wait for all agent threads to finish
        for thread in self.agentThreads.values():
            thread.join(timeout=5.0)
        
        # Wait for main loop to finish
        if self.mainLoopThread:
            self.mainLoopThread.join(timeout=5.0)
        
        logger.info("AgentManager: All agents and simulation stopped")
    
    def pause(self) -> None:
        """Pause all agents."""
        with self._lock:
            self._paused = True
        
        for agent in self.agents.values():
            agent.pause()
        
        logger.info("AgentManager: All agents paused")
    
    def resume(self) -> None:
        """Resume all agents."""
        with self._lock:
            self._paused = False
        
        for agent in self.agents.values():
            agent.resume()
        
        logger.info("AgentManager: All agents resumed")
    
    def _simulationLoop(self) -> None:
        """
        Main simulation loop (runs in background thread).
        Iterates through dates, executes agents on each ticker, closes aged shorts.
        """
        try:
            while True:
                with self._lock:
                    if not self._running:
                        break
                    
                    if self._paused:
                        current_date_str = self.currentDate.strftime('%Y-%m-%d')
                        logger.debug(f"Simulation paused at {current_date_str}")
                        # Short sleep to avoid busy-wait
                        time.sleep(0.5)
                        continue
                    
                    # Check if we've reached end date
                    if self.currentDate > self.endDate:
                        logger.info("Simulation reached end date")
                        self._running = False
                        break
                    
                    current_date_str = self.currentDate.strftime('%Y-%m-%d')
                
                # Execute one iteration for all agents on all tickers
                self._executeIterationForDate(current_date_str)
                
                # Auto-close aged shorts for all agents
                for agent in self.agents.values():
                    try:
                        self.exchange.checkAndAutoCloseShorts(agent.accountId, current_date_str)
                    except Exception as e:
                        logger.error(f"Error auto-closing shorts for agent {agent.agentId}: {e}")
                
                # Advance date
                with self._lock:
                    self.currentDate += timedelta(days=self.dateStep)
                
                # Small sleep between iterations to avoid spinning
                time.sleep(0.1)
        
        except Exception as e:
            logger.error(f"Simulation loop error: {e}")
            with self._lock:
                self._running = False
    
    def _executeIterationForDate(self, simDate: str) -> None:
        """
        Execute one iteration for all agents on all tickers on a specific date.
        
        Args:
            simDate: Simulation date (YYYY-MM-DD)
        """
        for ticker in self.tickers:
            for agent in self.agents.values():
                try:
                    agent.runIteration(self.exchange, ticker, simDate)
                except Exception as e:
                    logger.error(f"Error in agent iteration: {e}")
    
    def getAgentStats(self, agentId: int) -> Dict[str, Any]:
        """
        Get statistics for a specific agent.
        
        Args:
            agentId: Agent ID
        
        Returns:
            Dict with agent metrics, total trades, portfolio value, etc.
        """
        if agentId not in self.agents:
            return {}
        
        agent = self.agents[agentId]
        return {
            'agentId': agentId,
            'isRunning': agent.isRunning(),
            'isPaused': agent.isPaused(),
            'totalTrades': agent.totalTrades,
            'strategyMetrics': agent.performanceTracker.getAllMetrics(),
            'executionLog': agent._executionLog[-20:],  # Last 20 trades
            'strategyWeights': agent.getStrategyWeights()
        }
    
    def getAllAgentStats(self) -> Dict[int, Dict[str, Any]]:
        """
        Get statistics for all agents.
        
        Returns:
            Dict mapping agentId -> stats
        """
        return {agentId: self.getAgentStats(agentId) for agentId in self.agents.keys()}
    
    def getAggregateStats(self) -> Dict[str, Any]:
        """
        Get aggregated statistics across all agents.
        
        Returns:
            Dict with aggregate metrics (total trades, avg win rate, etc.)
        """
        if not self.agents:
            return {'error': 'No agents registered'}
        
        total_trades = sum(agent.totalTrades for agent in self.agents.values())
        all_metrics = {}
        
        for agent in self.agents.values():
            for strategy, metrics in agent.performanceTracker.getAllMetrics().items():
                if strategy not in all_metrics:
                    all_metrics[strategy] = {
                        'totalTrades': 0,
                        'winCount': 0,
                        'totalPnL': 0.0,
                        'profitFactors': []
                    }
                all_metrics[strategy]['totalTrades'] += metrics.get('totalTrades', 0)
                all_metrics[strategy]['winCount'] += metrics.get('winCount', 0)
                all_metrics[strategy]['totalPnL'] += metrics.get('totalPnL', 0.0)
                all_metrics[strategy]['profitFactors'].append(metrics.get('profitFactor', 0.0))
        
        # Calculate averages
        for strategy in all_metrics:
            pfs = all_metrics[strategy]['profitFactors']
            all_metrics[strategy]['avgProfitFactor'] = sum(pfs) / len(pfs) if pfs else 0.0
            del all_metrics[strategy]['profitFactors']
        
        return {
            'agentCount': len(self.agents),
            'currentDate': self.currentDate.strftime('%Y-%m-%d'),
            'totalTrades': total_trades,
            'strategyMetrics': all_metrics
        }
    
    def getCurrentDate(self) -> str:
        """Get current simulation date."""
        with self._lock:
            return self.currentDate.strftime('%Y-%m-%d')
    
    def setCurrentDate(self, date_str: str) -> bool:
        """
        Override current simulation date (for reset/replay).
        
        Args:
            date_str: Date in YYYY-MM-DD format
        
        Returns:
            True if successful, False if invalid date
        """
        try:
            new_date = datetime.strptime(date_str, '%Y-%m-%d')
            
            if new_date < self.startDate or new_date > self.endDate:
                logger.warning(f"Date {date_str} outside valid range")
                return False
            
            with self._lock:
                self.currentDate = new_date
            
            logger.info(f"Simulation date reset to {date_str}")
            return True
        except ValueError:
            logger.error(f"Invalid date format: {date_str}")
            return False
    
    def isRunning(self) -> bool:
        """Check if simulation is running."""
        with self._lock:
            return self._running
    
    def isPaused(self) -> bool:
        """Check if simulation is paused."""
        with self._lock:
            return self._paused

