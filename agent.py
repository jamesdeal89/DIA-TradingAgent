'''
Stock Trading Intelligent Agent with Hyper-Heuristic Delegator.

Multi-strategy agent that selects trading approaches based on past performance (profit factor).
Strategies include: Sentiment Analysis, Mean Reversion, Technical Indicators, Fundamental Analysis, an LSTM, and an RL Deep Q-Network.

Each agent runs in a separate thread and has:
- Per-agent performance tracking.
- Epsilon-greedy strategy selection.
- Query interface for GUI integration.

Shared market: All agents trade on single StockExchange/MySQL instance, but have a unique ID to separate their portfolios.
'''

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import threading
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from strategies import (
    TradingStrategy,
    DeepQLearningStrategy,
    LSTMStrategy,
    SentimentStrategy,
    MeanReversionStrategy,
    TechnicalStrategy,
    FundamentalStrategy,
    StateBuilder
)
from performanceTracker import PerformanceTracker
from strategySelector import StrategySelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    '''
    Stock trading agent with hyper-heuristic strategy delegator.
    
    - Runs in background thread.
    - Maintains per-agent performance tracking.
    - Selects strategies based on past profit factors.
    - Provides query interface for GUI.
    - Analyses all monitored stocks in market at each timestep. Acts on strategy recommendations.
    '''
    
    # Market Identifier Code to tickers.
    # These are the stocks which are monitored per MIC.
    MIC_TICKERS = {
        'XNAS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'],
        'XLON': ['BP', 'SHEL', 'AZN', 'VOD', 'HSBA'],
        'XHKG': ['0700', '0388', '1113', '0005', '0011'],
        'XJPX': ['7203', '6758', '9984', '6861', '8053']
    }

    def __init__(self, agentId: int, accountId: int, mic: str = 'XLON', preferredStrategy: Optional[str] = None, 
                 bannedStrategies: List[str] = [], simDate: str = None, endDate: str = None, decisionPeriod: int = 1):
        '''
        Initialise the agent.
        
        Args:
            agentId: Unique agent identifier
            accountId: Trading account ID (for executing trades)
            mic: Market Identifier Code ('XNAS', 'XLON', 'XHKG', 'XJPX')
            preferredStrategy: Optional strategy to preferentially select
            bannedStrategies: List of strategy names to exclude from selection
            simDate: Initial simulation date (YYYY-MM-DD format)
            endDate: End date for simulation (YYYY-MM-DD format). If None, simulation runs until max news date
            decisionPeriod: Days between trading decisions (default 1 = daily decisions)
        '''
        self.agentId = agentId
        self.accountId = accountId
        self.mic = mic
        self.preferredStrategy = preferredStrategy
        self.bannedStrategies = bannedStrategies
        self.simDate = simDate
        self.endDate = endDate
        self.decisionPeriod = decisionPeriod
        
        # Performance / strategy management
        self.performanceTracker = PerformanceTracker(windowSize=50)
        self.strategies: Dict[str, TradingStrategy] = {}  # Will be populated by _initialiseStrategies()
        self.strategySelector: Optional[StrategySelector] = None
        
        # Trade tracking
        self.totalTrades = 0
        self._executionLog: List[Dict] = []
        # Used in end of experiment summaries as well as to periodically train LSTM and DQN
        self._timestepCounter = 0  
        
        # DeepQL learning: cache entry states for trades so we can compute rewards on close
        # Maps (ticker): (state, action, entryPrice, entryDate) for open DeepQL trades
        self._deepql_state_cache: Dict[str, Dict[str, Any]] = {}
        # Track when we last trained to avoid overtraining
        self._last_training_timestep = 0  
        
        # Lifecycle control
        self._pauseFlag = False
        self._stopFlag = False
        self._lock = threading.Lock()
        
        # Initialise strategies
        self._initialiseStrategies()
        
        logger.info(f"Agent {agentId} initialised (accountId={accountId}, MIC={mic}, simDate={simDate}, endDate={endDate}, decisionPeriod={self.decisionPeriod}d)")
    
    def _initialiseStrategies(self) -> None:
        """Initialise all available strategies, respecting banned/preferred."""
        # Instantiate all trading strategies
        all_strategies = {
            'Sentiment': SentimentStrategy(),
            'MeanReversion': MeanReversionStrategy(),
            'Technical': TechnicalStrategy(),
            'Fundamental': FundamentalStrategy(),
            'DeepQL': DeepQLearningStrategy(),
            'LSTM': LSTMStrategy(),
        }
        
        # Apply bans
        self.strategies = {name: strat for name, strat in all_strategies.items() 
                          if name not in self.bannedStrategies}
        
        # If preferredStrategy is set and not banned, prioritise it
        if self.preferredStrategy and self.preferredStrategy in self.strategies:
            logger.info(f"Agent {self.agentId}: Using preferred strategy {self.preferredStrategy}")
        
        logger.info(f"Agent {self.agentId}: Initialised {len(self.strategies)} strategies: {list(self.strategies.keys())}")
        
        self.strategySelector = StrategySelector(self.strategies, preferredStrategy=self.preferredStrategy, epsilon=0.35, minTradesRequired=2)
    
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
        
        self._timestepCounter += 1
        
        # Train LSTM periodically during simulation (every 10 timesteps)
        if 'LSTM' in self.strategies and self._timestepCounter % 10 == 0:
            try:
                buffer_size = len(self.strategies['LSTM'].replayBuffer)
                if buffer_size >= 16:  # Only train if we have enough experience
                    logger.info(f"Agent {self.agentId}: Intermediate LSTM training at timestep {self._timestepCounter} (buffer size: {buffer_size})")
                    self.strategies['LSTM'].train(batchSize=16, epochs=1)
            except Exception as e:
                logger.error(f"Agent {self.agentId}: Intermediate LSTM training failed: {e}")
        
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
            print(f"[Agent {self.agentId}] Timestep {self._timestepCounter}: Selected strategy {selected_strategy_name}")
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
        
        # Score all HOLD recommendations against actual price movement
        self.performanceTracker.scoreRecommendations(exchange, simDate, self.mic, thresholdPct=2.0)
        
        # Process DeepQL learning: check for closed trades, compute rewards, and train
        self._processDeepQLearning(exchange, simDate)
    
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
                        selected_strategy_name, action_rec, ticker, price_at_rec, confidence_rec, simDate, self.mic
                    )
                    
                    # For DeepQL, cache the state for ALL actions (long, short, hold, sell) for learning
                    if selected_strategy_name == 'DeepQL':
                        state = recommendation.get('state', None)
                        if state is not None:
                            # Map action string to action index: 0=long, 1=short, 2=hold, 3=sell
                            action_idx = 0 if action_rec == 'long' else (1 if action_rec == 'short' else (3 if action_rec == 'sell' else 2))
                            self._deepql_state_cache[ticker] = {
                                'state': state,
                                'action': action_idx,
                                'entryPrice': price_at_rec,
                                'entryDate': simDate,
                                'action_str': action_rec
                            }
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
                self.totalTrades += 1
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
                self.totalTrades += 1
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
                        self.totalTrades += 1
                    else:
                        logger.info(f"Agent {self.agentId}: No long position to sell for {ticker}")
                        return
                else:
                    logger.info(f"Agent {self.agentId}: No portfolio data for {ticker}")
                    return
            
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
    
    # DEEP Q-LEARNING: EXPERIENCE RECORDING & TRAINING
    
    def _processDeepQLearning(self, exchange, simDate: str) -> None:
        """
        Process Deep Q-Learning training: check for closed trades and scored HOLD recommendations.
        Converts outcomes to rewards and records experiences for network learning.
        Called at end of each timestep to enable learning from actual trading outcomes.
        
        For LONG/SHORT: reward = (exit_price - entry_price) / entry_price
        For HOLD: reward = +0.01 if CORRECT, -0.02 if MISSED_LONG or MISSED_SHORT
        
        Args:
            exchange: StockExchange instance for price lookups
            simDate: Current simulation date (YYYY-MM-DD)
        """
        try:
            # Get the DeepQL strategy instance
            deepql_strategy = self.strategies.get('DeepQL', None)
            if not deepql_strategy:
                # DeepQL not enabled, skip learning
                return  
            
            # Check portfolio for closed DeepQL trades (LONG/SHORT)
            portfolio = exchange.checkPortfolio(self.accountId, simDate)
            processed_tickers = set()
            
            # ========== PHASE 1: PROCESS EXECUTED TRADES (LONG/SHORT) ==========
            if portfolio is not None and not portfolio.empty:
                # Iterate through closed positions
                for idx, row in portfolio.iterrows():
                    if row.get('closed') is False:
                        continue  # Position still open
                    
                    ticker = row.get('ticker')
                    strategy_name = row.get('strategyName', '')
                    
                    # Only process DeepQL trades
                    if strategy_name != 'DeepQL' or ticker in processed_tickers:
                        continue
                    
                    processed_tickers.add(ticker)
                    
                    # If we have cached state from when this trade opened
                    if ticker not in self._deepql_state_cache:
                        # No cached state, skip
                        continue  
                    
                    cache_entry = self._deepql_state_cache[ticker]
                    entry_state = cache_entry['state']
                    entry_action = cache_entry['action']
                    entry_price = cache_entry['entryPrice']
                    action_str = cache_entry.get('action_str', '')
                    
                    try:
                        # Get current price to compute reward
                        try:
                            current_price = exchange.getPrice(ticker, self.mic, simDate)
                        except ValueError:
                            # Stock delisted or no data, skip
                            del self._deepql_state_cache[ticker]
                            continue
                        
                        # Calculate reward based on price change and trade type
                        trade_type = row.get('tradeType', '')
                        if trade_type == 'long':
                            # For long: profit if price went up
                            reward = (current_price - entry_price) / entry_price
                        elif trade_type == 'short':
                            # For short: profit if price went down
                            reward = (entry_price - current_price) / entry_price
                        else:
                            continue  # Unknown trade type
                        
                        # Build current state (next state from learning perspective)
                        next_state = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        
                        # Record experience: (state, action, reward, next_state, done=True)
                        import numpy as np
                        entry_state_array = np.array(entry_state) if not isinstance(entry_state, np.ndarray) else entry_state
                        next_state_array = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
                        
                        deepql_strategy.recordExperience(
                            state=entry_state_array,
                            action=entry_action,
                            reward=reward,
                            next_state=next_state_array,
                            done=True
                        )
                        
                        logger.info(f"Agent {self.agentId}: Recorded DeepQL {action_str.upper()} for {ticker}: reward={reward:.4f}")
                        
                        # Clear cache for this ticker (no longer needed)
                        del self._deepql_state_cache[ticker]
                    
                    except Exception as exp_err:
                        logger.error(f"Agent {self.agentId}: Error recording trade experience for {ticker}: {exp_err}")
                        if ticker in self._deepql_state_cache:
                            del self._deepql_state_cache[ticker]
            
            # ========== PHASE 2: PROCESS SCORED HOLD & SELL RECOMMENDATIONS ==========
            # Query recommendations table for DeepQL HOLD and SELL actions with scored outcomes
            try:
                cursor = exchange.connection.cursor()
                query = """
                    SELECT ticker, action, priceAtRecommendation, timestampRecommended, outcome
                    FROM strategy_recommendations
                    WHERE accountId = %s AND strategyName = 'DeepQL' 
                    AND action IN ('hold', 'sell') AND outcome IN ('CORRECT', 'MISSED_LONG', 'MISSED_SHORT')
                    AND DATE(timestampRecommended) <= %s
                    ORDER BY timestampRecommended DESC
                    LIMIT 100
                """
                cursor.execute(query, (self.accountId, simDate))
                scored_recommendations = cursor.fetchall()
                cursor.close()
                
                for rec in scored_recommendations:
                    ticker = rec[0]
                    action = rec[1]
                    price_at_rec = rec[2]
                    timestamp_rec = rec[3]
                    outcome = rec[4]
                    
                    # Skip if we already processed this in trades phase
                    if ticker in processed_tickers:
                        continue
                    
                    # Only process if we have cached state for this action
                    if ticker not in self._deepql_state_cache:
                        continue
                    
                    cache_entry = self._deepql_state_cache[ticker]
                    entry_state = cache_entry['state']
                    entry_action = cache_entry['action']
                    action_str = cache_entry.get('action_str', '')
                    
                    # Only process if cached action matches this recommendation action
                    # 2 = HOLD, 3 = SELL
                    expected_action_idx = 2 if action == 'hold' else (3 if action == 'sell' else -1)
                    if entry_action != expected_action_idx:
                        continue
                    
                    # Skip if outcome not yet determined (don't train on incomplete information)
                    if outcome == 'PENDING':
                        logger.debug(f"Agent {self.agentId}: Skipping HOLD {ticker} - outcome still PENDING")
                        continue
                    
                    try:
                        # Convert outcome to reward (same logic for HOLD and SELL)
                        # Scaled to match LONG/SHORT reward magnitudes (50x larger than before)
                        if outcome == 'CORRECT':
                            # HOLD/SELL was right - price stayed flat or fell (good exit)
                            reward = 0.05  # Increased from 0.01
                        elif outcome == 'MISSED_LONG':
                            # HOLD/SELL was wrong - missed an up move
                            reward = -0.10  # Increased from -0.02
                        elif outcome == 'MISSED_SHORT':
                            # HOLD/SELL was wrong - missed a down move (less common)
                            reward = -0.10  # Increased from -0.02
                        else:
                            # Unknown outcome, skip
                            continue
                        
                        # Build current state at simDate
                        next_state = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        
                        # Record experience
                        import numpy as np
                        entry_state_array = np.array(entry_state) if not isinstance(entry_state, np.ndarray) else entry_state
                        next_state_array = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
                        
                        deepql_strategy.recordExperience(
                            state=entry_state_array,
                            action=entry_action,  # 2 = HOLD or 3 = SELL
                            reward=reward,
                            next_state=next_state_array,
                            done=True
                        )
                        
                        logger.info(f"Agent {self.agentId}: Recorded DeepQL {action.upper()} for {ticker}: outcome={outcome}, reward={reward:.4f}")
                        processed_tickers.add(ticker)
                    
                    except Exception as rec_err:
                        logger.error(f"Agent {self.agentId}: Error processing {action.upper()} for {ticker}: {rec_err}")
            
            except Exception as hold_query_err:
                logger.debug(f"Agent {self.agentId}: Could not query HOLD recommendations: {hold_query_err}")
            
            # ========== PHASE 3: TRAIN NETWORK ==========
            # Train network on accumulated experiences
            # Only train if we've had some recent experiences and enough timesteps have passed
            if len(deepql_strategy.experience_buffer) > 0 and (self._timestepCounter - self._last_training_timestep) >= 5:
                try:
                    batch_size = min(32, len(deepql_strategy.experience_buffer))
                    loss = deepql_strategy.train(batch_size=batch_size)
                    self._last_training_timestep = self._timestepCounter
                    logger.info(f"Agent {self.agentId}: DeepQL training complete. Loss={loss:.6f}, Buffer size={len(deepql_strategy.experience_buffer)}")
                except Exception as train_err:
                    logger.error(f"Agent {self.agentId}: DeepQL training failed: {train_err}")
        
        except Exception as e:
            logger.error(f"Agent {self.agentId}: Error in _processDeepQLearning: {e}")
    
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
        Get recent execution log across all strategies.
        
        Returns:
            List of execution dicts (last N executions, most recent first)
        """
        return self._executionLog[-limit:]
    
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
    
    def setEndDate(self, endDate: str) -> None:
        '''
        Set the end date for the simulation (for repeatable historical backtests).
        
        Args:
            endDate: End date (YYYY-MM-DD). Simulation will stop when currentDate > endDate
        '''
        self.endDate = endDate
        print(f"DEBUG: Agent {self.agentId} endDate updated to {self.endDate}")
    
    def getEndDate(self) -> Optional[str]:
        '''
        Get the simulation end date.
        
        Returns:
            End date (YYYY-MM-DD) or None if not set
        '''
        return self.endDate
    
    def setDecisionPeriod(self, decisionPeriod: int) -> None:
        '''
        Update the decision period for trading (window size in days).
        
        Args:
            decisionPeriod: Number of days between trading decisions (minimum 1)
        '''
        self.decisionPeriod = max(1, decisionPeriod)
        logger.info(f"Agent {self.agentId}: decision period changed to {self.decisionPeriod} days")
    
    def getDecisionPeriod(self) -> int:
        '''
        Get the current decision period.
        
        Returns:
            Current decision period in days
        '''
        return self.decisionPeriod
    
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

