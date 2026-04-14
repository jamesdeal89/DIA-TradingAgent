"""
Stock Trading Intelligent Agent with Hyper-Heuristic Delegator.

Multi-strategy agent that selects trading approaches based on past performance (profit factor).
Strategies include: Sentiment Analysis, Mean Reversion, Technical Indicators, Fundamental Analysis, an LSTM, and an RL Deep Q-Network.

Each agent runs in a separate thread and has:
- Per-agent performance tracking.
- Epsilon-greedy strategy selection.
- Query interface for GUI integration.

Shared market: All agents trade on single StockExchange/MySQL instance, but have a unique ID to separate their portfolios.
"""
from abc import ABC, abstractmethod
import threading
import time
import logging
import numpy as np
from datetime import datetime, timedelta
from strategies import TradingStrategy, DeepQLearningStrategy, LSTMStrategy, SentimentStrategy, MeanReversionStrategy, TechnicalStrategy, FundamentalStrategy, StateBuilder
from performanceTracker import PerformanceTracker
from strategySelector import StrategySelector
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    """
    Stock trading agent with hyper-heuristic strategy delegator.
    
    - Runs in background thread.
    - Maintains per-agent performance tracking.
    - Selects strategies based on past profit factors.
    - Provides query interface for GUI.
    - Analyses all monitored stocks in market at each timestep. Acts on strategy recommendations.
    """
    MIC_TICKERS = {'XNAS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'], 'XLON': ['BP', 'SHEL', 'AZN', 'VOD', 'HSBA'], 'XHKG': ['0700', '0388', '1113', '0005', '0011'], 'XJPX': ['7203', '6758', '9984', '6861', '8053']}

    def __init__(self, agentId, accountId, mic='XLON', preferredStrategy=None, bannedStrategies=[], simDate=None, endDate=None, decisionPeriod=1, selectorEpsilon=0.3, minTradesBeforeMetrics=10):
        """
        Initialise the agent.
        
        Args:
            agentId: Unique agent identifier.
            accountId: Trading account ID (for executing trades via the stock exchange.)
            mic: Market Identifier Code to set what exchange this agent operates in.
            preferredStrategy: Optional - used to test a single specific strategy.
            bannedStrategies: List of strategy names to exclude from selection.
            simDate: Initial simulation date (YYYY-MM-DD format)
            endDate: End date for simulation (YYYY-MM-DD format). At the end of the experiment, a report is printed in the logs.
            decisionPeriod: Days between trading decisions (default 1 = daily decisions.) Can be changed mid-run via setDecisionPeriod().
            selectorEpsilon: Probability of exploring random strategy (0.0-1.0). Default 0.30.
            minTradesBeforeMetrics: Number of trades before using performance metrics. Default 10.
        """
        self.agentId = agentId
        self.accountId = accountId
        self.mic = mic
        self.preferredStrategy = preferredStrategy
        self.bannedStrategies = bannedStrategies
        self.simDate = simDate
        self.endDate = endDate
        self.decisionPeriod = decisionPeriod
        self.selectorEpsilon = selectorEpsilon
        self.minTradesBeforeMetrics = minTradesBeforeMetrics
        self.performanceTracker = PerformanceTracker(windowSize=50)
        self.strategies = {}
        self.strategySelector = None
        self.totalTrades = 0
        self._executionLog = []
        self._timestepCounter = 0
        self._deepql_state_cache = {}
        self._last_training_timestep = 0
        self._pauseFlag = False
        self._stopFlag = False
        self._lock = threading.Lock()
        self._initialiseStrategies()
        logger.info(f'Agent {agentId} initialised (accountId= {accountId}, MIC= {mic}, simDate= {simDate}, endDate= {endDate}, decisionPeriod= {self.decisionPeriod} days)')

    def _initialiseStrategies(self):
        """Initialise all available strategies, considering banned/preferred."""
        all_strategies = {'Sentiment': SentimentStrategy(), 'MeanReversion': MeanReversionStrategy(), 'Technical': TechnicalStrategy(), 'Fundamental': FundamentalStrategy(), 'DeepQL': DeepQLearningStrategy(), 'LSTM': LSTMStrategy()}
        self.strategies = {name: strat for name, strat in all_strategies.items() if name not in self.bannedStrategies}
        if self.preferredStrategy and self.preferredStrategy in self.strategies:
            logger.info(f'Agent {self.agentId}: Using preferred strategy {self.preferredStrategy}')
        logger.info(f'Agent {self.agentId}: Initialised {len(self.strategies)} strategies: {list(self.strategies.keys())}')
        self.strategySelector = StrategySelector(self.strategies, preferredStrategy=self.preferredStrategy, epsilon=self.selectorEpsilon, minTradesRequired=self.minTradesBeforeMetrics)

    def pause(self):
        """Pause agent execution between iterations."""
        with self._lock:
            self._pauseFlag = True
            logger.info(f'Agent {self.agentId} paused')

    def resume(self):
        """Resume agent execution."""
        with self._lock:
            self._pauseFlag = False
            logger.info(f'Agent {self.agentId} resumed')

    def stop(self):
        """Stop agent execution cleanly."""
        with self._lock:
            self._stopFlag = True
            logger.info(f'Agent {self.agentId} stop signal sent')

    def isRunning(self):
        """Check if agent is running (not stopped)."""
        with self._lock:
            return not self._stopFlag

    def isPaused(self):
        """Check if agent is paused."""
        with self._lock:
            return self._pauseFlag

    def runTimestep(self, exchange, simDate=None):
        """
        Execute one timestep of the trading loop.
        
        Called by background thread every decision period. Since date advancement is handled
        by the GUI loop (which advances by decisionPeriod days), this method completes a single analysis and trade cycle.
        
        Args:
            exchange: StockExchange instance for trade execution and data access.
            simDate: Current simulation date as passed by chatGUI, if None, uses agent.simDate.
        """
        if simDate is None:
            simDate = self.simDate
        with self._lock:
            if self._stopFlag:
                logger.info(f'Agent {self.agentId}: stop_flag set, skipping timestep')
                return
            if self._pauseFlag:
                logger.debug(f'Agent {self.agentId}: paused, skipping timestep')
                return
        self._timestepCounter += 1
        if 'LSTM' in self.strategies and self._timestepCounter % 10 == 0:
            try:
                buffer_size = len(self.strategies['LSTM'].replayBuffer)
                if buffer_size >= 16:
                    logger.info(f'Agent {self.agentId}: Intermediate LSTM training at timestep {self._timestepCounter} (buffer size: {buffer_size})')
                    self.strategies['LSTM'].train(batchSize=16, epochs=1)
            except Exception as e:
                logger.error(f'Agent {self.agentId}: Intermediate LSTM training failed: {e}')
        tickers = self.MIC_TICKERS.get(self.mic, [])
        if not tickers:
            logger.warning(f'Agent {self.agentId}: No tickers found for MIC {self.mic}')
            return
        print(f'[Agent {self.agentId}] Decision Day {simDate} (period={self.decisionPeriod}d) on {self.mic}: analyzing {len(tickers)} stocks')
        try:
            metrics = self.performanceTracker.getAllMetrics()
            selected_strategy_name = self.strategySelector.selectStrategy(metrics, self.totalTrades)
            selected_strategy = self.strategies[selected_strategy_name]
            print(f'[Agent {self.agentId}] Timestep {self._timestepCounter}: Selected strategy {selected_strategy_name}')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy selection failed - {e}')
            logger.error(f'Agent {self.agentId}: strategy selection failed: {e}')
            return
        for ticker in tickers:
            try:
                self._analyseAndTrade(exchange, ticker, simDate, selected_strategy_name, selected_strategy)
            except Exception as e:
                print(f'[Agent {self.agentId}] ERROR processing {ticker}: {e}')
                logger.error(f'Agent {self.agentId}: failed to process {ticker}: {e}')
        exchange.logPortfolioSnapshot(self.accountId, self.agentId, simDate)
        self.performanceTracker.scoreRecommendations(exchange, simDate, self.mic, thresholdPct=2.0)
        self._processDeepQLearning(exchange, simDate)

    def _analyseAndTrade(self, exchange, ticker, simDate, selected_strategy_name, selected_strategy):
        """
        Internal method: analyse a single ticker using the given strategy and execute trade if recommended.
        Only called on decision days (every N days based on decisionPeriod).
        
        Args:
            exchange: StockExchange instance
            ticker: Stock ticker symbol
            simDate: Current simulation date (YYYY-MM-DD)
            selected_strategy_name: Name of the strategy to use
            selected_strategy: Strategy instance to use for analysis
        """
        try:
            recommendation = selected_strategy.analyse(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
            action_rec = recommendation.get('action', 'hold')
            confidence_rec = recommendation.get('confidence', 0)
            print(f'[Agent {self.agentId}] {ticker}: {selected_strategy_name} -> {action_rec.upper()} ({confidence_rec * 100:.0f}%)')
            logger.info(f'Agent {self.agentId}: {ticker} {selected_strategy_name} recommended {action_rec} ({confidence_rec:.1%})')
            try:
                price_at_rec = exchange.getPrice(ticker, self.mic, simDate)
                self.performanceTracker.recordRecommendation(selected_strategy_name, action_rec, ticker, price_at_rec, confidence_rec, simDate, self.mic)
                if selected_strategy_name == 'DeepQL':
                    state = recommendation.get('state', None)
                    if state is not None:
                        action_idx = 0 if action_rec == 'long' else 1 if action_rec == 'short' else 3 if action_rec == 'sell' else 2
                        self._deepql_state_cache[ticker] = {'state': state, 'action': action_idx, 'entryPrice': price_at_rec, 'entryDate': simDate, 'action_str': action_rec}
            except Exception as rec_err:
                logger.debug(f'Could not record recommendation for {ticker}: {rec_err}')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy analysis failed for {ticker} - {e}')
            logger.error(f'Agent {self.agentId}: strategy.analyse({ticker}) failed: {e}')
            return
        action = recommendation.get('action', 'hold')
        if action == 'hold':
            logger.debug(f'Agent {self.agentId}: HOLD on {ticker}')
            return
        try:
            target_quantity = recommendation.get('targetQuantity', 0)
            self.executeAction(exchange, ticker, action, target_quantity, simDate, selected_strategy_name, confidence_rec)
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Trade execution failed for {ticker} - {e}')
            logger.error(f'Agent {self.agentId}: executeAction({ticker}) failed: {e}')

    def executeAction(self, exchange, ticker, action, quantity, simDate, strategyName, confidence=0.0):
        """
        Execute a trade action (LONG, SHORT, SELL, or HOLD).
        
        Args:
            exchange: StockExchange instance
            ticker: Stock ticker symbol
            action: Trade action ('long', 'short', 'sell', or 'hold')
            quantity: Number of shares to trade
            simDate: Current simulation date (YYYY-MM-DD)
            strategyName: Name of strategy recommending the trade
            confidence: Confidence score from strategy (0.0-1.0)
        """
        try:
            if action == 'long':
                print(f'[Agent {self.agentId}] EXECUTING LONG: {ticker} x{quantity}')
                result = exchange.placeLong(ticker, self.mic, quantity, self.accountId, simDate, strategyName=strategyName, agentId=self.agentId)
                if result == -1:
                    print(f'[Agent {self.agentId}] TRADE ERROR: long {ticker} failed - insufficient balance or delisted stock')
                    logger.error(f'Agent {self.agentId}: long {ticker} execution failed')
                    return
                print(f'[Agent {self.agentId}] SUCCESS: LONG {ticker} x{quantity} via {strategyName}')
                logger.info(f'Agent {self.agentId}: LONG {ticker} x{quantity} executed via {strategyName}')
                self.totalTrades += 1
            elif action == 'short':
                print(f'[Agent {self.agentId}] EXECUTING SHORT: {ticker} x{quantity}')
                result = exchange.placeShort(ticker, self.mic, quantity, self.accountId, simDate, strategyName=strategyName, agentId=self.agentId)
                if result == -1:
                    print(f'[Agent {self.agentId}] TRADE ERROR: short {ticker} failed - check error log')
                    logger.error(f'Agent {self.agentId}: short {ticker} execution failed')
                    return
                print(f'[Agent {self.agentId}] SUCCESS: SHORT {ticker} x{quantity} via {strategyName}')
                logger.info(f'Agent {self.agentId}: SHORT {ticker} x{quantity} executed via {strategyName}')
                self.totalTrades += 1
            elif action == 'sell':
                portfolio = exchange.checkPortfolio(self.accountId, simDate)
                if portfolio is not None and (not portfolio.empty):
                    long_positions = portfolio[(portfolio['ticker'] == ticker) & (portfolio['tradeType'] == 'long')]
                    if not long_positions.empty:
                        pos = long_positions.iloc[0]
                        sell_qty = int(pos['quantity'])
                        entry_price = float(pos.get('entryPrice', 0))
                        entry_date = pos.get('entryDate', simDate)
                        pos_strategy = pos.get('strategyName', strategyName)
                        print(f'[Agent {self.agentId}] EXECUTING SELL: {ticker} x{sell_qty}')
                        result = exchange.sellLong(self.accountId, ticker, self.mic, sell_qty, simDate, strategyName=pos_strategy, entryPrice=entry_price, entryDate=entry_date)
                        if result == -1:
                            print(f'[Agent {self.agentId}] TRADE ERROR: sell {ticker} failed - delisted stock')
                            logger.error(f'Agent {self.agentId}: sell {ticker} execution failed')
                            return
                        print(f'[Agent {self.agentId}] SUCCESS: SOLD {ticker} x{sell_qty} via {strategyName}')
                        logger.info(f'Agent {self.agentId}: SOLD {ticker} x{sell_qty} executed via {strategyName}')
                        self.totalTrades += 1
                    else:
                        logger.info(f'Agent {self.agentId}: No long position to sell for {ticker}')
                        return
                else:
                    logger.info(f'Agent {self.agentId}: No portfolio data for {ticker}')
                    return
            self._executionLog.append({'strategy': strategyName, 'ticker': ticker, 'action': action, 'quantity': quantity, 'confidence': confidence, 'timestamp': simDate})
        except Exception as e:
            print(f'[Agent {self.agentId}] TRADE ERROR: {action} {ticker} failed - {e}')
            logger.error(f'Agent {self.agentId}: {action} {ticker} execution failed: {e}')
            raise

    def _processDeepQLearning(self, exchange, simDate):
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
            deepql_strategy = self.strategies.get('DeepQL', None)
            if not deepql_strategy:
                return
            portfolio = exchange.checkPortfolio(self.accountId, simDate)
            processed_tickers = set()
            if portfolio is not None and (not portfolio.empty):
                for idx, row in portfolio.iterrows():
                    if row.get('closed') is False:
                        continue
                    ticker = row.get('ticker')
                    strategy_name = row.get('strategyName', '')
                    if strategy_name != 'DeepQL' or ticker in processed_tickers:
                        continue
                    processed_tickers.add(ticker)
                    if ticker not in self._deepql_state_cache:
                        continue
                    cache_entry = self._deepql_state_cache[ticker]
                    entry_state = cache_entry['state']
                    entry_action = cache_entry['action']
                    entry_price = cache_entry['entryPrice']
                    action_str = cache_entry.get('action_str', '')
                    try:
                        try:
                            current_price = exchange.getPrice(ticker, self.mic, simDate)
                        except ValueError:
                            del self._deepql_state_cache[ticker]
                            continue
                        trade_type = row.get('tradeType', '')
                        if trade_type == 'long':
                            reward = (current_price - entry_price) / entry_price
                        elif trade_type == 'short':
                            reward = (entry_price - current_price) / entry_price
                        else:
                            continue
                        next_state = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        import numpy as np
                        entry_state_array = np.array(entry_state) if not isinstance(entry_state, np.ndarray) else entry_state
                        next_state_array = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
                        deepql_strategy.recordExperience(state=entry_state_array, action=entry_action, reward=reward, next_state=next_state_array, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded DeepQL {action_str.upper()} for {ticker}: reward={reward:.4f}')
                        del self._deepql_state_cache[ticker]
                    except Exception as exp_err:
                        logger.error(f'Agent {self.agentId}: Error recording trade experience for {ticker}: {exp_err}')
                        if ticker in self._deepql_state_cache:
                            del self._deepql_state_cache[ticker]
            try:
                cursor = exchange.connection.cursor()
                query = "\n                    SELECT ticker, action, priceAtRecommendation, timestampRecommended, outcome\n                    FROM strategy_recommendations\n                    WHERE accountId = %s AND strategyName = 'DeepQL' \n                    AND action IN ('hold', 'sell') AND outcome IN ('CORRECT', 'MISSED_LONG', 'MISSED_SHORT')\n                    AND DATE(timestampRecommended) <= %s\n                    ORDER BY timestampRecommended DESC\n                    LIMIT 100\n                "
                cursor.execute(query, (self.accountId, simDate))
                scored_recommendations = cursor.fetchall()
                cursor.close()
                for rec in scored_recommendations:
                    ticker = rec[0]
                    action = rec[1]
                    price_at_rec = rec[2]
                    timestamp_rec = rec[3]
                    outcome = rec[4]
                    if ticker in processed_tickers:
                        continue
                    if ticker not in self._deepql_state_cache:
                        continue
                    cache_entry = self._deepql_state_cache[ticker]
                    entry_state = cache_entry['state']
                    entry_action = cache_entry['action']
                    action_str = cache_entry.get('action_str', '')
                    expected_action_idx = 2 if action == 'hold' else 3 if action == 'sell' else -1
                    if entry_action != expected_action_idx:
                        continue
                    if outcome == 'PENDING':
                        logger.debug(f'Agent {self.agentId}: Skipping HOLD {ticker} - outcome still PENDING')
                        continue
                    try:
                        if outcome == 'CORRECT':
                            reward = 0.05
                        elif outcome == 'MISSED_LONG':
                            reward = -0.1
                        elif outcome == 'MISSED_SHORT':
                            reward = -0.1
                        else:
                            continue
                        next_state = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        import numpy as np
                        entry_state_array = np.array(entry_state) if not isinstance(entry_state, np.ndarray) else entry_state
                        next_state_array = np.array(next_state) if not isinstance(next_state, np.ndarray) else next_state
                        deepql_strategy.recordExperience(state=entry_state_array, action=entry_action, reward=reward, next_state=next_state_array, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded DeepQL {action.upper()} for {ticker}: outcome={outcome}, reward={reward:.4f}')
                        processed_tickers.add(ticker)
                    except Exception as rec_err:
                        logger.error(f'Agent {self.agentId}: Error processing {action.upper()} for {ticker}: {rec_err}')
            except Exception as hold_query_err:
                logger.debug(f'Agent {self.agentId}: Could not query HOLD recommendations: {hold_query_err}')
            if len(deepql_strategy.experience_buffer) > 0 and self._timestepCounter - self._last_training_timestep >= 5:
                try:
                    batch_size = min(32, len(deepql_strategy.experience_buffer))
                    loss = deepql_strategy.train(batch_size=batch_size)
                    self._last_training_timestep = self._timestepCounter
                    logger.info(f'Agent {self.agentId}: DeepQL training complete. Loss={loss:.6f}, Buffer size={len(deepql_strategy.experience_buffer)}')
                except Exception as train_err:
                    logger.error(f'Agent {self.agentId}: DeepQL training failed: {train_err}')
        except Exception as e:
            logger.error(f'Agent {self.agentId}: Error in _processDeepQLearning: {e}')

    def getPerformanceMetrics(self, strategyName=None):
        """
        Get performance metrics for one or all strategies.
        
        Args:
            strategyName: If specified, get metrics for this strategy, otherwise all strategies
        
        Returns:
            Single strategy metrics or Dict[strategyName -> metrics]
        """
        if strategyName:
            return self.performanceTracker.getMetrics(strategyName)
        else:
            return self.performanceTracker.getAllMetrics()

    def getRecentTrades(self, limit=10):
        """
        Get recent execution log across all strategies.
        
        Returns:
            List of execution dicts (last N executions, most recent first)
        """
        return self._executionLog[-limit:]

    def getStrategyWeights(self):
        """Get current epsilon-greedy weights for each strategy."""
        if self.strategySelector:
            return self.strategySelector.getWeights()
        return {}

    def setSimDate(self, simDate):
        """
        Update the current simulation date for the agent.
        
        Args:
            simDate: New simulation date (YYYY-MM-DD)
        """
        self.simDate = simDate
        print(f'DEBUG: Agent {self.agentId} simDate updated to {self.simDate}')

    def getSimDate(self):
        """
        Get the current simulation date for the agent.
        
        Returns:
            Current simulation date (YYYY-MM-DD)
        """
        return self.simDate

    def setEndDate(self, endDate):
        """
        Set the end date for the simulation (for repeatable historical backtests).
        
        Args:
            endDate: End date (YYYY-MM-DD). Simulation will stop when currentDate > endDate
        """
        self.endDate = endDate
        print(f'DEBUG: Agent {self.agentId} endDate updated to {self.endDate}')

    def getEndDate(self):
        """
        Get the simulation end date.
        
        Returns:
            End date (YYYY-MM-DD) or None if not set
        """
        return self.endDate

    def setDecisionPeriod(self, decisionPeriod):
        """
        Update the decision period for trading (window size in days).
        
        Args:
            decisionPeriod: Number of days between trading decisions (minimum 1)
        """
        self.decisionPeriod = max(1, decisionPeriod)
        logger.info(f'Agent {self.agentId}: decision period changed to {self.decisionPeriod} days')

    def getDecisionPeriod(self):
        """
        Get the current decision period.
        
        Returns:
            Current decision period in days
        """
        return self.decisionPeriod

    def getTimestepCounter(self):
        """
        Get the total number of timesteps executed.
        
        Returns:
            Total timesteps processed since agent creation
        """
        return self._timestepCounter

    def getTotalTrades(self):
        """
        Get the total number of trades placed.
        
        Returns:
            Total trades executed
        """
        return self.totalTrades

    def getExecutionLog(self):
        """
        Get the agent's execution log.
        
        Returns:
            List of execution log entries
        """
        return self._executionLog

    def getStrategies(self):
        """
        Get the available strategies.
        
        Returns:
            Dictionary mapping strategy names to TradingStrategy instances
        """
        return self.strategies

    def getPerformanceTracker(self):
        """
        Get the performance tracker instance.
        
        Returns:
            PerformanceTracker instance
        """
        return self.performanceTracker

    def runIteration(self, exchange, ticker, simDate=None):
        """
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
        """
        if simDate is None:
            simDate = self.simDate
        with self._lock:
            if self._stopFlag:
                logger.info(f'Agent {self.agentId}: stop_flag set, exiting iteration')
                return
            if self._pauseFlag:
                logger.debug(f'Agent {self.agentId}: paused, skipping iteration')
                return
        if not self.strategies:
            self._initialiseStrategies()
        try:
            metrics = self.performanceTracker.getAllMetrics()
            selected_strategy_name = self.strategySelector.selectStrategy(metrics, self.totalTrades)
            selected_strategy = self.strategies[selected_strategy_name]
            print(f'[Agent {self.agentId}] Selected strategy: {selected_strategy_name}')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy selection failed - {e}')
            logger.error(f'Agent {self.agentId}: strategy selection failed: {e}')
            return
        try:
            recommendation = selected_strategy.analyse(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
            action_rec = recommendation.get('action', 'hold')
            confidence_rec = recommendation.get('confidence', 0)
            logger.info(f'Agent {self.agentId}: {selected_strategy_name} recommended {action_rec} ({confidence_rec:.1%} confidence)')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy analysis failed - {e}')
            logger.error(f'Agent {self.agentId}: strategy.analyse() failed: {e}')
            return
        action = recommendation.get('action', 'hold')
        if action == 'hold':
            print(f'[Agent {self.agentId}] HOLD on {ticker}')
            logger.info(f'Agent {self.agentId}: HOLD (no action taken)')
            return
        try:
            quantity = recommendation.get('targetQuantity', 1)
            if action == 'long':
                print(f'[Agent {self.agentId}] EXECUTING LONG: {ticker} x{quantity}')
                exchange.placeLong(ticker, self.mic, quantity, self.accountId, simDate, strategyName=selected_strategy_name, agentId=self.agentId)
                print(f'[Agent {self.agentId}] SUCCESS: LONG {ticker} x{quantity} via {selected_strategy_name}')
                logger.info(f'Agent {self.agentId}: LONG {ticker} x{quantity} executed via {selected_strategy_name}')
            elif action == 'short':
                print(f'[Agent {self.agentId}] EXECUTING SHORT: {ticker} x{quantity}')
                exchange.placeShort(ticker, self.mic, quantity, self.accountId, simDate, strategyName=selected_strategy_name, agentId=self.agentId)
                print(f'[Agent {self.agentId}] SUCCESS: SHORT {ticker} x{quantity} via {selected_strategy_name}')
                logger.info(f'Agent {self.agentId}: SHORT {ticker} x{quantity} executed via {selected_strategy_name}')
            self.totalTrades += 1
            self._executionLog.append({'strategy': selected_strategy_name, 'ticker': ticker, 'action': action, 'quantity': quantity, 'confidence': recommendation.get('confidence', 0), 'timestamp': simDate})
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Trade execution failed - {e}')
            logger.error(f'Agent {self.agentId}: trade execution failed: {e}')

class AgentManager:
    """
    Manages multiple trading agents running in parallel.
    Each agent runs in its own background thread.
    Manages simulation loop (date iteration, ticker rotation).
    Thread-safe: all shared state protected by locks.
    """

    def __init__(self, exchange, tickers, mics, date_range, date_step_days=1):
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
        self.startDate = datetime.strptime(date_range[0], '%Y-%m-%d')
        self.endDate = datetime.strptime(date_range[1], '%Y-%m-%d')
        self.currentDate = self.startDate
        self.agents = {}
        self.agentThreads = {}
        self._lock = threading.RLock()
        self._running = False
        self._paused = False
        self.mainLoopThread = None
        logger.info(f'AgentManager: Initialised with {len(tickers)} tickers, date range {date_range[0]} to {date_range[1]}')

    def addAgent(self, agentId, accountId, mic, preferredStrategy=None, bannedStrategies=None):
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
                logger.warning(f'Agent {agentId} already exists, skipping')
                return self.agents[agentId]
            agent = Agent(agentId=agentId, accountId=accountId, mic=mic, preferredStrategy=preferredStrategy, bannedStrategies=bannedStrategies)
            self.agents[agentId] = agent
            logger.info(f'AgentManager: Added agent {agentId} (accountId={accountId}, MIC={mic})')
            return agent

    def removeAgent(self, agentId):
        """
        Remove an agent from the manager.
        
        Args:
            agentId: Agent ID to remove
        
        Returns:
            True if successful, False if not found
        """
        with self._lock:
            if agentId not in self.agents:
                logger.warning(f'Agent {agentId} not found')
                return False
            if agentId in self.agentThreads and self.agentThreads[agentId].is_alive():
                self.agents[agentId].stop()
                self.agentThreads[agentId].join(timeout=5.0)
            del self.agents[agentId]
            if agentId in self.agentThreads:
                del self.agentThreads[agentId]
            logger.info(f'AgentManager: Removed agent {agentId}')
            return True

    def start(self):
        """Start the simulation loop in a background thread."""
        with self._lock:
            if self._running:
                logger.warning('AgentManager already running')
                return
            self._running = True
            self._paused = False
        self.mainLoopThread = threading.Thread(target=self._simulationLoop, daemon=False)
        self.mainLoopThread.start()
        logger.info('AgentManager: Simulation started')

    def stop(self):
        """Stop all agents and the simulation loop."""
        with self._lock:
            self._running = False
        for agent in self.agents.values():
            agent.stop()
        for thread in self.agentThreads.values():
            thread.join(timeout=5.0)
        if self.mainLoopThread:
            self.mainLoopThread.join(timeout=5.0)
        logger.info('AgentManager: All agents and simulation stopped')

    def pause(self):
        """Pause all agents."""
        with self._lock:
            self._paused = True
        for agent in self.agents.values():
            agent.pause()
        logger.info('AgentManager: All agents paused')

    def resume(self):
        """Resume all agents."""
        with self._lock:
            self._paused = False
        for agent in self.agents.values():
            agent.resume()
        logger.info('AgentManager: All agents resumed')

    def _simulationLoop(self):
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
                        logger.debug(f'Simulation paused at {current_date_str}')
                        time.sleep(0.5)
                        continue
                    if self.currentDate > self.endDate:
                        logger.info('Simulation reached end date')
                        self._running = False
                        break
                    current_date_str = self.currentDate.strftime('%Y-%m-%d')
                self._executeIterationForDate(current_date_str)
                for agent in self.agents.values():
                    try:
                        self.exchange.checkAndAutoCloseShorts(agent.accountId, current_date_str)
                    except Exception as e:
                        logger.error(f'Error auto-closing shorts for agent {agent.agentId}: {e}')
                with self._lock:
                    self.currentDate += timedelta(days=self.dateStep)
                time.sleep(0.1)
        except Exception as e:
            logger.error(f'Simulation loop error: {e}')
            with self._lock:
                self._running = False

    def _executeIterationForDate(self, simDate):
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
                    logger.error(f'Error in agent iteration: {e}')

    def getAgentStats(self, agentId):
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
        return {'agentId': agentId, 'isRunning': agent.isRunning(), 'isPaused': agent.isPaused(), 'totalTrades': agent.totalTrades, 'strategyMetrics': agent.performanceTracker.getAllMetrics(), 'executionLog': agent._executionLog[-20:], 'strategyWeights': agent.getStrategyWeights()}

    def getAllAgentStats(self):
        """
        Get statistics for all agents.
        
        Returns:
            Dict mapping agentId -> stats
        """
        return {agentId: self.getAgentStats(agentId) for agentId in self.agents.keys()}

    def getAggregateStats(self):
        """
        Get aggregated statistics across all agents.
        
        Returns:
            Dict with aggregate metrics (total trades, avg win rate, etc.)
        """
        if not self.agents:
            return {'error': 'No agents registered'}
        total_trades = sum((agent.totalTrades for agent in self.agents.values()))
        all_metrics = {}
        for agent in self.agents.values():
            for strategy, metrics in agent.performanceTracker.getAllMetrics().items():
                if strategy not in all_metrics:
                    all_metrics[strategy] = {'totalTrades': 0, 'winCount': 0, 'totalPnL': 0.0, 'profitFactors': []}
                all_metrics[strategy]['totalTrades'] += metrics.get('totalTrades', 0)
                all_metrics[strategy]['winCount'] += metrics.get('winCount', 0)
                all_metrics[strategy]['totalPnL'] += metrics.get('totalPnL', 0.0)
                all_metrics[strategy]['profitFactors'].append(metrics.get('profitFactor', 0.0))
        for strategy in all_metrics:
            pfs = all_metrics[strategy]['profitFactors']
            all_metrics[strategy]['avgProfitFactor'] = sum(pfs) / len(pfs) if pfs else 0.0
            del all_metrics[strategy]['profitFactors']
        return {'agentCount': len(self.agents), 'currentDate': self.currentDate.strftime('%Y-%m-%d'), 'totalTrades': total_trades, 'strategyMetrics': all_metrics}

    def getCurrentDate(self):
        """Get current simulation date."""
        with self._lock:
            return self.currentDate.strftime('%Y-%m-%d')

    def setCurrentDate(self, date_str):
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
                logger.warning(f'Date {date_str} outside valid range')
                return False
            with self._lock:
                self.currentDate = new_date
            logger.info(f'Simulation date reset to {date_str}')
            return True
        except ValueError:
            logger.error(f'Invalid date format: {date_str}')
            return False

    def isRunning(self):
        """Check if simulation is running."""
        with self._lock:
            return self._running

    def isPaused(self):
        """Check if simulation is paused."""
        with self._lock:
            return self._paused