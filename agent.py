"""
Stock Trading Intelligent Agent with Hyper-Heuristic Delegator.

Each agent runs in a separate thread and has:
- Per-agent performance tracking.
- Epsilon-greedy strategy selection.
- Query interface for GUI integration.
"""
import threading
import logging
import numpy as np
from strategies import DeepQLearningStrategy, LSTMStrategy, SentimentStrategy, MeanReversionStrategy, TechnicalStrategy, FundamentalStrategy, StateBuilder
from performanceTracker import PerformanceTracker
from strategySelector import StrategySelector
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Agent:
    """
    Stock trading agent with hyper-heuristic strategy delegator.
    
    Runs in background thread.
    Maintains per-agent performance tracking.
    Selects strategies based on past profit factors.
    Provides query interface for GUI.
    Analyses all monitored stocks in market at each timestep. Acts on strategy recommendations.
    """
    MIC_TICKERS = {'XNAS': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'AMZN'], 
                   'XLON': ['BP', 'SHEL', 'AZN', 'VOD', 'HSBA'], 
                   'XHKG': ['0700', '0388', '1113', '0005', '0011'], 
                   'XJPX': ['7203', '6758', '9984', '6861', '8053']}

    def __init__(self, agentId, accountId, mic='XLON', preferredStrategy=None, bannedStrategies=[], simDate=None, endDate=None, decisionPeriod=1, selectorEpsilon=0.3, minTradesBeforeMetrics=10):
        """
        Initialise the agent.
        
        agentId: Unique agent identifier.
        accountId: Trading account ID (for executing trades via the stock exchange.)
        mic: Market Identifier Code to set what exchange this agent operates in.
        preferredStrategy: Optional - used to test a single specific strategy.
        bannedStrategies: List of strategy names to exclude from selection.
        simDate: Initial simulation date (YYYY-MM-DD str format)
        endDate: End date for simulation (YYYY-MM-DD str format). At the end of the experiment, a report is printed in the logs.
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
        self.executionLog = []
        self.timestepCounter = 0
        self.deepQlStateCache = {}
        self.lstmStateCache = {}
        self.lastTrainingTimestep = 0
        self.pauseFlag = False
        self.lock = threading.Lock()
        self.initialiseStrategies()
        logger.info(f'Agent {agentId} initialised (accountId= {accountId}, MIC= {mic}, simDate= {simDate}, endDate= {endDate}, decisionPeriod= {self.decisionPeriod} days)')

    def initialiseStrategies(self):
        """Initialise all available strategies, considering banned/preferred."""
        all_strategies = {'Sentiment': SentimentStrategy(), 'MeanReversion': MeanReversionStrategy(), 'Technical': TechnicalStrategy(), 'Fundamental': FundamentalStrategy(), 'DeepQL': DeepQLearningStrategy(), 'LSTM': LSTMStrategy()}
        self.strategies = {name: strat for name, strat in all_strategies.items() if name not in self.bannedStrategies}
        if self.preferredStrategy and self.preferredStrategy in self.strategies:
            logger.info(f'Agent {self.agentId}: Using preferred strategy {self.preferredStrategy}')
        logger.info(f'Agent {self.agentId}: Initialised {len(self.strategies)} strategies: {list(self.strategies.keys())}')
        self.strategySelector = StrategySelector(self.strategies, preferredStrategy=self.preferredStrategy, epsilon=self.selectorEpsilon, minTradesRequired=self.minTradesBeforeMetrics)

    def pause(self):
        """Pause agent execution between iterations."""
        with self.lock:
            self.pauseFlag = True
            logger.info(f'Agent {self.agentId} paused')

    def resume(self):
        """Resume agent execution."""
        with self.lock:
            self.pauseFlag = False
            logger.info(f'Agent {self.agentId} resumed')

    def isPaused(self):
        """Check if agent is paused."""
        with self.lock:
            return self.pauseFlag

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
        with self.lock:
            if self.pauseFlag:
                logger.debug(f'Agent {self.agentId}: paused, skipping timestep')
                return
        self.timestepCounter += 1
        tickers = self.MIC_TICKERS.get(self.mic, [])
        if not tickers:
            logger.warning(f'Agent {self.agentId}: No tickers found for MIC {self.mic}')
            return
        print(f'[Agent {self.agentId}] Decision Day {simDate} (period={self.decisionPeriod}d) on {self.mic}: analysing {len(tickers)} stocks')
        try:
            metrics = self.performanceTracker.getAllMetrics()
            selectedStrategyName = self.strategySelector.selectStrategy(metrics, len(self.executionLog))
            selectedStrategy = self.strategies[selectedStrategyName]
            print(f'[Agent {self.agentId}] Timestep {self.timestepCounter}: Selected strategy {selectedStrategyName}')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy selection failed - {e}')
            logger.error(f'Agent {self.agentId}: strategy selection failed: {e}')
            return
        for ticker in tickers:
            try:
                self.analyseAndTrade(exchange, ticker, simDate, selectedStrategyName, selectedStrategy)
            except Exception as e:
                print(f'[Agent {self.agentId}] ERROR processing {ticker}: {e}')
                logger.error(f'Agent {self.agentId}: failed to process {ticker}: {e}')
        exchange.logPortfolioSnapshot(self.accountId, self.agentId, simDate)
        self.performanceTracker.scoreRecommendations(exchange, simDate, self.mic, thresholdPct=2.0)
        self.processDeepQlLearning(exchange, simDate)
        self.processLstm(exchange, simDate)

    def analyseAndTrade(self, exchange, ticker, simDate, selectedStrategyName, selectedStrategy):
        """
        Internal method: analyse a single ticker using the given strategy and execute trade if recommended.
        Only called on decision days (every N days based on decisionPeriod).
        
        Args:
            exchange: StockExchange instance
            ticker: Stock ticker symbol
            simDate: Current simulation date (YYYY-MM-DD)
            selectedStrategyName: Name of the strategy to use
            selectedStrategy: Strategy instance to use for analysis
        """
        try:
            recommendation = selectedStrategy.analyse(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
            actionRec = recommendation.get('action', 'hold')
            confidenceRec = recommendation.get('confidence', 0)
            print(f'[Agent {self.agentId}] {ticker}: {selectedStrategyName} -> {actionRec.upper()} ({confidenceRec * 100:.0f}%)')
            logger.info(f'Agent {self.agentId}: {ticker} {selectedStrategyName} recommended {actionRec} ({confidenceRec:.1%})')
            try:
                priceAtRec = exchange.getPrice(ticker, self.mic, simDate)
                self.performanceTracker.recordRecommendation(selectedStrategyName, actionRec, ticker, priceAtRec, confidenceRec, simDate, self.mic)
                if selectedStrategyName == 'DeepQL':
                    state = recommendation.get('state', None)
                    if state is not None:
                        actionIdx = 0 if actionRec == 'long' else 1 if actionRec == 'short' else 3 if actionRec == 'sell' else 2
                        self.deepQlStateCache[ticker] = {'state': state, 'action': actionIdx, 'entryPrice': priceAtRec, 'entryDate': simDate, 'actionStr': actionRec}
                elif selectedStrategyName == 'LSTM':
                    state = recommendation.get('state', None)
                    if state is not None:
                        actionIdx = 0 if actionRec == 'long' else 1 if actionRec == 'short' else 3 if actionRec == 'sell' else 2
                        self.lstmStateCache[ticker] = {'state': state, 'action': actionIdx, 'entryPrice': priceAtRec, 'entryDate': simDate, 'actionStr': actionRec}
            except Exception as recErr:
                logger.debug(f'Could not record recommendation for {ticker}: {recErr}')
        except Exception as e:
            print(f'[Agent {self.agentId}] ERROR: Strategy analysis failed for {ticker} - {e}')
            logger.error(f'Agent {self.agentId}: strategy.analyse({ticker}) failed: {e}')
            return
        action = recommendation.get('action', 'hold')
        if action == 'hold':
            logger.debug(f'Agent {self.agentId}: HOLD on {ticker}')
            return
        try:
            targetQuantity = recommendation.get('targetQuantity', 0)
            self.executeAction(exchange, ticker, action, targetQuantity, simDate, selectedStrategyName, confidenceRec)
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
            elif action == 'short':
                print(f'[Agent {self.agentId}] EXECUTING SHORT: {ticker} x{quantity}')
                result = exchange.placeShort(ticker, self.mic, quantity, self.accountId, simDate, strategyName=strategyName, agentId=self.agentId)
                if result == -1:
                    print(f'[Agent {self.agentId}] TRADE ERROR: short {ticker} failed - check error log')
                    logger.error(f'Agent {self.agentId}: short {ticker} execution failed')
                    return
                print(f'[Agent {self.agentId}] SUCCESS: SHORT {ticker} x{quantity} via {strategyName}')
                logger.info(f'Agent {self.agentId}: SHORT {ticker} x{quantity} executed via {strategyName}')
            elif action == 'sell':
                portfolio = exchange.checkPortfolio(self.accountId, simDate)
                if portfolio is not None and (not portfolio.empty):
                    longPositions = portfolio[(portfolio['ticker'] == ticker) & (portfolio['tradeType'] == 'long')]
                    if not longPositions.empty:
                        pos = longPositions.iloc[0]
                        sellQty = int(pos['quantity'])
                        print(f'[Agent {self.agentId}] EXECUTING SELL: {ticker} x{sellQty}')
                        result = exchange.sellLong(self.accountId, ticker, self.mic, sellQty, simDate)
                        if result == -1:
                            print(f'[Agent {self.agentId}] TRADE ERROR: sell {ticker} failed - delisted stock')
                            logger.error(f'Agent {self.agentId}: sell {ticker} execution failed')
                            return
                        print(f'[Agent {self.agentId}] SUCCESS: SOLD {ticker} x{sellQty} via {strategyName}')
                        logger.info(f'Agent {self.agentId}: SOLD {ticker} x{sellQty} executed via {strategyName}')
                    else:
                        logger.info(f'Agent {self.agentId}: No long position to sell for {ticker}')
                        return
                else:
                    logger.info(f'Agent {self.agentId}: No portfolio data for {ticker}')
                    return
            self.executionLog.append({'strategy': strategyName, 'ticker': ticker, 'action': action, 'quantity': quantity, 'confidence': confidence, 'timestamp': simDate})
        except Exception as e:
            print(f'[Agent {self.agentId}] TRADE ERROR: {action} {ticker} failed - {e}')
            logger.error(f'Agent {self.agentId}: {action} {ticker} execution failed: {e}')
            raise

    def processDeepQlLearning(self, exchange, simDate):
        """
        Process Deep Q-Learning training: check for closed trades and scored HOLD recommendations.
        Converts outcomes to rewards and records experiences for network learning.
        Called at end of each timestep to enable learning from actual trading outcomes.
        
        exchange: StockExchange instance for price lookups.
        simDate: Current simulation date.
        """
        try:
            deepQlStrategy = self.strategies.get('DeepQL', None)
            if not deepQlStrategy:
                return
            portfolio = exchange.checkPortfolio(self.accountId, simDate)
            processedTickers = set()
            if portfolio is not None and (not portfolio.empty):
                for idx, row in portfolio.iterrows():
                    if row.get('closed') is False:
                        continue
                    ticker = row.get('ticker')
                    strategyName = row.get('strategyName', '')
                    if strategyName != 'DeepQL' or ticker in processedTickers:
                        continue
                    processedTickers.add(ticker)
                    if ticker not in self.deepQlStateCache:
                        continue
                    cacheEntry = self.deepQlStateCache[ticker]
                    entryState = cacheEntry['state']
                    entryAction = cacheEntry['action']
                    entryPrice = cacheEntry['entryPrice']
                    actionStr = cacheEntry.get('actionStr', '')
                    try:
                        try:
                            currentPrice = exchange.getPrice(ticker, self.mic, simDate)
                        except ValueError:
                            del self.deepQlStateCache[ticker]
                            continue
                        tradeType = row.get('tradeType', '')
                        if tradeType == 'long' or tradeType == 'sell':
                            reward = (currentPrice - entryPrice) / entryPrice
                        elif tradeType == 'short':
                            reward = (entryPrice - currentPrice) / entryPrice
                        else:
                            continue
                        nextState = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        entryStateArray = np.array(entryState) if not isinstance(entryState, np.ndarray) else entryState
                        nextStateArray = np.array(nextState) if not isinstance(nextState, np.ndarray) else nextState
                        deepQlStrategy.recordExperience(state=entryStateArray, action=entryAction, reward=reward, nextState=nextStateArray, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded DeepQL {actionStr.upper()} for {ticker}: reward={reward:.4f}')
                        del self.deepQlStateCache[ticker]
                    except Exception as expErr:
                        logger.error(f'Agent {self.agentId}: Error recording trade experience for {ticker}: {expErr}')
                        if ticker in self.deepQlStateCache:
                            del self.deepQlStateCache[ticker]
            try:
                scoredRecommendations = self.performanceTracker.getScoredRecommendations('DeepQL', simDate=simDate, actions=['hold'])
                for rec in scoredRecommendations:
                    ticker = rec.get('ticker')
                    action = rec.get('action')
                    outcome = rec.get('outcome')
                    if ticker in processedTickers:
                        continue
                    if ticker not in self.deepQlStateCache:
                        continue
                    cacheEntry = self.deepQlStateCache[ticker]
                    entryState = cacheEntry['state']
                    entryAction = cacheEntry['action']
                    expectedActionIdx = 2 if action == 'hold' else -1
                    if entryAction != expectedActionIdx:
                        continue
                    try:
                        if outcome == 'CORRECT':
                            reward = 0.05
                        elif outcome in ('MISSED_LONG', 'MISSED_SHORT'):
                            reward = -0.1
                        else:
                            continue
                        nextState = StateBuilder.buildState(ticker, self.mic, simDate, exchange, analysisPeriod=self.decisionPeriod)
                        entryStateArray = np.array(entryState) if not isinstance(entryState, np.ndarray) else entryState
                        nextStateArray = np.array(nextState) if not isinstance(nextState, np.ndarray) else nextState
                        deepQlStrategy.recordExperience(state=entryStateArray, action=entryAction, reward=reward, nextState=nextStateArray, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded DeepQL {action.upper()} for {ticker}: outcome={outcome}, reward={reward:.4f}')
                        processedTickers.add(ticker)
                    except Exception as recErr:
                        logger.error(f'Agent {self.agentId}: Error processing {action.upper()} for {ticker}: {recErr}')
            except Exception as holdQueryErr:
                logger.debug(f'Agent {self.agentId}: Could not query scored recommendations: {holdQueryErr}')
            if len(deepQlStrategy.experience_buffer) > 0 and self.timestepCounter - self.lastTrainingTimestep >= 5:
                try:
                    batchSize = min(32, len(deepQlStrategy.experience_buffer))
                    loss = deepQlStrategy.train(batchSize=batchSize)
                    self.lastTrainingTimestep = self.timestepCounter
                    logger.info(f'Agent {self.agentId}: DeepQL training complete. Loss={loss:.6f}, Buffer size={len(deepQlStrategy.experience_buffer)}')
                except Exception as trainErr:
                    logger.error(f'Agent {self.agentId}: DeepQL training failed: {trainErr}')
        except Exception as e:
            logger.error(f'Agent {self.agentId}: Error in processDeepQlLearning: {e}')

    def processLstm(self, exchange, simDate):
        """
        Process LSTM training: check for closed trades and record experiences.
        Called at end of each timestep to enable learning from actual trading outcomes.
        
        exchange: StockExchange instance for price lookups
        simDate: Current simulation date
        """
        try:
            lstmStrategy = self.strategies.get('LSTM', None)
            if not lstmStrategy:
                return
            portfolio = exchange.checkPortfolio(self.accountId, simDate)
            processedTickers = set()
            if portfolio is not None and (not portfolio.empty):
                for idx, row in portfolio.iterrows():
                    if row.get('closed') is False:
                        continue
                    ticker = row.get('ticker')
                    strategyName = row.get('strategyName', '')
                    if strategyName != 'LSTM' or ticker in processedTickers:
                        continue
                    processedTickers.add(ticker)
                    if ticker not in self.lstmStateCache:
                        continue
                    cacheEntry = self.lstmStateCache[ticker]
                    entryState = cacheEntry['state']
                    entryAction = cacheEntry['action']
                    entryPrice = cacheEntry['entryPrice']
                    actionStr = cacheEntry.get('actionStr', '')
                    try:
                        try:
                            currentPrice = exchange.getPrice(ticker, self.mic, simDate)
                        except ValueError:
                            del self.lstmStateCache[ticker]
                            continue
                        tradeType = row.get('tradeType', '')
                        if tradeType == 'long' or tradeType == 'sell':
                            reward = (currentPrice - entryPrice) / entryPrice
                        elif tradeType == 'short':
                            reward = (entryPrice - currentPrice) / entryPrice
                        else:
                            continue
                        nextState = lstmStrategy._getSequenceFeatures(ticker, self.mic, simDate, exchange, lookbackDays=20)
                        if nextState is None:
                            nextState = entryState
                        entryStateArray = np.array(entryState) if not isinstance(entryState, np.ndarray) else entryState
                        nextStateArray = np.array(nextState) if not isinstance(nextState, np.ndarray) else nextState
                        lstmStrategy.recordExperience(state=entryStateArray, action=entryAction, reward=reward, nextState=nextStateArray, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded LSTM {actionStr.upper()} for {ticker}: reward={reward:.4f}')
                        del self.lstmStateCache[ticker]
                    except Exception as expErr:
                        logger.error(f'Agent {self.agentId}: Error recording LSTM trade experience for {ticker}: {expErr}')
                        if ticker in self.lstmStateCache:
                            del self.lstmStateCache[ticker]
            try:
                scoredRecommendations = self.performanceTracker.getScoredRecommendations('LSTM', simDate=simDate, actions=['hold'])
                for rec in scoredRecommendations:
                    ticker = rec.get('ticker')
                    action = rec.get('action')
                    outcome = rec.get('outcome')
                    if ticker in processedTickers:
                        continue
                    if ticker not in self.lstmStateCache:
                        continue
                    cacheEntry = self.lstmStateCache[ticker]
                    entryState = cacheEntry['state']
                    entryAction = cacheEntry['action']
                    expectedActionIdx = 2 if action == 'hold' else -1
                    if entryAction != expectedActionIdx:
                        continue
                    try:
                        if outcome == 'CORRECT':
                            reward = 0.05
                        elif outcome in ('MISSED_LONG', 'MISSED_SHORT'):
                            reward = -0.1
                        else:
                            continue
                        nextState = lstmStrategy._getSequenceFeatures(ticker, self.mic, simDate, exchange, lookbackDays=20)
                        if nextState is None:
                            nextState = entryState
                        entryStateArray = np.array(entryState) if not isinstance(entryState, np.ndarray) else entryState
                        nextStateArray = np.array(nextState) if not isinstance(nextState, np.ndarray) else nextState
                        lstmStrategy.recordExperience(state=entryStateArray, action=entryAction, reward=reward, nextState=nextStateArray, done=True)
                        logger.info(f'Agent {self.agentId}: Recorded LSTM {action.upper()} for {ticker}: outcome={outcome}, reward={reward:.4f}')
                        processedTickers.add(ticker)
                        del self.lstmStateCache[ticker]
                    except Exception as recErr:
                        logger.error(f'Agent {self.agentId}: Error processing LSTM {action.upper()} for {ticker}: {recErr}')
            except Exception as holdQueryErr:
                logger.debug(f'Agent {self.agentId}: Could not query scored LSTM recommendations: {holdQueryErr}')
            if len(lstmStrategy.replayBuffer) > 0 and self.timestepCounter - self.lastTrainingTimestep >= 5:
                try:
                    batchSize = min(32, len(lstmStrategy.replayBuffer))
                    loss = lstmStrategy.train(batchSize=batchSize, epochs=1)
                    self.lastTrainingTimestep = self.timestepCounter
                    logger.info(f'Agent {self.agentId}: LSTM training complete. Loss={loss:.6f}, Buffer size={len(lstmStrategy.replayBuffer)}')
                except Exception as trainErr:
                    logger.error(f'Agent {self.agentId}: LSTM training failed: {trainErr}')
        except Exception as e:
            logger.error(f'Agent {self.agentId}: Error in processLstm: {e}')

    def setSimDate(self, simDate):
        """
        Update the current simulation date for the agent.

        simDate: New simulation date (YYYY-MM-DD)
        """
        self.simDate = simDate
        print(f'DEBUG: Agent {self.agentId} simDate updated to {self.simDate}')

    def getSimDate(self):
        """
        Get the current simulation date for the agent.
        
        Current simulation date (YYYY-MM-DD)
        """
        return self.simDate

    def getEndDate(self):
        """
        Get the simulation end date.
        
        End date (YYYY-MM-DD) or None if not set
        """
        return self.endDate

    def setDecisionPeriod(self, decisionPeriod):
        """
        Update the decision period for trading (window size in days).

        decisionPeriod: Number of days between trading decisions (minimum 1)
        """
        self.decisionPeriod = max(1, decisionPeriod)
        logger.info(f'Agent {self.agentId}: decision period changed to {self.decisionPeriod} days')

    def getDecisionPeriod(self):
        """
        Get the current decision period.
        """
        return self.decisionPeriod

    def getTimestepCounter(self):
        """
        Get the total number of timesteps executed.
        """
        return self.timestepCounter

    def getTotalTrades(self):
        """
        Get the total number of trades placed.
        """
        return len(self.executionLog)

    def getExecutionLog(self):
        """
        Get the agent's execution log.
        """
        return self.executionLog

    def getStrategies(self):
        """
        Get the available strategies.
        """
        return self.strategies

    def getPerformanceTracker(self):
        """
        Get the performance tracker instance.
        """
        return self.performanceTracker

