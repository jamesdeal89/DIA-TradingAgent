"""
Strategy Selector using Epsilon-Greedy selection.

Selects trading strategies based on past performance.
Balances exploitation (use top performing) with exploration (try others randomly).
"""
import random
import logging
import numpy as np
logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Selects which strategy to use at a given timestep using epsilon-greedy selection.
    """

    def __init__(self, strategies, preferredStrategy=None, epsilon=0.3, minTradesRequired=5):
        """
        Initialise strategy selector with optional preferred strategy.
        If a preferred strategy is set and available, it will be prioritised over others.
        
        strategies: a dict mapping strategy names to TradingStrategy instances
        preferredStrategy: an optional strategy name to preferentially select
        epsilon: probability of random exploration (default: 0.2 = 20% random)
        minTradesRequired: min trades per strategy before ranking (i.e. initial pure exploration mode)
        """
        self.strategies = strategies
        self.preferredStrategy = preferredStrategy
        self.epsilon = epsilon
        self.minTradesRequired = minTradesRequired
        self.selectionHistory = []

    def selectStrategy(self, performanceMetrics, currentTradeCount):
        """
        Select a strategy using epsilon-greedy approach with preferred strategy support.
        
        If a preferred strategy is set and available, it has priority.
        Exploration mode (uniform random): if tradeCount < minTradesRequired.
        Exploitation mode: with probability of epsilon, choose randomly (still some exploration), otherwise best performing strategy.
        
        performanceMetrics: a dict of strategyName -> metrics from PerformanceTracker.getMetrics()
        currentTradeCount: total trades executed so far by this agent.
        
        Returns the selected strategy name.
        """
        availableStrategies = list(self.strategies.keys())
        if self.preferredStrategy and self.preferredStrategy in availableStrategies:
            logger.info(f'[StrategySelector] Using preferred strategy: {self.preferredStrategy}')
            self.selectionHistory.append(self.preferredStrategy)
            return self.preferredStrategy
        if currentTradeCount < self.minTradesRequired:
            selected = random.choice(availableStrategies)
            logger.info(f'[StrategySelector] Exploration mode ({currentTradeCount}/{self.minTradesRequired} trades): selected {selected}')
            self.selectionHistory.append(selected)
            return selected
        if random.random() < self.epsilon:
            selected = random.choice(availableStrategies)
            logger.info(f'[StrategySelector] Exploration phase (epsilon={self.epsilon}): selected {selected}')
        else:
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
            selected = random.choice(bestStrategies) if bestStrategies else random.choice(availableStrategies)
            logger.info(f'[StrategySelector] Exploitation: selected {selected} (PF={bestPf:.2f})')
        self.selectionHistory.append(selected)
        return selected
