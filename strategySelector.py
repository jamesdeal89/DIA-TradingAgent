"""
Strategy Selector using Epsilon-Greedy Algorithm.

Selects trading strategies based on past performance.
Balances exploitation (use top performer) with exploration (try others).
"""
import random
import logging
from typing import Dict, Optional
import numpy as np
from strategies import TradingStrategy
logger = logging.getLogger(__name__)

class StrategySelector:
    """
    Selects which strategy to use at a given timestep using epsilon-greedy sampling.
    Balances exploitation (use top performer) with exploration (try others).
    """

    def __init__(self, strategies, preferredStrategy=None, epsilon=0.3, minTradesRequired=5):
        """
        Initialise strategy selector with optional preferred strategy.
        
        If a preferred strategy is set and available, it will be prioritised over others.
        
        Args:
            strategies: Dict mapping strategy names to TradingStrategy instances
            preferredStrategy: Optional strategy name to preferentially select
            epsilon: Probability of random exploration (default: 0.2 = 20% random)
            minTradesRequired: Minimum trades per strategy before ranking (default: 3)
        """
        self.strategies = strategies
        self.preferredStrategy = preferredStrategy
        self.epsilon = epsilon
        self.minTradesRequired = minTradesRequired
        self._selectionHistory = []

    def selectStrategy(self, performanceMetrics, currentTradeCount):
        """
        Select a strategy using epsilon-greedy approach with preferred strategy support.
        
        If a preferred strategy is set and available, it has priority.
        Exploration mode (uniform random): if tradeCount < minTradesRequired.
        Exploitation mode: with probability of epsilon, choose randomly (still some exploration), otherwise best performing strategy.
        
        Args:
            performanceMetrics: Dict of strategyName -> metrics from PerformanceTracker.getMetrics()
            currentTradeCount: Total trades executed so far by this agent
        
        Returns:
            Selected strategy name
        """
        availableStrategies = list(self.strategies.keys())
        if self.preferredStrategy and self.preferredStrategy in availableStrategies:
            logger.info(f'[StrategySelector] Using preferred strategy: {self.preferredStrategy}')
            self._selectionHistory.append(self.preferredStrategy)
            return self.preferredStrategy
        if currentTradeCount < self.minTradesRequired:
            selected = random.choice(availableStrategies)
            logger.info(f'[StrategySelector] Exploration mode ({currentTradeCount}/{self.minTradesRequired} trades): selected {selected}')
            self._selectionHistory.append(selected)
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
        self._selectionHistory.append(selected)
        return selected

    def getWeights(self):
        """
        Get current weight distribution across strategies based on selection history.
        
        Returns:
            Dict mapping strategy names to their selection weights (sum = 1.0)
        """
        if not self._selectionHistory:
            n = len(self.strategies)
            return {name: 1.0 / n for name in self.strategies.keys()}
        counts = Counter(self._selectionHistory)
        total = sum(counts.values())
        return {strategyName: counts.get(strategyName, 0) / total for strategyName in self.strategies.keys()}