"""
Abstract base class for trading strategies.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class TradingStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    Each strategy implements analyse() to return a trade recommendation.
    """

    def __init__(self, name, version='1.0'):
        """Initialise strategy with name and version."""
        self.name = name
        self.version = version

    @abstractmethod
    def analyse(self, ticker, mic, simDate, exchange):
        """
        Analyse a stock and return a trade recommendation.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON')
            simDate: Current simulation date in YYYY-MM-DD format
            exchange: StockExchange instance for data access
        
        Returns:
            {
                'action': 'long' | 'short' | 'sell' | 'hold',
                'confidence': float [0.0, 1.0],
                'reason': str (explanation of decision),
                'targetQuantity': int (shares to trade if action != 'hold')
            }
        """
        pass

    def getName(self):
        """Return strategy name."""
        return self.name

    def getVersion(self):
        """Return strategy version."""
        return self.version