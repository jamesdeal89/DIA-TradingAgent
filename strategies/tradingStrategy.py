"""
Abstract base class for trading strategies.
"""
from abc import ABC, abstractmethod

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
        
        ticker: Stock ticker symbol (e.g., 'AAPL')
        mic: Market Identifier Code (e.g., 'XNAS', 'XLON')
        simDate: Current simulation date in YYYY-MM-DD format
        exchange: StockExchange instance for data access
        
        Returns a dict:
            {
                'action': 'long', 'short', 'sell', 'hold',
                'confidence': 0.0 to 1.0,
                'reason': string (explanation of decision),
                'targetQuantity': int (shares to trade if action not 'hold')
            }
        """
        pass
