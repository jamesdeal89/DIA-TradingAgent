"""
Deep Q-Learning Trading Strategy.

Uses a neural network to learn optimal trading decisions based on a multi-dimensional state:
- Sentiment score (from sentiment analyser)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price performance metrics (volatility, momentum, trend)
- Market microstructure features

The agent learns Q-values: Q(state, action) -> value
Actions: LONG (buy), SHORT (sell), HOLD (no trade)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import random
import logging
from tradingStrategy import TradingStrategy
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Neural network mapping state to Q-values."""
    
    def __init__(self, state_size: int, action_size: int = 4, hidden_size: int = 128):
        """Initialise network with given state and action sizes."""
        super(DQNNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Network architecture: input -> hidden -> hidden -> output
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass returning Q-values for each action."""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values


class StateBuilder:
    """Builds state vectors from market data (sentiment, technical, price, volume features)."""
    
    STATE_SIZE = 12  # sentiment(1) + technical(3) + price(3) + market(2) + extra(3)
    
    @staticmethod
    def buildState(ticker: str, mic: str, simDate: str, exchange, 
                   analysisPeriod: int = 1, sentimentAnalyser=None) -> np.ndarray:
        """Build normalized state vector for a stock."""
        state = []
        
        try:
            sentiment_score = StateBuilder._getSentimentFeature(ticker, mic, simDate, exchange, sentimentAnalyser)
            state.append(sentiment_score)
            
            rsi, macd, bbPosition = StateBuilder._getTechnicalFeatures(ticker, simDate, exchange, analysisPeriod)
            state.extend([rsi, macd, bbPosition])
            
            momentum, volatility, trend = StateBuilder._getPricePerformanceFeatures(ticker, simDate, exchange, analysisPeriod)
            state.extend([momentum, volatility, trend])
            
            volume_trend, price_level, extra_feature = StateBuilder._getMarketFeatures(ticker, simDate, exchange)
            state.extend([volume_trend, price_level, extra_feature])
            
            state = np.array(state, dtype=np.float32)
            
            if len(state) < StateBuilder.STATE_SIZE:
                state = np.pad(state, (0, StateBuilder.STATE_SIZE - len(state)), mode='constant', constant_values=0.0)
            elif len(state) > StateBuilder.STATE_SIZE:
                state = state[:StateBuilder.STATE_SIZE]
            
            state = np.clip(state, -1.0, 1.0)  
            
            return state
        
        except Exception as e:
            logger.error(f"Error building state for {ticker} on {simDate}: {e}")
            return np.zeros(StateBuilder.STATE_SIZE, dtype=np.float32)
    
    @staticmethod
    def _getSentimentFeature(ticker: str, mic: str, simDate: str, exchange, sentimentAnalyser) -> float:
        """Fetch and normalize sentiment score from news headlines."""
        if sentimentAnalyser is None or exchange is None:
            return 0.0
        
        try:
            # Fetch headlines for this stock on this date
            headlines = exchange.getNewsForStock(ticker, mic, simDate)
            
            if not headlines:
                return 0.0
            
            # Analyse headlines using SentimentAnalyser
            metrics = sentimentAnalyser.analyse(headlines)
            
            # Extract and return the average sentiment score
            avgSentiment = metrics.get('avgSentiment', 0.0)
            return float(np.clip(avgSentiment, -1.0, 1.0))
        
        except Exception as e:
            logger.warning(f"Error getting sentiment feature for {ticker}: {e}")
            return 0.0
    
    @staticmethod
    def _getTechnicalFeatures(ticker: str, simDate: str, exchange, analysisPeriod: int = 1) -> Tuple[float, float, float]:
        """Compute RSI, MACD, and Bollinger Band position."""
        try:
            # Fetch price history - scale lookback with analysisPeriod
            from datetime import datetime, timedelta
            end_date = datetime.strptime(simDate, '%Y-%m-%d')
            lookback_days = max(60, analysisPeriod * 2)  # More data for longer analysis windows
            start_date = end_date - timedelta(days=lookback_days)
            data = exchange.getStockData(ticker, start=start_date.strftime('%Y-%m-%d'), end=simDate)
            
            if data is None or len(data) < 14:
                return 0.0, 0.0, 0.0
            
            prices = data['Close'].values
            
            # RSI with period scaled to analysisPeriod
            rsi_period = max(14, analysisPeriod // 2)
            rsi = StateBuilder._calculateRsi(prices, period=rsi_period)
            rsiNorm = (rsi - 50) / 50  # Centre at 0, scale to [-1, 1]
            
            # MACD with periods scaled to analysisPeriod
            macd_fast = max(12, analysisPeriod // 2)
            macd_slow = max(26, analysisPeriod)
            macd, signal = StateBuilder._calculateMacd(prices, fast=macd_fast, slow=macd_slow, signal=9)
            macdDiff = macd - signal if macd is not None and signal is not None else 0.0
            macdNorm = np.clip(macdDiff / 100, -1.0, 1.0)  # Normalise
            
            # Bollinger Band position over analysisPeriod window
            bbPos = StateBuilder._calculateBollingerBandPosition(prices, period=max(20, analysisPeriod))
            bbNorm = (bbPos - 0.5) * 2  # Map [0, 1] to [-1, 1]
            
            return float(rsiNorm), float(macdNorm), float(bbNorm)
        
        except Exception as e:
            logger.warning(f"Error computing technical features: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def _getPricePerformanceFeatures(ticker: str, simDate: str, exchange, analysisPeriod: int = 1) -> Tuple[float, float, float]:
        """Compute momentum, volatility, and trend features."""
        try:
            # Fetch price history - scale to analysisPeriod
            from datetime import datetime, timedelta
            end_date = datetime.strptime(simDate, '%Y-%m-%d')
            lookback_days = max(60, analysisPeriod * 2)
            start_date = end_date - timedelta(days=lookback_days)
            prices_df = exchange.getStockData(ticker, start=start_date.strftime('%Y-%m-%d'), end=simDate)
            
            if prices_df is None or len(prices_df) < 2:
                return 0.0, 0.0, 0.0
            
            prices = prices_df['Close'].values
            
            # Momentum: (current - analysisPeriod-day avg) / current
            momentum_window = max(20, min(analysisPeriod, len(prices)))
            momentum = (prices[-1] - np.mean(prices[-momentum_window:])) / prices[-1] if prices[-1] != 0 else 0.0
            momentum = np.clip(momentum, -1.0, 1.0)
            
            # Volatility: std dev normalised over analysisPeriod window
            vol_window = max(20, min(analysisPeriod, len(prices)))
            volatility = np.std(prices[-vol_window:]) / np.mean(prices[-vol_window:]) if np.mean(prices[-vol_window:]) != 0 else 0.0
            volatility = np.clip(volatility / 0.1, -1.0, 1.0)  # Assume 10% is max reasonable vol
            
            # Trend: slope of analysisPeriod-day linear regression
            trend_window = max(10, min(analysisPeriod, len(prices)))
            x = np.arange(trend_window)
            y = prices[-trend_window:]
            z = np.polyfit(x, y, 1)
            trend = np.clip(z[0] / prices[-1], -1.0, 1.0)
            
            return float(momentum), float(volatility), float(trend)
        
        except Exception as e:
            logger.warning(f"Error computing price features: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def _getMarketFeatures(ticker: str, simDate: str, exchange) -> Tuple[float, float, float]:
        """Compute volume trend, price level, and price acceleration."""
        try:
            # Volume trend: get recent volume data
            data = exchange.getStockData(ticker, period='1mo')
            if data is None or len(data) < 1:
                return 0.0, 0.0, 0.0
            
            volumes = data['Volume'].values
            vol_trend = (volumes[-1] - np.mean(volumes[:-1])) / np.mean(volumes[:-1]) if np.mean(volumes[:-1]) != 0 else 0.0
            vol_trend = np.clip(vol_trend, -1.0, 1.0)
            
            # Price level: current price as percentile of 52-week range
            price_year = exchange.getStockData(ticker, period='1y')
            price_level = 0.0
            price_accel = 0.0
            
            if price_year is not None:
                try:
                    if len(price_year) > 0:
                        prices_year = price_year['Close'].values
                        price_level = (prices_year[-1] - np.min(prices_year)) / (np.max(prices_year) - np.min(prices_year)) if np.max(prices_year) != np.min(prices_year) else 0.5
                        price_level = (price_level - 0.5) * 2  # Map [0, 1] to [-1, 1]
                        
                        # Price acceleration: second derivative of price
                        if len(prices_year) > 2:
                            returns = np.diff(prices_year) / prices_year[:-1]
                            accel = np.diff(returns)
                            price_accel = np.clip(np.mean(accel[-10:]) * 100, -1.0, 1.0) if len(accel) > 0 else 0.0
                except (ValueError, TypeError):
                    pass
            
            return float(vol_trend), float(price_level), float(price_accel)
        
        except Exception as e:
            logger.warning(f"Error computing market features: {e}")
            return 0.0, 0.0, 0.0
    
    @staticmethod
    def _calculateRsi(prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    @staticmethod
    def _calculateMacd(prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float]]:
        """Calculate MACD."""
        if len(prices) < slow:
            return None, None
        
        ema_fast = StateBuilder._calculateEMA(prices, fast)
        ema_slow = StateBuilder._calculateEMA(prices, slow)
        macd = ema_fast - ema_slow
        signal_line = StateBuilder._calculateEMA(np.array([macd] * signal), signal)
        
        return float(macd), float(signal_line)
    
    @staticmethod
    def _calculateEMA(prices: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema
    
    @staticmethod
    def _calculateBollingerBandPosition(prices: np.ndarray, period: int = 20) -> float:
        """Calculate Bollinger Band position (0-1)."""
        if len(prices) < period:
            return 0.5
        
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        
        if std == 0:
            return 0.5
        
        bb_pos = (prices[-1] - (sma - 2 * std)) / (4 * std)
        return float(np.clip(bb_pos, 0.0, 1.0))


class DeepQLearningStrategy(TradingStrategy):
    """Deep Q-Learning strategy: learns optimal trading actions via neural network."""
    
    # Action encoding
    ACTION_LONG = 0
    ACTION_SHORT = 1
    ACTION_HOLD = 2
    ACTION_SELL = 3
    ACTION_MAP = {0: 'long', 1: 'short', 2: 'hold', 3: 'sell'}
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99, 
                 epsilon: float = 0.1, model_path: Optional[str] = None):
        """Initialise Deep Q-Learning strategy."""
        super().__init__(name="DeepQLearning", version="1.0")
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Initialise Q-network
        self.state_size = StateBuilder.STATE_SIZE
        self.action_size = 4  # LONG, SHORT, HOLD, SELL
        self.network = DQNNetwork(self.state_size, self.action_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()
        
        # Experience buffer for replay
        self.experience_buffer = deque(maxlen=1000)  # Store last 1000 experiences
        
        # Load pre-trained model if provided
        if model_path:
            try:
                self.network.load_state_dict(torch.load(model_path))
                logger.info(f"Loaded pre-trained model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load model from {model_path}: {e}")
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, analysisPeriod: int = 1) -> Dict[str, Any]:
        """Recommend trading action using Q-network inference."""
        try:
            # Build state with larger lookback window for longer analysis periods
            state = StateBuilder.buildState(ticker, mic, simDate, exchange, analysisPeriod=analysisPeriod)
            
            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
            
            # Get Q-values from network (no gradient needed for inference)
            with torch.no_grad():
                q_values = self.network(state_tensor).squeeze(0)  # Shape: (action_size,)
            
            # Select action: epsilon-greedy
            if random.random() < self.epsilon:
                action = random.randint(0, self.action_size - 1)
            else:
                action = torch.argmax(q_values).item()
            
            # Convert action index to string
            action_str = self.ACTION_MAP[action]
            
            # Compute confidence: softmax to convert Q-values to probabilities
            q_probs = torch.softmax(q_values, dim=0)
            confidence = float(q_probs[action].item())
            
            return {
                'action': action_str,
                'confidence': confidence,
                'targetQuantity': 1,  
                'qvalues': q_values.cpu().numpy().tolist(),
                'state': state.tolist()
            }
        
        except Exception as e:
            logger.error(f"Error in Deep Q-Learning analyse: {e}")
            return {'action': 'hold', 'confidence': 0.0, 'targetQuantity': 0, 'reason': str(e)}
    
    def recordExperience(self, state: np.ndarray, action: int, reward: float, 
                        next_state: np.ndarray, done: bool) -> None:
        """Record experience for training via experience replay."""
        self.experience_buffer.append((state, action, reward, next_state, done))
    
    def train(self, batch_size: int = 32) -> float:
        """Train network on random batch from experience buffer."""
        if len(self.experience_buffer) < batch_size:
            return 0.0
        
        # Sample random batch from experience buffer
        batch = random.sample(self.experience_buffer, batch_size)
        
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions)
        rewards_t = torch.FloatTensor(rewards)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones)
        
        # Q-learning: Q(s,a) = r + gamma * max(Q(s', a'))
        current_q = self.network(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        
        with torch.no_grad():
            max_next_q = self.network(next_states_t).max(dim=1)[0]
            target_q = rewards_t + self.gamma * max_next_q * (1 - dones_t)
        
        # Compute loss and backpropagate
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return float(loss.item())
    
    def saveModel(self, path: str) -> bool:
        """Save model weights to file."""
        try:
            torch.save(self.network.state_dict(), path)
            logger.info(f"Model saved to {path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def loadModel(self, path: str) -> bool:
        """Load model weights from file."""
        try:
            self.network.load_state_dict(torch.load(path))
            logger.info(f"Model loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
