"""
LSTM Trading Strategy.

Uses Long Short-Term Memory neural network to learn trading decisions from 
historical price sequences. The network processes past lookback-day price 
histories and predicts optimal trading actions based on temporal patterns.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Any, Optional
from collections import deque
import random
import logging
from datetime import datetime, timedelta
from .tradingStrategy import TradingStrategy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LSTMNetwork(nn.Module):
    """LSTM network for Q-value estimation."""
    
    def __init__(self, lookbackDays: int, inputFeatures: int = 6, 
                 hiddenSize: int = 32, actionSize: int = 4):
        """
        Initialise LSTM network.
        
        Args:
            lookbackDays: Number of days in sequence
            inputFeatures: Number of features per day (OHLCV + RSI)
            hiddenSize: LSTM hidden layer size
            actionSize: Number of actions (4: BUY, SELL, SHORT, CLOSE)
        """
        super(LSTMNetwork, self).__init__()
        self.lookbackDays = lookbackDays
        self.inputFeatures = inputFeatures
        self.hiddenSize = hiddenSize
        
        self.lstm = nn.LSTM(
            input_size=inputFeatures,
            hidden_size=hiddenSize,
            num_layers=1,
            batch_first=True
        )
        
        self.fc1 = nn.Linear(hiddenSize, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, actionSize)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape [batch, lookbackDays, inputFeatures]
        
        Returns:
            Q-values for each action [batch, actionSize]
        """
        lstmOut, (h_n, c_n) = self.lstm(x)
        lastHidden = lstmOut[:, -1, :]
        
        x = self.relu(self.fc1(lastHidden))
        x = self.relu(self.fc2(x))
        qValues = self.fc3(x)
        
        return qValues


class LSTMStrategy(TradingStrategy):
    """
    LSTM-based reinforcement learning strategy.
    Processes price history sequences to predict trading actions.
    """
    
    def __init__(self, hiddenSize: int = 32, learningRate: float = 0.001, 
                 epsilon: float = 0.1, gamma: float = 0.99):
        """
        Initialise LSTM strategy.
        
        Args:
            hiddenSize: LSTM hidden layer size
            learningRate: Adam optimiser learning rate
            epsilon: Epsilon-greedy exploration rate
            gamma: Discount factor for future rewards
        """
        super().__init__(name="LSTM", version="1.0")
        self.hiddenSize = hiddenSize
        self.learningRate = learningRate
        self.epsilon = epsilon
        self.gamma = gamma
        self.inputFeatures = 6
        
        self.network = None
        self.optimiser = None
        self.lossFn = nn.MSELoss()
        
        self.replayBuffer = deque(maxlen=5000)
        self.actionSpace = ["BUY", "SELL", "SHORT", "CLOSE"]
    
    def _initialiseNetwork(self, lookbackDays: int):
        """Initialise network for given lookback window."""
        if self.network is None:
            self.network = LSTMNetwork(
                lookbackDays=lookbackDays,
                inputFeatures=self.inputFeatures,
                hiddenSize=self.hiddenSize,
                actionSize=len(self.actionSpace)
            )
            self.optimiser = optim.Adam(
                self.network.parameters(),
                lr=self.learningRate
            )
    
    def _computeRSI(self, prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Compute Relative Strength Index."""
        if len(prices) < period:
            return np.zeros(len(prices))
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100. / (1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            if delta > 0:
                upval = delta
                downval = 0.
            else:
                upval = 0.
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            rs = up / down if down != 0 else 0
            rsi[i] = 100. - 100. / (1. + rs)
        
        return rsi
    
    def _getSequenceFeatures(self, ticker: str, mic: str, simDate: str, 
                             exchange, lookbackDays: int) -> Optional[np.ndarray]:
        """
        Fetch price history and compute feature sequence.
        
        Returns:
            numpy array of shape [lookbackDays, 6] or None if insufficient data
        """
        try:
            endDate = datetime.strptime(simDate, "%Y-%m-%d")
            startDate = endDate - timedelta(days=lookbackDays + 50)
            
            suffixedTicker = exchange.getMicTicker(ticker, mic)
            data = exchange.getStockData(suffixedTicker, start=None, end=simDate)
            
            if data is None or len(data) < lookbackDays:
                return None
            
            close = data['Close'].values
            high = data['High'].values
            low = data['Low'].values
            openPrices = data['Open'].values
            volume = data['Volume'].values
            
            rsi = self._computeRSI(close)
            
            features = np.column_stack((openPrices, high, low, close, volume, rsi))
            
            if len(features) < lookbackDays:
                return None
            
            latestSequence = features[-lookbackDays:]
            
            mean = latestSequence.mean(axis=0)
            std = latestSequence.std(axis=0)
            std[std == 0] = 1
            
            normalised = (latestSequence - mean) / std
            
            return normalised
        
        except Exception as e:
            logger.warning(f"Failed to fetch sequence for {ticker}: {e}")
            return None
    
    def analyse(self, ticker: str, mic: str, simDate: str, exchange, 
                analysisPeriod: int = 20) -> Dict[str, Any]:
        """
        Predict trading action using LSTM network.
        
        Args:
            analysisPeriod: Historical lookback window in days
        
        Returns:
            Trading recommendation with action, confidence, and reasoning
        """
        try:
            lookbackDays = max(10, analysisPeriod)
            self._initialiseNetwork(lookbackDays)
            
            features = self._getSequenceFeatures(ticker, mic, simDate, exchange, lookbackDays)
            
            if features is None:
                logger.info(f"LSTM {ticker}: insufficient historical data at {simDate} (need {lookbackDays} days)")
                return {
                    'action': 'hold',
                    'confidence': 0.0,
                    'reason': f'Insufficient historical data (< {lookbackDays} days)',
                    'targetQuantity': 1
                }
            
            stateTensor = torch.FloatTensor(features).unsqueeze(0)
            
            with torch.no_grad():
                qValues = self.network(stateTensor)
            
            if random.random() < self.epsilon:
                actionIdx = random.randint(0, len(self.actionSpace) - 1)
                logger.debug(f"LSTM exploration: random action {actionIdx} ({self.actionSpace[actionIdx]})")
            else:
                actionIdx = torch.argmax(qValues[0]).item()
                logger.debug(f"LSTM exploitation: argmax action {actionIdx} ({self.actionSpace[actionIdx]}), Q-values={qValues[0].tolist()}")
            
            action = self.actionSpace[actionIdx]
            
            # Calculate confidence using softmax probability distribution (like DeepQL)
            qProbs = torch.softmax(qValues[0], dim=0)
            confidence = float(qProbs[actionIdx].item())
            
            actionMap = {
                "BUY": "long",
                "SELL": "sell",
                "SHORT": "short",
                "CLOSE": "sell"
            }
            
            return {
                'action': actionMap[action],
                'confidence': confidence,
                'reason': f'LSTM predicts {action} (Q={qValues[0][actionIdx].item():.3f}) over {lookbackDays}d',
                'targetQuantity': 1
            }
        
        except Exception as e:
            logger.error(f"Error in LSTM analysis for {ticker} at {simDate}: {e}")
            return {
                'action': 'hold',
                'confidence': 0.0,
                'reason': f'Analysis failed: {str(e)}',
                'targetQuantity': 1
            }
    
    def remember(self, state: np.ndarray, action: int, reward: float, 
                nextState: np.ndarray, done: bool):
        """Store experience in replay buffer."""
        self.replayBuffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'nextState': nextState,
            'done': done
        })
    
    def train(self, batchSize: int = 32, epochs: int = 1):
        """
        Train network on random samples from replay buffer.
        
        Args:
            batchSize: Number of samples per batch
            epochs: Number of training iterations
        """
        if len(self.replayBuffer) < batchSize or self.network is None:
            logger.debug(f"LSTM training skipped: buffer size {len(self.replayBuffer)} < batch size {batchSize}")
            return
        
        logger.info(f"LSTM training started: buffer size={len(self.replayBuffer)}, batch size={batchSize}, epochs={epochs}")
        totalLoss = 0.0
        
        for epoch in range(epochs):
            batch = random.sample(self.replayBuffer, min(batchSize, len(self.replayBuffer)))
            
            states = torch.stack([torch.FloatTensor(b['state']) for b in batch])
            actions = torch.LongTensor([b['action'] for b in batch])
            rewards = torch.FloatTensor([b['reward'] for b in batch])
            nextStates = torch.stack([torch.FloatTensor(b['nextState']) for b in batch])
            dones = torch.FloatTensor([b['done'] for b in batch])
            
            qPred = self.network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            
            with torch.no_grad():
                qNext = self.network(nextStates).max(1)[0]
                qTarget = rewards + (1 - dones) * self.gamma * qNext
            
            loss = self.lossFn(qPred, qTarget)
            totalLoss += loss.item()
            
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        
        avgLoss = totalLoss / epochs
        logger.info(f"LSTM training completed: average loss={avgLoss:.6f}")
