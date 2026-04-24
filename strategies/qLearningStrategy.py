"""
Deep Q-Learning Trading Strategy with MLP.

Uses a neural network to learn optimal trading decisions based on a large market state snapshot:
- Sentiment score (from sentiment analyser)
- Technical indicators (RSI, MACD, Bollinger Bands)
- Price performance metrics (volatility, momentum, trend)
- Market conditions (volume trend, acceleration)

The agent learns Q-values: Q(state, action) -> value.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import logging
from .tradingStrategy import TradingStrategy
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DQNNetwork(nn.Module):
    """MLP mapping state to Q-values."""

    def __init__(self, stateSize, actionSize=4, hiddenSize=128):
        """Initialise network with given state and action sizes."""
        super(DQNNetwork, self).__init__()
        self.stateSize = stateSize
        self.actionSize = actionSize
        self.fc1 = nn.Linear(stateSize, hiddenSize)
        self.fc2 = nn.Linear(hiddenSize, hiddenSize)
        self.fc3 = nn.Linear(hiddenSize, actionSize)
        self.relu = nn.ReLU()

    def forward(self, state):
        """Forward pass returning Q-values for each action."""
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        qValues = self.fc3(x)
        return qValues

class StateBuilder:
    """Builds state vectors from market data."""
    STATE_SIZE = 10

    @staticmethod
    def buildState(ticker, mic, simDate, exchange, analysisPeriod=1, sentimentAnalyser=None):
        """Build normalised state vector for a stock."""
        state = []
        try:
            sentimentScore = StateBuilder.getSentimentFeature(ticker, mic, simDate, exchange, sentimentAnalyser)
            state.append(sentimentScore)
            rsi, macd, bbPosition = StateBuilder.getTechnicalFeatures(ticker, mic, simDate, exchange, analysisPeriod)
            state.extend([rsi, macd, bbPosition])
            momentum, volatility, trend = StateBuilder.getPricePerformanceFeatures(ticker, mic, simDate, exchange, analysisPeriod)
            state.extend([momentum, volatility, trend])
            volumeTrend, priceLevel, extraFeature = StateBuilder.getMarketFeatures(ticker, mic, simDate, exchange)
            state.extend([volumeTrend, priceLevel, extraFeature])
            state = np.array(state, dtype=np.float32)
            if len(state) != StateBuilder.STATE_SIZE:
                raise ValueError(f'Expected {StateBuilder.STATE_SIZE} features, got {len(state)}')
            state = np.clip(state, -1.0, 1.0)
            return state
        except Exception as e:
            logger.error(f'Error building state for {ticker} on {simDate}: {e}')
            return np.zeros(StateBuilder.STATE_SIZE, dtype=np.float32)

    @staticmethod
    def getSentimentFeature(ticker, mic, simDate, exchange, sentimentAnalyser):
        """Fetch and normalise sentiment score from news headlines."""
        if sentimentAnalyser is None or exchange is None:
            return 0.0
        try:
            headlines = exchange.getNewsForStock(ticker, mic, simDate)
            if not headlines:
                return 0.0
            metrics = sentimentAnalyser.analyse(headlines)
            avgSentiment = metrics.get('avgSentiment', 0.0)
            return float(np.clip(avgSentiment, -1.0, 1.0))
        except Exception as e:
            logger.warning(f'Error getting sentiment feature for {ticker}: {e}')
            return 0.0

    @staticmethod
    def getTechnicalFeatures(ticker, mic, simDate, exchange, analysisPeriod=1):
        """Compute RSI, MACD, and Bollinger Band position."""
        try:
            from datetime import datetime, timedelta
            endDate = datetime.strptime(simDate, '%Y-%m-%d')
            lookbackDays = max(60, analysisPeriod * 2)
            startDate = endDate - timedelta(days=lookbackDays)
            data = exchange.getStockData(ticker, mic=mic, start=startDate.strftime('%Y-%m-%d'), end=simDate)
            prices = StateBuilder.extractNumericSeries(data, 'Close', minLength=14)
            if prices is None:
                return (0.0, 0.0, 0.0)
            rsiPeriod = max(14, analysisPeriod // 2)
            rsi = StateBuilder.calculateRsi(prices, period=rsiPeriod)
            rsiNorm = (rsi - 50) / 50
            macdFast = max(12, analysisPeriod // 2)
            macdSlow = max(26, analysisPeriod)
            macd, signal = StateBuilder.calculateMacd(prices, fast=macdFast, slow=macdSlow, signal=9)
            macdDiff = macd - signal if macd is not None and signal is not None else 0.0
            macdNorm = np.clip(macdDiff / 100, -1.0, 1.0)
            bbPos = StateBuilder.calculateBollingerBandPosition(prices, period=max(20, analysisPeriod))
            bbNorm = (bbPos - 0.5) * 2
            return (float(rsiNorm), float(macdNorm), float(bbNorm))
        except Exception as e:
            logger.warning(f'Error computing technical features: {e}')
            return (0.0, 0.0, 0.0)

    @staticmethod
    def getPricePerformanceFeatures(ticker, mic, simDate, exchange, analysisPeriod=1):
        """Compute momentum, volatility, and trend features."""
        try:
            from datetime import datetime, timedelta
            endDate = datetime.strptime(simDate, '%Y-%m-%d')
            lookbackDays = max(60, analysisPeriod * 2)
            startDate = endDate - timedelta(days=lookbackDays)
            pricesDf = exchange.getStockData(ticker, mic=mic, start=startDate.strftime('%Y-%m-%d'), end=simDate)
            prices = StateBuilder.extractNumericSeries(pricesDf, 'Close', minLength=2)
            if prices is None:
                return (0.0, 0.0, 0.0)
            momentumWindow = max(20, min(analysisPeriod, len(prices)))
            momentum = (prices[-1] - np.mean(prices[-momentumWindow:])) / prices[-1] if prices[-1] != 0 else 0.0
            momentum = np.clip(momentum, -1.0, 1.0)
            volWindow = max(20, min(analysisPeriod, len(prices)))
            volatility = np.std(prices[-volWindow:]) / np.mean(prices[-volWindow:]) if np.mean(prices[-volWindow:]) != 0 else 0.0
            volatility = np.clip(volatility / 0.1, -1.0, 1.0)
            trendWindow = max(10, min(analysisPeriod, len(prices)))
            if trendWindow < 2:
                return (float(momentum), float(volatility), 0.0)
            x = np.arange(trendWindow)
            y = prices[-trendWindow:]
            z = np.polyfit(x, y, 1)
            trend = np.clip(z[0] / prices[-1], -1.0, 1.0)
            return (float(momentum), float(volatility), float(trend))
        except Exception as e:
            logger.warning(f'Error computing price features: {e}')
            return (0.0, 0.0, 0.0)

    @staticmethod
    def getMarketFeatures(ticker, mic, simDate, exchange):
        """Compute volume trend, price level, and price acceleration."""
        try:
            endDate = datetime.strptime(simDate, '%Y-%m-%d')
            monthStart = (endDate - timedelta(days=35)).strftime('%Y-%m-%d')
            yearStart = (endDate - timedelta(days=370)).strftime('%Y-%m-%d')
            data = exchange.getStockData(ticker, mic=mic, start=monthStart, end=simDate)
            volumes = StateBuilder.extractNumericSeries(data, 'Volume', minLength=1)
            if volumes is None:
                return (0.0, 0.0, 0.0)
            histMean = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
            volTrend = (volumes[-1] - histMean) / histMean if histMean != 0 else 0.0
            volTrend = np.clip(volTrend, -1.0, 1.0)
            priceYear = exchange.getStockData(ticker, mic=mic, start=yearStart, end=simDate)
            priceLevel = 0.0
            priceAccel = 0.0
            pricesYear = StateBuilder.extractNumericSeries(priceYear, 'Close', minLength=1)
            if pricesYear is not None:
                try:
                    priceLevel = (pricesYear[-1] - np.min(pricesYear)) / (np.max(pricesYear) - np.min(pricesYear)) if np.max(pricesYear) != np.min(pricesYear) else 0.5
                    priceLevel = (priceLevel - 0.5) * 2
                    if len(pricesYear) > 2:
                        returns = np.diff(pricesYear) / pricesYear[:-1]
                        accel = np.diff(returns)
                        priceAccel = np.clip(np.mean(accel[-10:]) * 100, -1.0, 1.0) if len(accel) > 0 else 0.0
                except (ValueError, TypeError):
                    pass
            return (float(volTrend), float(priceLevel), float(priceAccel))
        except Exception as e:
            logger.warning(f'Error computing market features: {e}')
            return (0.0, 0.0, 0.0)

    @staticmethod
    def extractNumericSeries(data, columnName, minLength=1):
        """Safely extract a numeric series from dataframes."""
        if data is None:
            return None
        try:
            if columnName not in data.columns:
                return None
            series = data[columnName]
            if series is None:
                return None
            values = np.asarray(series.values, dtype=np.float32)
            values = values[np.isfinite(values)]
            if len(values) < minLength:
                return None
            return values
        except Exception:
            return None

    @staticmethod
    def calculateRsi(prices, period=14):
        """Calculate RSI."""
        if len(prices) < period + 1:
            return 50.0
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avgGain = np.mean(gains[-period:])
        avgLoss = np.mean(losses[-period:])
        if avgLoss == 0:
            return 100.0 if avgGain > 0 else 50.0
        rs = avgGain / avgLoss
        rsi = 100 - 100 / (1 + rs)
        return float(rsi)

    @staticmethod
    def calculateMacd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD."""
        if len(prices) < slow:
            return (None, None)
        emaFast = StateBuilder.calculateEma(prices, fast)
        emaSlow = StateBuilder.calculateEma(prices, slow)
        macd = emaFast - emaSlow
        signalLine = StateBuilder.calculateEma(np.array([macd] * signal), signal)
        return (float(macd), float(signalLine))

    @staticmethod
    def calculateEma(prices, period):
        """Calculate EMA."""
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = price * multiplier + ema * (1 - multiplier)
        return ema

    @staticmethod
    def calculateBollingerBandPosition(prices, period=20):
        """Calculate Bollinger Band position (0-1)."""
        if len(prices) < period:
            return 0.5
        sma = np.mean(prices[-period:])
        std = np.std(prices[-period:])
        if std == 0:
            return 0.5
        bbPos = (prices[-1] - (sma - 2 * std)) / (4 * std)
        return float(np.clip(bbPos, 0.0, 1.0))

class DeepQLearningStrategy(TradingStrategy):
    """Deep Q-Learning strategy: learns optimal trading actions via neural network."""
    ACTION_LONG = 0
    ACTION_SHORT = 1
    ACTION_HOLD = 2
    ACTION_SELL = 3
    ACTION_MAP = {0: 'long', 1: 'short', 2: 'hold', 3: 'sell'}

    def __init__(self, learningRate=0.001, gamma=0.99, epsilon=0.1, modelPath=None):
        """Initialise Deep Q-Learning strategy."""
        super().__init__(name='DeepQLearning', version='1.0')
        self.learningRate = learningRate
        self.gamma = gamma
        self.epsilon = epsilon
        self.stateSize = StateBuilder.STATE_SIZE
        self.actionSize = 4
        self.network = DQNNetwork(self.stateSize, self.actionSize)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learningRate)
        self.lossFn = nn.MSELoss()
        self.experienceBuffer = deque(maxlen=1000)
        if modelPath:
            try:
                self.network.load_state_dict(torch.load(modelPath))
                logger.info(f'Loaded pre-trained model from {modelPath}')
            except Exception as e:
                logger.warning(f'Could not load model from {modelPath}: {e}')

    def analyse(self, ticker, mic, simDate, exchange, analysisPeriod=1):
        """Recommend trading action using Q-network inference."""
        try:
            state = StateBuilder.buildState(ticker, mic, simDate, exchange, analysisPeriod=analysisPeriod)
            stateTensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                qValues = self.network(stateTensor).squeeze(0)
            if random.random() < self.epsilon:
                action = random.randint(0, self.actionSize - 1)
            else:
                action = torch.argmax(qValues).item()
            actionStr = self.ACTION_MAP[action]
            qProbs = torch.softmax(qValues, dim=0)
            confidence = float(qProbs[action].item())
            return {'action': actionStr, 'confidence': confidence, 'targetQuantity': 1, 'qvalues': qValues.cpu().numpy().tolist(), 'state': state.tolist()}
        except Exception as e:
            logger.error(f'Error in Deep Q-Learning analyse: {e}')
            return {'action': 'hold', 'confidence': 0.0, 'targetQuantity': 0, 'reason': str(e)}

    def recordExperience(self, state, action, reward, nextState, done):
        """Record experience for training via experience replay."""
        self.experienceBuffer.append((state, action, reward, nextState, done))

    def train(self, batchSize=32):
        """Train network on random batch from experience buffer."""
        if len(self.experienceBuffer) < batchSize:
            return 0.0
        batch = random.sample(self.experienceBuffer, batchSize)
        states, actions, rewards, nextStates, dones = zip(*batch)
        statesT = torch.FloatTensor(np.array(states))
        actionsT = torch.LongTensor(actions)
        rewardsT = torch.FloatTensor(rewards)
        nextStates_t = torch.FloatTensor(np.array(nextStates))
        donesT = torch.FloatTensor(dones)
        currentQ = self.network(statesT).gather(1, actionsT.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            maxNextQ = self.network(nextStates_t).max(dim=1)[0]
            targetQ = rewardsT + self.gamma * maxNextQ * (1 - donesT)
        loss = self.lossFn(currentQ, targetQ)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())
