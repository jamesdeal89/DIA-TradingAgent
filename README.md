# Stock Trading Intelligent Agent

This project aims to explore different intelligent trading strategies for an agent which takes stock information, press releases, general news, financials, and learns to place effective trades.
The agent will be able to act unsupervised, while being checked on by a user via a conversational user interface.
The strategies will be evaluated across several stock markets to judge its generalisability across exchanges and geographic regions.

## Strategies

These are the broad strategies I plan to implement, although it should be noted that the implementation of a given strategy might not be exclusive to only one of these approaches.
For example, the 'state' of the DQN may be defined by considering the 'sentiment state' of the market.

### Deep Q-Network (DQN)

Citation: https://doi.org/10.48550/arXiv.1312.5602 "Playing Atari with Deep Reinforcement Learning" by Volodymyr Mnih et al.

Agent uses a neural network to predict a Q-value for each of several actions given a particular state.
Here, the state would be the current state of the stock market.
The action would be placing a trade for a specific company (and potentially a specific type of trade.)
Learns a policy over time to effectively maximise profit.

### NLP / NLU for Sentiment Analysis - FinBERT

FinBERT - BERT fine-tuned for financial market sentiment classification: https://github.com/ProsusAI/finBERT.

Potential API for getting historical stock news: https://site.financialmodelingprep.com/developer/docs#stock-news.

Kaggle Dataset of Financial News 2009-2020, ~4m articles: https://www.kaggle.com/datasets/miguelaenlle/massive-stock-news-analysis-db-for-nlpbacktests

Agent collects news and quotes relating to listed companies. Companies with the most positive sentiment in their earnings reports, public statements, related news will be bought;
whereas companies with negative sentiment will be shorted, for example.

### Mean Reversion / Statistical Equilibrium

Core idea is that prices move towards a long-term average. Agent identifies instances where a particular company's stock has deviated from it's historical average.
The agent will assume the value will return to that mean and trade based on that assumption. E.g. shorting an asset well above it's mean, versus acquiring an asset well bellow.

## Feature Engineering / Data Pre-processing

To enable the agent to interpret the environment's data, raw stock data may need to be converted into 'technical indicators'.
These can then serve as features for a neural networks. For example, as part of a Deep Q-Network.

### Moving Averages - Smoothing

This provides some smoothing to the price data; removing noise and small short-term price fluctuations which are unlikely to have meaningful impact on the agents decisions.
A naive approach would be to use a moving average directly, where some look-back period is set to average over. However, this suffers from a lag effect where all the data points in the window have the same weighting as the current / most recent data point.
Instead, the Exponential Moving Average (EMA) assigns an exponentially increasing weight to more recent data points. This is implemented using a smoothing multiplier alpha (a).

> a = 2/(n+1)

> EMA_t = (P_t * a) + (EMA_t-1 * (1-a)).

The EMA means the agent can quickly detect trend changes that the naive MA would hide.

### Relative Strength Index (RSI)

RSI measures the magnitude of price changes to determine if a stock is 'oversold' or 'overbought'. The RSI can be between 0 and 100.

> RS = average gain/average loss

> RSI = 100-(100/1+RS).

The convention in trading is that an RSI above 70 is considered overbought - this should signal to the agent that it should sell the stock.
Whereas an RSI below 30 is oversold - which means the agent should likely buy.

## Quantitative Fundamental Analysis

QFA attempts to determine the 'intrinsic' value of a stock regardless of the exchange price.

### Price to Earnings Ratio

Share price compared to earnings per share. A lower P/E relative to competing companies could indicate an undervalued stock that should be bought.

### PEG Ratio

Takes the price-earnings ratio and divides it by the earnings-per-share (EPS) growth rate.
Convention is that a PEG below 1 are likely undervalued.
