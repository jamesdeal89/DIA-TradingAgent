# Stock Trading Intelligent Agent

This project aims to explore different intelligent trading strategies for an agent which takes stock information, press releases, general news, financials, and learns to place effective trades.
The agent will be able to act unsupervised, while being checked on by a user via a conversational user interface. 
The strategies will be evaluated across several stock markets to judge its generalisability across exchanges and geographic regions. 

## Strategies 

These are the broad strategies I plan to implement, although it should be noted that the implementation of a given strategy might not be exclusive to only one of these approaches.
For example, the 'state' of the DQN may be defined by considering the 'sentiment state' of the market.

### Deep Q-Network (DQN)

Agent uses a neural network to predict a Q-value for each of several actions given a particular state.
Here, the state would be the current state of the stock market.
The action would be placing a trade for a specific company (and potentially a specific type of trade.)
Learns a policy over time to effectively maximise profit.

### NLP / NLU for Sentiment Analysis - BERT/FinBERT

Agent collects news and quotes relating to listed companies. Companies with the most positive sentiment in their earnings reports, public statements, related news will be bought; 
whereas companies with negative sentiment will be shorted, for example.

### Mean Reversion / Statistical Equilibrium

Core idea is that prices move towards a long-term average. Agent identifies instances where a particular company's stock has deviated from it's historical average.
The agent will assume the value will return to that mean and trade based on that assumption. E.g. shorting an asset well above it's mean, versus acquiring an asset well bellow. 

## Feature Engineering / Data Pre-processing

To enable the agent to interpret the environment's dta, raw stock data may need to be converted into 'technical indicators'.
These 
