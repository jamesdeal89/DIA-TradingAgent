# Stock Trading Intelligent Agent

This project aims to explore different intelligent trading strategies for an agent which takes stock information, press releases, general news, financials, and learns to place effective trades.
The agent will be able to act unsupervised, while being checked on by a user via a conversational user interface.
The strategies will be evaluated across several stock markets to judge its generalisability across exchanges and geographic regions.

## Requirements

Please use `pip install -r requirements.txt` to install prerequsites for this project.

## Dot Files

Ensure that the MySQL service is running and the corresponding credentials are placed in a `.env` file with format:

```
DBHOST = ""
DBUSER = ""
DBPASS = ""
```

## Usage

### Initilisation of Stock News and FinBERT

Run `python3 stockNews.py` to fill the database with historical news data ready for use.

The database will automatically be initialised here - *IMPORTANT* - ensure that MySQL is running prior to use.

Alternatively, using `python3 stockNews.py --finetine` will carry out unsupervised fine-tuning of FinBERT on the data.

## Running Agents

Run `streamlit run chatGUI.py` to start the GUI. This will automatically open the GUI in your default browser.

The GUI allows initialising agents, setting preferred strategies, adjusting epsilon for epsilon-greedy selection, etc.

It also allows you to 'chat' with the agent to see it's trades and performance in real-time.

Stopping an agent may be automatic, if an end-date was set, or manual via the 'Stop Trading' button.
