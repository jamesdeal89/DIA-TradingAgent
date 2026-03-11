from typing import Any
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from dotenv import load_dotenv
load_dotenv()

class StockExchange:
    '''
    Acts as the simulated environment (stock exchange.)
    Agent can interact to:
    1. Place different types of trade.
    2. Check their investment account standing.
    3. Check prices of stocks (current or historical).
    4. Retrieve news on stocks.
    '''

    def __init__(self):
        '''
        Initialises DB.
        Creates stockExchange table.
        Creates accounts table.
        Creates portfolio table.
        Creates trades table.
        '''
        self.connection = self.__createServerConnection()
        query = "CREATE DATABASE IF NOT EXISTS StockExchange;" 
        self.__executeDatabaseQuery(query)

        self.connection.database = 'StockExchange'

        query = "CREATE TABLE IF NOT EXISTS trades (" \
        "accountId int," \
        "ticker varchar(255)," \
        "mic ENUM('XNAS','XLON','XHKG','XJPX')," \
        "tradeType ENUM('long','short')," \
        # Stocks can typically only be bought with a precision of 4 decimal places for general trading.
        # DECIMAL(10,4) specifies precision and scale.
        # 10 is the total number of digits (precision.)
        # 4 is the number of digits after the decimal point (scale.)
        "quantity DECIMAL(10,4),"
        "price FLOAT(32),"
        "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
        ");" 
        self.__executeDatabaseQuery(query)

        query = "CREATE TABLE IF NOT EXISTS portfolios (" \
        "accountId int," \
        "ticker varchar(255)," \
        "mic ENUM('XNAS','XLON','XHKG','XJPX')," \
        "tradeType ENUM('long','short')," \
        "quantity DECIMAL(10,4),"  \
        # If a short trade has been closed or not.
        "closed BOOL"  \
        ");"
        self.__executeDatabaseQuery(query)

        query = "CREATE TABLE IF NOT EXISTS accounts (" \
        "accountId int," \
        # For new account, automatically fund with a set amount to work with.
        "balance DECIMAL(32,2) DEFAULT 10000.00" \
        ");"
        self.__executeDatabaseQuery(query)
        
    def __createServerConnection(self):
        '''
        Establishes a connection to the server for storing trades the agent made and it's current portfolio.
        '''
        connection = None
        try:
            connection = mysql.connector.connect(
                host=os.getenv("DBHOST"),
                user=os.getenv("DBUSER"),
                passwd=os.getenv("DBPASS")
            )    
            print("MySQL Database connection successful.")
        except Error as e:
            print(f'Error: {e}')
        return connection

    def __executeDatabaseQuery(self, query, params=None):
        '''
        Executes an SQL query on the database.
        Example queries: 'CREATE DATABASE IF NOT EXISTS stockExchange' 'CREATE TABLE IF NOT EXISTS trades'.
        '''
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query,params)
            else:
                cursor.execute(query)
            self.connection.commit()
            print("Database query successful!")
            # Useful to check if updates worked.
            return cursor.rowcount
        except Error as e:
            print(f'Error: {e}')
            return -1
        
    def __getMicTicker(self, ticker, mic):
        '''
        yFinance uses suffixes to check a ticker from a specific MIC.
        This maps the official standard conventional MIC code to a MIC-specific ticker compatible with yFinance.
        '''
        # NOTE: Nasdaq / YSE do not have a suffix.
        suffix = {'XLON': '.L', 'XHKG': '.HK', 'XJPX': '.T'}
        # .get() is used as it prevents exceptions if an MIC is passed which is not in suffix dict (like XNAS which has no suffix.)
        # Instead returns default value '' if no match in dict.
        return f"{ticker}{suffix.get(mic, '')}"

    def initialiseAccount(self, accountId):
        '''
        Register a new trading account.
        Will automatically be funded with the default seed balance (10,000.00 USD).
        If the account already exists, it will not be overwritten, no exception will be thrown.
        '''
        query = "INSERT IGNORE INTO accounts (accountId) VALUES (%s)"
        self.__executeDatabaseQuery(query, (accountId,))

    def getStockData(self, ticker: str, period: str) -> Any:
        '''
        Returns a dataframe for the provided ticker filled with that period's daily:
        - open, high, low, close, volume, dividends, stock splits.

        Example usage:
        print(getStockData('AAPL', '3mo').head())

        Valid periods include: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        '''

        stock = yf.Ticker(ticker)
        return stock.history(period=period)

    def placeShort(self, ticker, mic, quantity, accountId):
        '''
        Short (with automated borrowing) - sell a stock not owned via borrowing the stock. 
        Automatically borrows the stock at current price, then immediately sells it.
        Once 14 days pass, the current price is used to purchase a new stock and repay the broker. 
        Idea is that if the price falls, the shorter profits. 

        No balance check performed as the stock is not being 'bought'.
        Returns -1 if accountId does not exist.
        '''
        # Add current price of shorted stock to account balance.
        price = self.getStockData(self.__getMicTicker(ticker,mic), '1d')['Close'].iloc[-1]
        cost = float(price * quantity)
        query = "UPDATE accounts SET balance = balance + %s WHERE accountId = %s"
        rowsAffected = self.__executeDatabaseQuery(query,(cost,accountId))

        # Place the short trade and add to portfolio.
        if rowsAffected:
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price) VALUES (%s,%s,%s,'short',%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, closed) VALUES (%s,%s,%s,'short',%s, FALSE);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity))
        else:
            print(f"ERROR: Account with ID {accountId} does not exist.")
            return -1
        
    def placeLong(self, ticker, mic, quantity, accountId):
        '''
        Long - purchase stock outright.
        Stock is bought for current market price. Stock is added to portfolio.

        The balance of account with passed ID will be checked: returns -1 if not enough balance / accountId does not exist.
        '''
        # Fetch current price as of close.
        price = self.getStockData(self.__getMicTicker(ticker,mic), '1d')['Close'].iloc[-1]
        # Calculate total.
        cost = float(price * quantity)
        # Check against balance and deduct.
        query = "UPDATE accounts SET balance = balance - %s WHERE accountId = %s AND balance >= %s"
        rowsAffected = self.__executeDatabaseQuery(query,(cost,accountId,cost))
        # Check if the balance check passed and the balance was deducted
        if rowsAffected:
            print("Balance deducted, trade is executing...")
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price) VALUES (%s,%s,%s,'long',%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity) VALUES (%s,%s,%s,'long',%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity))
        else:
            print(f"ERROR: Balance for account {accountId} too low for trade of cost {cost}")
            return -1


    
