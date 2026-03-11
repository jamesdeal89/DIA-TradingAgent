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
        self.__executeDatabaseQuery(self.connection, query)

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

        self.__executeDatabaseQuery(self.connection, query)
        query = "CREATE TABLE IF NOT EXISTS portfolio (" \
        "accountId int," \
        "ticker varchar(255)," \
        "mic ENUM('XNAS','XLON','XHKG','XJPX')," \
        "tradeType ENUM('long','short')," \
        "quantity DECIMAL(10,4),"  \
        # If a short trade has been closed or not.
        "closed BOOL"  \
        ");"
        self.__executeDatabaseQuery(self.connection, query)
        


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

    def __executeDatabaseQuery(self, connection, query):
        '''
        Executes an SQL query on the database.
        Example queries: 'CREATE DATABASE IF NOT EXISTS stockExchange' 'CREATE TABLE IF NOT EXISTS trades'.
        '''
        cursor = connection.cursor()
        try:
            cursor.execute(query)
            connection.commit()
            print("Database query successful!")
        except Error as e:
            print(f'Error: {e}')


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

    def placeShort(self, ticker, quantity, accountId):
        '''
        Short (with automated borrowing) - sell a stock not owned via borrowing the stock. 
        Automatically borrows the stock at current price, then immediately sells it.
        Once 14 days pass, the current price is used to purchase a new stock and repay the broker. 
        Idea is that if the price falls, the shorter profits. 

        No balance check performed as the stock is not being 'bought'.
        '''

    def placeLong(self, ticker, quantity, accountId):
        '''
        Long - purchase stock outright.
        Stock is bought for current market price. Stock is added to portfolio.

        The balance of account with passed ID will be checked: returns -1 if not enough balance.
        '''


    
