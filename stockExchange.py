from typing import Any
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd

class stockExchange:
    '''
    Acts as the simulated environment (stock exchange.)
    Agent can interact to:
    1. Place different types of trade.
    2. Check their investment account standing.
    3. Check prices of stocks (current or historical).
    4. Retrieve news on stocks.
    '''

    def createServerConnection(hostName, userName, userPassword):
        '''
        Establishes a connection to the server for storing trades the agent made and it's current portfolio.
        '''
        connection = None
        try:
            connection = mysql.connector.connect(
                host=hostName,
                user=userName,
                passwd=userPassword
            )    
            print("MySQL Database connection successful.")
        except Error as e:
            print(f'Error: {e}')
        return connection
    

    def executeDatabaseQuery(connection, query):
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
            print(f'Error: e')


    def getStockData(ticker: str, period: str) -> Any:
        '''
        Returns a dataframe for the provided ticker filled with that period's daily:
        - open, high, low, close, volume, dividends, stock splits.

        Example usage:
        print(getStockData('AAPL', '3mo').head())

        Valid periods include: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        '''

        stock = yf.Ticker(ticker)
        return stock.history(period=period)
    

