from typing import Any, Dict, List
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from datetime import datetime, timedelta, date
from dotenv import load_dotenv
load_dotenv()

class StockExchange:
    """
    Acts as the simulated environment (stock exchange.)
    Agent can interact to:
    1. Place different types of trade.
    2. Check their investment account standing.
    3. Check prices of stocks (current or historical).
    4. Retrieve news on stocks.
    """

    def __init__(self, performanceTracker=None):
        """
        Initialises DB.
        Creates stockExchange table.
        Creates accounts table.
        Creates portfolio table.
        Creates trades table.
        
        Args:
            performanceTracker: Optional PerformanceTracker instance for strategy performance recording
        """
        self.connection = self.__createServerConnection()
        self.performanceTracker = performanceTracker
        query = 'CREATE DATABASE IF NOT EXISTS StockExchange;'
        self.__executeDatabaseQuery(query)
        self.connection.database = 'StockExchange'
        query = "CREATE TABLE IF NOT EXISTS trades (accountId INT, ticker VARCHAR(255), mic ENUM('XNAS','XLON','XHKG','XJPX'), tradeType ENUM('long','short'), quantity DECIMAL(10,4), price FLOAT, timestamp DATETIME, strategyName VARCHAR(255), agentId INT)"
        self.__executeDatabaseQuery(query)
        query = "CREATE TABLE IF NOT EXISTS portfolios (accountId INT, ticker VARCHAR(255), mic ENUM('XNAS','XLON','XHKG','XJPX'), tradeType ENUM('long','short'), quantity DECIMAL(10,4), closed BOOL, priceAtShort FLOAT, entryPrice FLOAT, strategyName VARCHAR(255), agentId INT, entryDate DATE, UNIQUE KEY (accountId, ticker, tradeType))"
        self.__executeDatabaseQuery(query)
        query = 'CREATE TABLE IF NOT EXISTS accounts (accountId INT PRIMARY KEY, balance DECIMAL(32,2) DEFAULT 10000.00)'
        self.__executeDatabaseQuery(query)
        query = 'CREATE TABLE IF NOT EXISTS news_headlines (id INT PRIMARY KEY, ticker VARCHAR(255), headline TEXT, url TEXT, publisher VARCHAR(255), date DATE, sentiment VARCHAR(20), sentiment_score FLOAT, confidence FLOAT, INDEX idx_ticker_date (ticker, date))'
        self.__executeDatabaseQuery(query)
        query = 'CREATE TABLE IF NOT EXISTS portfolio_history (accountId INT, snapshotDate DATE, cashBalance DECIMAL(32,2), portfolioValue DECIMAL(32,2), totalValue DECIMAL(32,2), agentId INT, UNIQUE KEY (accountId, snapshotDate))'
        self.__executeDatabaseQuery(query)
        query = "CREATE TABLE IF NOT EXISTS strategy_recommendations (id INT AUTO_INCREMENT PRIMARY KEY, accountId INT, ticker VARCHAR(255), mic ENUM('XNAS','XLON','XHKG','XJPX'), strategyName VARCHAR(255), action ENUM('long','short','hold'), priceAtRecommendation FLOAT, confidence FLOAT, timestampRecommended DATETIME, priceAtScoring FLOAT, timestampScored DATETIME, outcome ENUM('CORRECT','MISSED_LONG','MISSED_SHORT','SOLD','PENDING'), INDEX idx_strategy_date (strategyName, timestampRecommended))"
        self.__executeDatabaseQuery(query)

    def __createServerConnection(self):
        """
        Establishes a connection to the server for storing trades the agent made and it's current portfolio.
        """
        connection = None
        try:
            connection = mysql.connector.connect(host=os.getenv('DBHOST'), user=os.getenv('DBUSER'), passwd=os.getenv('DBPASS'))
            print('MySQL database connection established.')
        except Error as e:
            print(f'Error: {e}')
        return connection

    def __executeDatabaseQuery(self, query, params=None):
        """
        Executes an SQL query on the database.
        Example queries: 'CREATE DATABASE IF NOT EXISTS stockExchange' 'CREATE TABLE IF NOT EXISTS trades'.
        """
        cursor = self.connection.cursor()
        try:
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            self.connection.commit()
            print('Database query executed.')
            return cursor.rowcount
        except Error as e:
            print(f'Error: {e}')
            return -1

    def __getMicTicker(self, ticker, mic):
        """
        yFinance uses suffixes to check a ticker from a specific MIC.
        This maps the official standard conventional MIC code to a MIC-specific ticker compatible with yFinance.
        """
        suffix = {'XLON': '.L', 'XHKG': '.HK', 'XJPX': '.T'}
        return f'{ticker}{suffix.get(mic, '')}'

    def getPrice(self, ticker, mic, simDate=None):
        """
        Get price for a ticker on a given date.
        Raises ValueError if data is unavailable; caller handles errors as needed.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', '0700')
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON', 'XHKG', 'XJPX')
            simDate: Date for historical price (YYYY-MM-DD). If None, uses latest.
        
        Returns:
            float: Close price on specified date
        
        Raises:
            ValueError: If stock is delisted or has no data on that date
        """
        try:
            if simDate:
                data = self.getStockData(ticker, mic=mic, end=simDate)
            else:
                data = self.getStockData(ticker, mic=mic, period='1d')
            if data is None or data.empty or 'Close' not in data.columns:
                raise ValueError(f'No price data available for {ticker} on {simDate}. Stock may be delisted.')
            price = float(data['Close'].iloc[-1])
            return price
        except (IndexError, KeyError) as e:
            raise ValueError(f'Cannot retrieve price for {ticker} on {simDate}. Stock may be delisted or data unavailable.')
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f'Error fetching price for {ticker}: {str(e)}')

    def getMicTicker(self, ticker, mic):
        """
        Public method: Convert bare ticker to yfinance-compatible ticker with MIC suffix.
        
        Args:
            ticker: Bare ticker symbol (e.g., '0700', 'AAPL')
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON', 'XHKG', 'XJPX')
        
        Returns:
            Suffixed ticker for yfinance (e.g., '0700.HK', 'AAPL')
        """
        return self.__getMicTicker(ticker, mic)

    def initialiseAccount(self, accountId):
        """
        Register a new trading account.
        Will automatically be funded with the default seed balance (10,000.00 USD).
        If the account already exists, it will not be overwritten, no exception will be thrown.
        """
        query = 'INSERT IGNORE INTO accounts (accountId) VALUES (%s)'
        self.__executeDatabaseQuery(query, (accountId,))

    def getStockData(self, ticker, mic=None, period=None, start=None, end=None):
        """
        Returns a dataframe for the provided ticker.
        If mic is provided, converts bare ticker to yfinance format (e.g., '0700' -> '0700.HK').
        If period is provided, returns data for that period.
        If start and end are provided, returns data between those dates.

        Args:
            ticker: Stock ticker symbol (bare or suffixed)
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON', 'XHKG', 'XJPX'). If provided, ticker will be converted.
            period: Time period (e.g., '1d', '1mo', '1y')
            start: Start date string (YYYY-MM-DD)
            end: End date string (YYYY-MM-DD)
        
        Valid periods include: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        """
        if mic is not None:
            ticker = self.__getMicTicker(ticker, mic)
        stock = yf.Ticker(ticker)
        if start or end:
            return stock.history(start=start, end=end)
        return stock.history(period=period)

    def placeShort(self, ticker, mic, quantity, accountId, simDate, strategyName, agentId=None):
        """
        Short (with automated borrowing) - sell a stock not owned via borrowing the stock. 
        Automatically borrows the stock at current price, then immediately sells it.
        Once 30 days pass, the current price is used to purchase a new stock and repay the broker. 
        Idea is that if the price falls, the shorter profits. 

        simDate allows the agent to act on historical data. 
        The agent can act on historical data to enable repeatability of experiments.

        No balance check performed as the stock is not being 'bought'.
        Returns -1 if accountId does not exist or strategyName is empty.
        
        Args:
            strategyName: REQUIRED - name of strategy initiating this trade (for performance tracking)
            agentId: Optional agent ID for tracking
        """
        if not strategyName:
            print(f'ERROR: strategyName is required to place a short')
            return -1
        try:
            price = self.getPrice(ticker, mic, simDate)
        except ValueError as e:
            print(f'ERROR: {str(e)}')
            return -1
        cost = float(price * quantity)
        query = 'UPDATE accounts SET balance = balance + %s WHERE accountId = %s'
        rowsAffected = self.__executeDatabaseQuery(query, (cost, accountId))
        if rowsAffected:
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'short',%s,%s,%s,%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, closed, priceAtShort, strategyName, agentId, entryDate) VALUES (%s,%s,%s,'short',%s, FALSE, %s,%s,%s,%s) ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f'ERROR: Account with ID {accountId} does not exist.')
            return -1

    def placeLong(self, ticker, mic, quantity, accountId, simDate, strategyName, agentId=None):
        """
        Long - purchase stock outright.
        Stock is bought for current market price. Stock is added to portfolio.

        The balance of account with passed ID will be checked: returns -1 if not enough balance / accountId does not exist.
        
        Args:
            strategyName: REQUIRED - name of strategy initiating this trade (for performance tracking)
            agentId: Optional agent ID for tracking
        """
        if not strategyName:
            print(f'ERROR: strategyName is required to place a long')
            return -1
        try:
            price = self.getPrice(ticker, mic, simDate)
        except ValueError as e:
            print(f'ERROR: {str(e)}')
            return -1
        cost = float(price * quantity)
        query = 'UPDATE accounts SET balance = balance - %s WHERE accountId = %s AND balance >= %s;'
        rowsAffected = self.__executeDatabaseQuery(query, (cost, accountId, cost))
        if rowsAffected:
            print('Balance deducted, trade is executing...')
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, entryPrice, strategyName, agentId, entryDate, closed) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s,FALSE) ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f'ERROR: Balance for account {accountId} too low for trade of cost {cost}')
            return -1

    def checkBalance(self, accountId):
        """
        Returns the balance as a float for the account linked to the passed accountId.
        Gracefully handles exceptions - returns None if failed for any reason.
        """
        query = 'SELECT balance FROM accounts WHERE accountId = %s;'
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (accountId,))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                print(f'ERROR: No account found with accountId {accountId}')
                return None
        except Error as e:
            print(f'ERROR: {e}')
            return None

    def checkPortfolio(self, accountId, simDate):
        """
        Returns a Pandas dataframe of all of the currently held OPEN positions.
        Includes held assets as well as open shorts. 
        Gracefully handles exceptions - returns None if failed for any reason.
        """
        self.checkAndAutoCloseShorts(accountId, simDate)
        query = 'SELECT * FROM portfolios WHERE accountId = %s AND (closed = FALSE OR closed IS NULL)'
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (accountId,))
            rows = cursor.fetchall()
            columns = [i[0] for i in cursor.description]
            result = pd.DataFrame(rows, columns=columns)
        except Error as e:
            print(f'ERROR: {e}')
            return None
        return result

    def sellLong(self, accountId, ticker, mic, quantity, simDate, strategyName=None, entryPrice=None, entryDate=None):
        """
        Sell a held portfolio asset at current market price.
        Atomic operation - will only succeed if account has sufficient quantity.
        If quantity becomes 0, removes the position record to keep DB clean.
        
        Args:
            accountId: Account ID selling the position
            ticker: Stock ticker symbol
            mic: Market Identifier Code
            quantity: Number of shares to sell
            simDate: Current simulation date (YYYY-MM-DD)
            strategyName: Name of strategy that opened the original position (for performance tracking)
            entryPrice: Price at which position was opened (for P&L calculation)
            entryDate: Date when position was opened (for duration tracking)
        
        Returns:
            0 on success, -1 on failure
        """
        try:
            price = self.getPrice(ticker, mic, simDate)
        except ValueError as e:
            print(f'ERROR: {str(e)}')
            return -1
        proceeds = float(price * quantity)
        query = "UPDATE portfolios SET quantity = quantity - %s WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND quantity >= %s"
        rowsAffected = self.__executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        if rowsAffected > 0:
            query = 'UPDATE accounts SET balance = balance + %s WHERE accountId = %s'
            self.__executeDatabaseQuery(query, (proceeds, accountId))
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName) VALUES (%s, %s, %s, 'sell', %s, %s, %s, %s)"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName))
            query = "DELETE FROM portfolios WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND quantity <= 0"
            self.__executeDatabaseQuery(query, (accountId, ticker))
            print(f'Sold {quantity} of {ticker} at {price}. Proceeds: {proceeds}')
            safe_strategy = strategyName if strategyName else 'Unknown'
            safe_entryPrice = entryPrice if entryPrice is not None else price
            safe_entryDate = entryDate if entryDate else simDate
            if self.performanceTracker:
                self.recordTradePnL(accountId, safe_strategy, entryPrice=safe_entryPrice, exitPrice=price, quantity=quantity, ticker=ticker, entryDate=safe_entryDate, exitDate=simDate, tradeType='long')
            return 0
        else:
            print(f'ERROR: Insufficient quantity of {ticker} to sell.')
            return -1

    def closeShort(self, accountId, ticker, mic, quantity, simDate, strategyName=None, priceAtShort=None, entryDate=None):
        """
        Settle a short early.
        Must purchase the stock at the current price to return it to the broker.
        If the price is lower than the priceAtShort, the difference is profit.
        If the price is higher, then the difference is a loss.
        Uses closed flag to prevent race conditions - won't close an already-closed position.
        Returns 0 if succeeded, -1 if failed.
        """
        try:
            price = float(self.getPrice(ticker, mic, simDate))
        except ValueError as e:
            print(f'ERROR: {str(e)}')
            return -1
        quantity = float(quantity)
        cost_to_buyback = float(price * quantity)
        if strategyName is None or priceAtShort is None or entryDate is None:
            query = "SELECT priceAtShort, strategyName, entryDate FROM portfolios WHERE accountId = %s AND ticker = %s AND tradeType = 'short' AND (closed = FALSE OR closed IS NULL)"
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, ticker))
                position = cursor.fetchone()
            except:
                position = None
            finally:
                cursor.close()
            if priceAtShort is None and position:
                priceAtShort = position[0]
            if strategyName is None and position:
                strategyName = position[1]
            if entryDate is None and position:
                entryDate = position[2]
        query = "UPDATE portfolios SET quantity = quantity - %s, closed = TRUE WHERE accountId = %s AND ticker = %s AND tradeType = 'short' AND quantity >= %s AND (closed = FALSE OR closed IS NULL)"
        rowsAffected = self.__executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        if rowsAffected > 0:
            query = 'UPDATE accounts SET balance = balance - %s WHERE accountId = %s'
            self.__executeDatabaseQuery(query, (cost_to_buyback, accountId))
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp) VALUES (%s, %s, %s, 'short', %s, %s, %s)"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate))
            print(f'Short for {ticker} closed at {price}. Cost to buy back: {cost_to_buyback}')
            safe_strategy = strategyName if strategyName else 'Unknown'
            safe_priceAtShort = priceAtShort if priceAtShort is not None else price
            safe_entryDate = entryDate if entryDate else simDate
            if self.performanceTracker:
                entryDateStr = safe_entryDate if isinstance(safe_entryDate, str) else safe_entryDate.strftime('%Y-%m-%d')
                self.recordTradePnL(accountId, safe_strategy, entryPrice=safe_priceAtShort, exitPrice=price, quantity=quantity, ticker=ticker, entryDate=entryDateStr, exitDate=simDate, tradeType='short')
            return 0
        else:
            print(f'ERROR: No open short found for {ticker} with sufficient quantity or already closed.')
            return -1

    def cleanupClosedPositions(self, accountId):
        """
        Remove fully closed positions (quantity <= 0 and closed = TRUE) from portfolios.
        Prevents stale data from accumulating in the DB.
        Returns the number of deleted records.
        """
        query = 'DELETE FROM portfolios WHERE accountId = %s AND closed IS TRUE AND quantity <= 0'
        rowsDeleted = self.__executeDatabaseQuery(query, (accountId,))
        if rowsDeleted > 0:
            print(f'Cleaned up {rowsDeleted} closed position(s).')
        return rowsDeleted

    def getClosedTradesPnL(self, accountId, strategyName=None):
        """
        Retrieve all closed trades with P&L calculations for performance tracking.
        
        For LONG trades: P&L = (exitPrice - entryPrice) * quantity
        For SHORT trades: P&L = (entryPrice - exitPrice) * quantity
        
        Args:
            accountId: Account to query
            strategyName: Optional filter for specific strategy
        
        Returns:
            List of dicts: [{strategy, agentId, pnl, pnlPercent, tradeType, ticker, entryPrice, exitPrice}]
        """
        if strategyName:
            query = '\n                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker, \n                       p.entryPrice, p.priceAtShort, p.quantity,\n                       (SELECT MAX(price) FROM trades \n                        WHERE accountId = %s AND ticker = p.ticker \n                        AND timestamp > p.entryDate) as exitPrice\n                FROM portfolios p\n                WHERE p.accountId = %s AND p.closed IS TRUE AND p.strategyName = %s\n                AND p.quantity <= 0\n            '
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, accountId, strategyName))
            except:
                return []
        else:
            query = '\n                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker,\n                       p.entryPrice, p.priceAtShort, p.quantity,\n                       (SELECT MAX(price) FROM trades\n                        WHERE accountId = %s AND ticker = p.ticker\n                        AND timestamp > p.entryDate) as exitPrice\n                FROM portfolios p\n                WHERE p.accountId = %s AND p.closed IS TRUE AND p.quantity <= 0\n            '
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, accountId))
            except:
                return []
        closed_trades = []
        try:
            for row in cursor.fetchall():
                strategy, agentId, tradeType, ticker, entryPrice, priceAtShort, quantity, exitPrice = row
                entryPrice = float(entryPrice) if entryPrice else 0.0
                priceAtShort = float(priceAtShort) if priceAtShort else 0.0
                quantity = float(quantity) if quantity else 0.0
                exitPrice = float(exitPrice) if exitPrice else 0.0
                if tradeType == 'long':
                    entry = entryPrice
                    exit_val = exitPrice
                    pnl = (exit_val - entry) * abs(quantity)
                    pnl_pct = (exit_val - entry) / entry * 100 if entry > 0 else 0
                elif tradeType == 'short':
                    entry = priceAtShort
                    exit_val = exitPrice
                    pnl = (entry - exit_val) * abs(quantity)
                    pnl_pct = (entry - exit_val) / entry * 100 if entry > 0 else 0
                else:
                    continue
                closed_trades.append({'strategy': strategy, 'agentId': agentId, 'pnl': pnl, 'pnlPercent': pnl_pct, 'tradeType': tradeType, 'ticker': ticker, 'entryPrice': entry, 'exitPrice': exit_val, 'quantity': abs(quantity)})
        except Error as e:
            print(f'ERROR querying closed trades: {e}')
        finally:
            cursor.close()
        return closed_trades

    def checkAndAutoCloseShorts(self, accountId, currentDate):
        """
        Auto-close any short positions that have been open for 30+ days.
        As per the domain model: shorts are held for max 30 days, then auto-repurchased.
        
        Args:
            accountId: Account to process
            currentDate: Current simulation date (YYYY-MM-DD)
        
        Returns:
            Number of shorts auto-closed
        """
        try:
            current_date_obj = datetime.strptime(currentDate, '%Y-%m-%d')
            cutoff_date = current_date_obj - timedelta(days=30)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
        except ValueError:
            print(f'ERROR: Invalid date format {currentDate}. Expected YYYY-MM-DD.')
            return 0
        query = "\n            SELECT ticker, mic, quantity, priceAtShort, strategyName, agentId, entryDate\n            FROM portfolios\n            WHERE accountId = %s AND tradeType = 'short' AND closed IS FALSE\n            AND entryDate <= %s\n        "
        cursor = self.connection.cursor()
        closed_count = 0
        try:
            cursor.execute(query, (accountId, cutoff_date_str))
            aged_shorts = cursor.fetchall()
            for ticker, mic, quantity, priceAtShort, strategyName, agentId, entryDate in aged_shorts:
                try:
                    if isinstance(entryDate, datetime):
                        entryDateObj = entryDate
                    elif isinstance(entryDate, date):
                        entryDateObj = datetime.combine(entryDate, datetime.min.time())
                    else:
                        entryDateObj = datetime.strptime(str(entryDate), '%Y-%m-%d')
                    expiryDate = entryDateObj + timedelta(days=30)
                    expiryDateStr = expiryDate.strftime('%Y-%m-%d')
                except (ValueError, AttributeError) as dateErr:
                    print(f'ERROR: Could not calculate expiry date for {ticker}: {dateErr}')
                    continue
                try:
                    exitPrice = self.getPrice(ticker, mic, expiryDateStr)
                except ValueError as e:
                    print(f'WARNING: {str(e)} - Skipping auto-close for {ticker}')
                    continue
                result = self.closeShort(accountId, ticker, mic, quantity, expiryDateStr, strategyName=strategyName, priceAtShort=priceAtShort, entryDate=entryDate)
                if result == 0:
                    closed_count += 1
                    print(f'Auto-closed aged short: {ticker} ({quantity} shares) at expiry date {expiryDateStr}')
        except Error as e:
            print(f'ERROR in checkAndAutoCloseShorts: {e}')
        finally:
            cursor.close()
        return closed_count

    def closeAllOpenPositions(self, accountId, currentDate, agentId=None):
        """
        Close ALL open positions (both longs and shorts) at current market price.
        Used at end of simulation to get true final P&L.
        
        Args:
            accountId: Account to process
            currentDate: Current simulation date (YYYY-MM-DD)
            agentId: Optional agent ID for portfolio snapshot logging
        
        Returns:
            Number of positions closed
        """
        closed_count = 0
        cursor = self.connection.cursor()
        try:
            query_longs = "\n                SELECT ticker, mic, quantity, strategyName, entryPrice, entryDate\n                FROM portfolios\n                WHERE accountId = %s AND tradeType = 'long' AND (closed = FALSE OR closed IS NULL)\n            "
            cursor.execute(query_longs, (accountId,))
            open_longs = cursor.fetchall()
            for ticker, mic, quantity, strategyName, entryPrice, entryDate in open_longs:
                try:
                    quantity_float = float(quantity)
                    entryPrice_float = float(entryPrice)
                    current_price = self.getPrice(ticker, mic, currentDate)
                    self.sellLong(accountId, ticker, mic, quantity_float, currentDate, strategyName, entryPrice_float, entryDate)
                    closed_count += 1
                    print(f'Closed open long: {ticker} {quantity_float} shares @ ${current_price:.2f}')
                except Exception as e:
                    print(f'Warning: Could not close long {ticker}: {e}')
            query_shorts = "\n                SELECT ticker, mic, quantity, strategyName, priceAtShort, entryDate\n                FROM portfolios\n                WHERE accountId = %s AND tradeType = 'short' AND (closed = FALSE OR closed IS NULL)\n            "
            cursor.execute(query_shorts, (accountId,))
            open_shorts = cursor.fetchall()
            for ticker, mic, quantity, strategyName, priceAtShort, entryDate in open_shorts:
                try:
                    quantity_float = float(quantity)
                    priceAtShort_float = float(priceAtShort)
                    current_price = self.getPrice(ticker, mic, currentDate)
                    self.closeShort(accountId, ticker, mic, quantity_float, currentDate, strategyName=strategyName, priceAtShort=priceAtShort_float, entryDate=entryDate)
                    closed_count += 1
                    print(f'Closed open short: {ticker} {quantity_float} shares @ ${current_price:.2f}')
                except Exception as e:
                    print(f'Warning: Could not close short {ticker}: {e}')
        except Error as e:
            print(f'ERROR in closeAllOpenPositions: {e}')
        finally:
            cursor.close()
        if agentId:
            self.logPortfolioSnapshot(accountId, agentId, currentDate)
            print(f'Recorded final portfolio snapshot')
        return closed_count

    def recordTradePnL(self, accountId, strategyName, entryPrice, exitPrice, quantity, ticker, entryDate, exitDate, tradeType='long'):
        """
        Record P&L for a closed trade to the performance tracker.
        Called when a trade is manually closed (sellLong or closeShort).
        
        Args:
            accountId: Account ID
            strategyName: Name of the strategy that initiated this trade (NEVER null - enforced at position opening)
            entryPrice: Price at which position was opened (for longs) or shorted (for shorts)
            exitPrice: Price at which position was closed
            quantity: Number of shares
            ticker: Stock ticker
            entryDate: Entry date (YYYY-MM-DD)
            exitDate: Exit date (YYYY-MM-DD)
            tradeType: 'long' or 'short' to determine P&L formula
        """
        if not self.performanceTracker:
            return
        try:
            quantity_float = float(quantity)
            entryPrice_float = float(entryPrice)
            exitPrice_float = float(exitPrice)
            self.performanceTracker.recordTrade(strategyName, entryPrice=entryPrice_float, exitPrice=exitPrice_float, quantity=int(quantity_float), ticker=ticker, entryDate=entryDate, exitDate=exitDate, tradeType=tradeType)
            if tradeType == 'short':
                pnl = (entryPrice_float - exitPrice_float) * quantity_float
            else:
                pnl = (exitPrice_float - entryPrice_float) * quantity_float
            pnl_pct = pnl / (entryPrice_float * quantity_float) * 100 if entryPrice_float > 0 else 0
            print(f'Recorded trade for {strategyName}: {ticker} ({tradeType}) P&L=${pnl:.2f} ({pnl_pct:.2f}%)')
        except Exception as e:
            print(f'ERROR recording trade P&L: {e}')

    def getNewsForStock(self, ticker, mic, simDate):
        """
        Retrieve news headlines for a stock on a given simulation date with on-demand sentiment computation.
        
        Implements lazy-loading sentiment cache:
        1. Query headlines from DB
        2. For headlines with NULL sentiment: compute using FinBERT at runtime
        3. Cache computed sentiments back to DB for next retrieval
        4. Return headlines with sentiment/score populated
        
        Accounts for date imprecision in Kaggle dataset by querying headlines from simDate through simDate+1.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            mic: Market Identifier Code for validation (e.g., 'XNAS', 'XLON')
            simDate: Date string in YYYY-MM-DD format for historical simulation
            
        Returns:
            List of headline dicts (one entry per headline):
            [
                {
                    'headline': str,        # Full headline text
                    'sentiment': str,       # 'positive', 'neutral', or 'negative'
                    'score': float,         # Normalised score [-1.0, +1.0]
                    'url': str,             # Article URL
                    'date': str,            # Publication date (YYYY-MM-DD)
                    'publisher': str        # News source
                },
                ...
            ]
            Returns empty list [] if no headlines found.
        """
        from datetime import datetime, timedelta
        import logging
        logger = logging.getLogger(__name__)
        try:
            sim_date_obj = datetime.strptime(simDate, '%Y-%m-%d')
            next_date = sim_date_obj + timedelta(days=1)
            next_date_str = next_date.strftime('%Y-%m-%d')
        except ValueError:
            print(f'ERROR: Invalid date format {simDate}. Expected YYYY-MM-DD.')
            return []
        query = 'SELECT id, headline, sentiment, sentiment_score, url, date, publisher FROM news_headlines WHERE ticker = %s AND date >= %s AND date <= %s ORDER BY date ASC'
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (ticker, simDate, next_date_str))
            rows = cursor.fetchall()
            headlines = []
            headlines_needing_sentiment = []
            for row in rows:
                headline_id, headline_text, sentiment, score, url, date, publisher = row
                if sentiment is None:
                    headlines_needing_sentiment.append({'id': headline_id, 'headline': headline_text, 'url': url, 'date': date, 'publisher': publisher, 'needs_computation': True})
                else:
                    headlines.append({'headline': headline_text, 'sentiment': sentiment, 'score': float(score) if score is not None else 0.0, 'url': url, 'date': str(date), 'publisher': publisher})
            if headlines_needing_sentiment:
                logger.info(f'[NewsDB] Computing sentiment for {len(headlines_needing_sentiment)} headlines with NULL sentiment')
                try:
                    from stockNews import NewsAnalyser
                    analyser = NewsAnalyser()
                    update_query = 'UPDATE news_headlines SET sentiment = %s, sentiment_score = %s, confidence = %s WHERE id = %s'
                    for headline_data in headlines_needing_sentiment:
                        sentiment_result = analyser.getSentiment(headline_data['headline'])
                        sentiment = sentiment_result['sentiment']
                        score = sentiment_result['score']
                        confidence = sentiment_result['confidence']
                        try:
                            cursor.execute(update_query, (sentiment, score, confidence, headline_data['id']))
                            self.connection.commit()
                            logger.debug(f'[NewsDB] Cached sentiment for headline ID {headline_data['id']}: {sentiment} ({score:.3f})')
                        except Exception as e:
                            logger.warning(f'[NewsDB] Failed to cache sentiment for ID {headline_data['id']}: {e}')
                        headlines.append({'headline': headline_data['headline'], 'sentiment': sentiment, 'score': score, 'url': headline_data['url'], 'date': str(headline_data['date']), 'publisher': headline_data['publisher']})
                except ImportError as e:
                    logger.warning(f'[NewsDB] Could not import NewsAnalyser for on-demand computation: {e}')
                    for headline_data in headlines_needing_sentiment:
                        headlines.append({'headline': headline_data['headline'], 'sentiment': 'neutral', 'score': 0.0, 'url': headline_data['url'], 'date': str(headline_data['date']), 'publisher': headline_data['publisher']})
            if headlines:
                logger.info(f'[NewsDB] Retrieved {len(headlines)} headlines for {ticker} on {simDate}')
                for i, h in enumerate(headlines[:2]):
                    logger.debug(f'  {i + 1}. [{h['sentiment']:8}] score={h['score']:6.3f} | {h['headline'][:70]}')
            return headlines
        except Exception as e:
            logger.error(f'ERROR querying news for {ticker}: {e}')
            return []

    def getMaxNewsDate(self):
        """
        Get the latest date available in the news_headlines dataset.
        
        Returns:
            Latest date as YYYY-MM-DD string, or None if dataset is empty.
        """
        query = 'SELECT MAX(date) FROM news_headlines'
        cursor = self.connection.cursor()
        try:
            cursor.execute(query)
            result = cursor.fetchone()
            if result and result[0]:
                return str(result[0])
            else:
                print('WARNING: No news data found in database.')
                return None
        except Exception as e:
            print(f'ERROR querying max news date: {e}')
            return None
        finally:
            cursor.close()

    def isValidSimulationDate(self, simDate):
        """
        Check if a simulation date is within valid bounds.
        Valid means: not in the future AND within the news_headlines dataset range.
        
        Args:
            simDate: Date string in YYYY-MM-DD format
        
        Returns:
            True if date is valid, False otherwise
        """
        try:
            sim_date_obj = datetime.strptime(simDate, '%Y-%m-%d')
            today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            if sim_date_obj > today:
                print(f'WARNING: Simulation date {simDate} is in the future. Setting to today.')
                return False
            max_news_date = self.getMaxNewsDate()
            if max_news_date:
                max_date_obj = datetime.strptime(max_news_date, '%Y-%m-%d')
                if sim_date_obj > max_date_obj:
                    print(f'WARNING: Simulation date {simDate} exceeds dataset limit {max_news_date}.')
                    return False
            return True
        except ValueError:
            print(f'ERROR: Invalid date format {simDate}. Expected YYYY-MM-DD.')
            return False

    def logPortfolioSnapshot(self, accountId, agentId, simDate):
        """
        Log current portfolio value snapshot to portfolio_history table.
        Call this once per decision period to track portfolio equity growth.
        
        Args:
            accountId: Account ID to snapshot
            agentId: Agent ID
            simDate: Snapshot date (YYYY-MM-DD)
        
        Returns:
            True if successful, False if error
        """
        try:
            cash_balance = self.checkBalance(accountId) or 0
            portfolio_df = self.checkPortfolio(accountId, simDate)
            portfolio_value = 0.0
            if portfolio_df is not None and (not portfolio_df.empty):
                for _, row in portfolio_df.iterrows():
                    ticker = row.get('ticker')
                    mic = row.get('mic')
                    qty = float(row.get('quantity', 0))
                    trade_type = row.get('tradeType')
                    if ticker and qty > 0:
                        try:
                            current_price = self.getPrice(ticker, mic, simDate)
                            if trade_type == 'long':
                                portfolio_value += current_price * qty
                            elif trade_type == 'short':
                                portfolio_value -= current_price * qty
                        except ValueError:
                            pass
            total_value = float(cash_balance) + portfolio_value
            cursor = self.connection.cursor()
            query = 'INSERT INTO portfolio_history (accountId, snapshotDate, cashBalance, portfolioValue, totalValue, agentId) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE cashBalance = VALUES(cashBalance), portfolioValue = VALUES(portfolioValue), totalValue = VALUES(totalValue)'
            cursor.execute(query, (accountId, simDate, float(cash_balance), portfolio_value, total_value, agentId))
            self.connection.commit()
            return True
        except Exception as e:
            print(f'ERROR logging portfolio snapshot: {e}')
            return False

    def getPortfolioHistory(self, accountId, startDate=None, endDate=None):
        """
        Retrieve portfolio history snapshots for an account.
        
        Args:
            accountId: Account ID
            startDate: Optional start date (YYYY-MM-DD)
            endDate: Optional end date (YYYY-MM-DD)
        
        Returns:
            DataFrame with columns: snapshotDate, cashBalance, portfolioValue, totalValue
        """
        try:
            query = 'SELECT snapshotDate, cashBalance, portfolioValue, totalValue FROM portfolio_history WHERE accountId = %s'
            params = [accountId]
            if startDate:
                query += ' AND snapshotDate >= %s'
                params.append(startDate)
            if endDate:
                query += ' AND snapshotDate <= %s'
                params.append(endDate)
            query += ' ORDER BY snapshotDate ASC'
            return pd.read_sql(query, self.connection, params=params)
        except Exception as e:
            print(f'ERROR retrieving portfolio history: {e}')
            return pd.DataFrame()