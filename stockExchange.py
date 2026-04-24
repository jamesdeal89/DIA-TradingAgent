import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from datetime import datetime, timedelta, date
import logging
from dotenv import load_dotenv
load_dotenv()

class StockExchange:
    """
    Acts as the environment for the agents.
    Agents can interact to:
    Place different types of trade.
    Check their investment account standing.
    Check prices of stocks (current or historical).
    Retrieve news on stocks.
    """

    # Map MIC codes to their base currencies
    MIC_CURRENCY = {
        'XNAS': 'USD',  
        'XLON': 'GBP',  
        'XHKG': 'HKD', 
        'XJPX': 'JPY'   
    }
    
    # Exchange rates to USD - abstract currency differences between markets from agents.
    EXCHANGE_RATES = {
        'USD': 1.0,
        'GBP': 1.35,
        'HKD': 0.13,
        'JPY': 0.0063
    }

    def __init__(self, performanceTracker=None):
        """
        Initialises DB.
        Creates stockExchange table.
        Creates accounts table.
        Creates portfolio table.
        Creates trades table.
        
        
        performanceTracker: Optional PerformanceTracker instance for strategy performance recording
        """
        self.connection = self.createServerConnection()
        # Agent creates tracker instance and passes it when init StockExchange.
        self.performanceTracker = performanceTracker
        query = 'CREATE DATABASE IF NOT EXISTS StockExchange;'
        self.executeDatabaseQuery(query)
        self.connection.database = 'StockExchange'
        
        query = '''
            CREATE TABLE IF NOT EXISTS trades (
                accountId INT, 
                ticker VARCHAR(255), 
                mic ENUM('XNAS','XLON','XHKG','XJPX'), 
                tradeType ENUM('long','short','sell'), 
                quantity DECIMAL(10,4), 
                price DECIMAL(10,4), 
                timestamp DATETIME, 
                strategyName VARCHAR(255), 
                agentId INT
            )
        '''
        self.executeDatabaseQuery(query)
        
        query = '''
            CREATE TABLE IF NOT EXISTS portfolios (
                accountId INT, 
                ticker VARCHAR(255), 
                mic ENUM('XNAS','XLON','XHKG','XJPX'), 
                tradeType ENUM('long','short'), 
                quantity DECIMAL(10,4), 
                closed BOOL, 
                priceAtShort DECIMAL(10,4), 
                entryPrice DECIMAL(10,4), 
                strategyName VARCHAR(255), 
                agentId INT, 
                entryDate DATE, 
                UNIQUE KEY (accountId, ticker, tradeType)
            )
        '''
        self.executeDatabaseQuery(query)
        
        query = '''
            CREATE TABLE IF NOT EXISTS accounts (
                accountId INT PRIMARY KEY, 
                balance DECIMAL(32,2) DEFAULT 10000.00
            )
        '''
        self.executeDatabaseQuery(query)
        
        query = '''
            CREATE TABLE IF NOT EXISTS news_headlines (
                id INT PRIMARY KEY, 
                ticker VARCHAR(255), 
                headline TEXT, 
                url TEXT, 
                publisher VARCHAR(255), 
                date DATE, 
                sentiment VARCHAR(20), 
                sentiment_score FLOAT, 
                confidence FLOAT, 
                INDEX idx_ticker_date (ticker, date)
            )
        '''
        self.executeDatabaseQuery(query)
        
        query = '''
            CREATE TABLE IF NOT EXISTS portfolio_history (
                accountId INT, 
                snapshotDate DATE, 
                cashBalance DECIMAL(32,2), 
                portfolioValue DECIMAL(32,2), 
                totalValue DECIMAL(32,2), 
                agentId INT, 
                UNIQUE KEY (accountId, snapshotDate)
            )
        '''
        self.executeDatabaseQuery(query)
        
        query = '''
            CREATE TABLE IF NOT EXISTS strategy_recommendations (
                id INT AUTO_INCREMENT PRIMARY KEY, 
                accountId INT, 
                ticker VARCHAR(255), 
                mic ENUM('XNAS','XLON','XHKG','XJPX'), 
                strategyName VARCHAR(255), 
                action ENUM('long','short','hold'), 
                priceAtRecommendation FLOAT, 
                confidence FLOAT, 
                timestampRecommended DATETIME, 
                priceAtScoring FLOAT, 
                timestampScored DATETIME, 
                outcome ENUM('CORRECT','MISSED_LONG','MISSED_SHORT','SOLD','PENDING'), 
                INDEX idx_strategy_date (strategyName, timestampRecommended)
            )
        '''
        self.executeDatabaseQuery(query)

    def createServerConnection(self):
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

    def executeDatabaseQuery(self, query, params=None):
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

    def getMicTicker(self, ticker, mic):
        """
        yFinance uses suffixes to check a ticker from a specific MIC.
        This maps the official standard conventional MIC code to a MIC-specific ticker compatible with yFinance.
        Seems to help avoid errors when fetching stock data in bulk.
        """
        suffix = {'XLON': '.L', 'XHKG': '.HK', 'XJPX': '.T'}
        return f'{ticker}{suffix.get(mic, '')}'

    def getExchangeRate(self, currency):
        """
        Get the exchange rate for a currency pair to USD.
        currency: Currency code 'GBP', 'HKD', 'JPY'
        Returns exchange rate (quantity of origin currency which equals 1 USD)
        
        ValueError if currency not supported
        """
        if currency not in self.EXCHANGE_RATES:
            raise ValueError(f'Currency {currency} not supported')
        
        return self.EXCHANGE_RATES[currency]

    def getPrice(self, ticker, mic, simDate=None):
        """
        Get price for a ticker on a given date, converted to USD.
        Raises ValueError if stock data is unavailable; caller handles errors as needed.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', '0700')
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON', 'XHKG', 'XJPX')
            simDate: Date for historical price (YYYY-MM-DD). If None, uses latest.
        
        Returns:
            float: Close price on specified date, converted to USD
        
        Raises:
            ValueError: If stock is delisted or has no data on that date
        """
        try:
            if simDate:
                data = self.getStockData(ticker, mic=mic, end=simDate)
            else:
                data = self.getStockData(ticker, mic=mic, period='1d')
            if data is None or data.empty or 'Close' not in data.columns:
                # Happens infrequently, could be rate-limiting from Yahoo Finance? 
                raise ValueError(f'No price data available for {ticker} on {simDate}. Stock may be delisted.')
            price = float(data['Close'].iloc[-1])
            
            # Local currency to USD conversion, if needed (for certain markets).
            # Abstract currency differences from the agent, Stock Exchange handles it, agent only sees USD no matter MIC.
            currency = self.MIC_CURRENCY.get(mic, 'USD')
            if currency != 'USD':
                exchange_rate = self.getExchangeRate(currency, simDate)
                price = price * exchange_rate
            
            return round(price, 4)
        except (IndexError, KeyError) as e:
            raise ValueError(f'Cannot retrieve price for {ticker} on {simDate}. Stock may be delisted or data unavailable.')
        except ValueError:
            raise
        except Exception as e:
            raise ValueError(f'Error fetching price for {ticker}: {str(e)}')

    def initialiseAccount(self, accountId):
        """
        Register a new trading account.
        Will automatically be funded with the default seed balance (10,000 USD).
        If the account already exists, it will not be overwritten, no exception will be thrown.
        """
        query = 'INSERT IGNORE INTO accounts (accountId) VALUES (%s)'
        self.executeDatabaseQuery(query, (accountId,))

    def getStockData(self, ticker, mic=None, period=None, start=None, end=None):
        """
        Returns a dataframe for the provided ticker.
        If mic is provided, converts bare ticker to yfinance format.
        If period is provided, returns data for that period.
        If start and end are provided, returns data between those dates.

        mic: Market Identifier Code (e.g., 'XNAS', 'XLON', 'XHKG', 'XJPX'). If provided, ticker will be converted.
        period: Time period (e.g., '1d', '1mo', '1y')
        start: Start date string (YYYY-MM-DD)
        end: End date string (YYYY-MM-DD)
        
        Valid periods include: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        """
        if mic is not None:
            ticker = self.getMicTicker(ticker, mic)
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
        
        strategyName: name of strategy initiating this trade (for performance tracking)
        agentId: optional agent ID for tracking
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
        rowsAffected = self.executeDatabaseQuery(query, (cost, accountId))
        if rowsAffected:
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'short',%s,%s,%s,%s,%s);"
            self.executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, closed, priceAtShort, strategyName, agentId, entryDate) VALUES (%s,%s,%s,'short',%s, FALSE, %s,%s,%s,%s) ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f'ERROR: Account with ID {accountId} does not exist.')
            return -1

    def placeLong(self, ticker, mic, quantity, accountId, simDate, strategyName, agentId=None):
        """
        Long - purchase stock.
        Stock is bought for current market price. Stock is added to portfolio.

        The balance of account with passed ID will be checked: returns -1 if not enough balance / accountId does not exist.
        
        strategyName: name of strategy initiating this trade (for performance tracking)
        agentId: optional agent ID for tracking
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
        rowsAffected = self.executeDatabaseQuery(query, (cost, accountId, cost))
        if rowsAffected:
            print('Balance deducted, trade is executing...')
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s);"
            self.executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, entryPrice, strategyName, agentId, entryDate, closed) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s,FALSE) ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f'ERROR: Balance for account {accountId} too low for trade of cost {cost}')
            return -1

    def checkBalance(self, accountId, simDate=None):
        """
        Returns the balance as a float for the account linked to the passed accountId.
        returns None if failed for any reason.

        accountId: Account ID to query.
        simDate: Optional simulation date (YYYY-MM-DD). When provided,
            aged shorts are auto-closed first so returned balance is up to date.
        """
        if simDate:
            self.checkAndAutoCloseShorts(accountId, simDate)
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
        returns None if failed for any reason.

        accountId: Account ID to query.
        simDate: Current simulation date str.
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
        
        accountId: Account ID selling the position
        ticker: Stock ticker symbol
        simDate: Current simulation date (YYYY-MM-DD)
        strategyName: Name of strategy that opened the original position (for performance tracking)
        entryPrice: Price at which position was opened (for P&L calculation)
        entryDate: Date when position was opened (for duration tracking)
        
        Returns 0 on success, -1 on failure
        """
        try:
            price = self.getPrice(ticker, mic, simDate)
        except ValueError as e:
            print(f'ERROR: {str(e)}')
            return -1

        if strategyName is None or entryPrice is None or entryDate is None:
            query = "SELECT strategyName, entryPrice, entryDate FROM portfolios WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND (closed = FALSE OR closed IS NULL)"
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, ticker))
                position = cursor.fetchone()
            except Error:
                position = None
            finally:
                cursor.close()
            if position:
                if strategyName is None:
                    strategyName = position[0]
                if entryPrice is None:
                    entryPrice = position[1]
                if entryDate is None:
                    entryDate = position[2]

        proceeds = float(price * quantity)
        query = "UPDATE portfolios SET quantity = quantity - %s WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND quantity >= %s"
        rowsAffected = self.executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        if rowsAffected > 0:
            query = 'UPDATE accounts SET balance = balance + %s WHERE accountId = %s'
            self.executeDatabaseQuery(query, (proceeds, accountId))
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName) VALUES (%s, %s, %s, 'sell', %s, %s, %s, %s)"
            self.executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName))
            query = "DELETE FROM portfolios WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND quantity <= 0"
            self.executeDatabaseQuery(query, (accountId, ticker))
            print(f'Sold {quantity} of {ticker} at {price}. Proceeds: {proceeds}')
            safeStrategy = strategyName if strategyName else 'Unknown'
            safeEntryPrice = entryPrice if entryPrice is not None else price
            safeEntryDate = entryDate if entryDate else simDate
            if self.performanceTracker:
                self.recordTradePnL(safeStrategy, entryPrice=safeEntryPrice, exitPrice=price, quantity=quantity, ticker=ticker, entryDate=safeEntryDate, exitDate=simDate, tradeType='long')
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
        costOfBuyback = float(price * quantity)
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
        rowsAffected = self.executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        if rowsAffected > 0:
            query = 'UPDATE accounts SET balance = balance - %s WHERE accountId = %s'
            self.executeDatabaseQuery(query, (costOfBuyback, accountId))
            print(f'Short for {ticker} closed at {price}. Cost to buy back: {costOfBuyback}')
            safeStrategy = strategyName if strategyName else 'Unknown'
            safePriceAtShort = priceAtShort if priceAtShort is not None else price
            safeEntryDate = entryDate if entryDate else simDate
            if self.performanceTracker:
                entryDateStr = safeEntryDate if isinstance(safeEntryDate, str) else safeEntryDate.strftime('%Y-%m-%d')
                self.recordTradePnL(safeStrategy, entryPrice=safePriceAtShort, exitPrice=price, quantity=quantity, ticker=ticker, entryDate=entryDateStr, exitDate=simDate, tradeType='short')
            return 0
        else:
            print(f'ERROR: No open short found for {ticker} with sufficient quantity or already closed.')
            return -1

    def getClosedTradesPnL(self, accountId, strategyName=None):
        """
        Retrieve all closed trades with P&L calculations for performance tracking.
        
        For LONG trades: P&L = (exitPrice - entryPrice) * quantity
        For SHORT trades: P&L = (entryPrice - exitPrice) * quantity
        
        accountId: Account to query
        strategyName: Optional filter for specific strategy
        
        Returns list of dicts: [{strategy, agentId, pnl, pnlPercent, tradeType, ticker, entryPrice, exitPrice}]
        """
        cursor = self.connection.cursor()
        if strategyName:
            query = '''
                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker, p.entryPrice, 
                       p.priceAtShort, p.quantity,
                       (SELECT MAX(price) FROM trades 
                        WHERE accountId = %s AND ticker = p.ticker AND timestamp > p.entryDate) as exitPrice
                FROM portfolios p
                WHERE p.accountId = %s AND p.closed = TRUE AND p.strategyName = %s AND p.quantity <= 0
            '''
            try:
                cursor.execute(query, (accountId, accountId, strategyName))
            except:
                return []
        else:
            query = '''
                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker, p.entryPrice, 
                       p.priceAtShort, p.quantity,
                       (SELECT MAX(price) FROM trades 
                        WHERE accountId = %s AND ticker = p.ticker AND timestamp > p.entryDate) as exitPrice
                FROM portfolios p
                WHERE p.accountId = %s AND p.closed = TRUE AND p.quantity <= 0
            '''
            try:
                cursor.execute(query, (accountId, accountId))
            except:
                return []
        closedTrades = []
        try:
            for row in cursor.fetchall():
                strategy, agentId, tradeType, ticker, entryPrice, priceAtShort, quantity, exitPrice = row
                entryPrice = float(entryPrice) if entryPrice else 0.0
                priceAtShort = float(priceAtShort) if priceAtShort else 0.0
                quantity = float(quantity) if quantity else 0.0
                exitPrice = float(exitPrice) if exitPrice else 0.0
                if tradeType == 'long':
                    entry = entryPrice
                    exitVal = exitPrice
                    pnl = (exitVal - entry) * abs(quantity)
                    pnlPct = (exitVal - entry) / entry * 100 if entry > 0 else 0
                elif tradeType == 'short':
                    entry = priceAtShort
                    exitVal = exitPrice
                    pnl = (entry - exitVal) * abs(quantity)
                    pnlPct = (entry - exitVal) / entry * 100 if entry > 0 else 0
                else:
                    continue
                closedTrades.append({'strategy': strategy, 'agentId': agentId, 'pnl': pnl, 'pnlPercent': pnlPct, 'tradeType': tradeType, 'ticker': ticker, 'entryPrice': entry, 'exitPrice': exitVal, 'quantity': abs(quantity)})
        except Error as e:
            print(f'ERROR querying closed trades: {e}')
        finally:
            cursor.close()
        return closedTrades

    def checkAndAutoCloseShorts(self, accountId, currentDate):
        """
        Auto-close any short positions that have been open for 30+ days.
        
        Returns number of shorts auto-closed.
        """
        try:
            currentDateObj = datetime.strptime(currentDate, '%Y-%m-%d')
            cutoffDate = currentDateObj - timedelta(days=30)
            cutoffDateStr = cutoffDate.strftime('%Y-%m-%d')
        except ValueError:
            print(f'ERROR: Invalid date format {currentDate}. Expected YYYY-MM-DD.')
            return 0
        query = """
            SELECT ticker, mic, quantity, priceAtShort, strategyName, agentId, entryDate
            FROM portfolios
            WHERE accountId = %s AND tradeType = 'short' AND closed IS FALSE
            AND entryDate <= %s
        """
        cursor = self.connection.cursor()
        closedCount = 0
        try:
            cursor.execute(query, (accountId, cutoffDateStr))
            agedShorts = cursor.fetchall()
            for ticker, mic, quantity, priceAtShort, strategyName, _, entryDate in agedShorts:
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
                if quantity <= 0:
                    print(f'WARNING: Skipping auto-close for {ticker} due to non-positive quantity: {quantity}')
                    continue
                result = self.closeShort(accountId, ticker, mic, quantity, expiryDateStr, strategyName=strategyName, priceAtShort=priceAtShort, entryDate=entryDate)
                if result == 0:
                    closedCount += 1
                    print(f'Auto-closed aged short: {ticker} ({quantity} shares) at expiry date {expiryDateStr}')
        except Error as e:
            print(f'ERROR in checkAndAutoCloseShorts: {e}')
        finally:
            cursor.close()
        return closedCount

    def closeAllOpenPositions(self, accountId, currentDate, agentId=None):
        """
        Close ALL open positions (both longs and shorts) at current market price.
        Used at end of simulation to get final P&L and balance.
        
        agentId is optional: used for portfolio snapshot logging.
        
        Returns number of positions closed
        """
        closedCount = 0
        cursor = self.connection.cursor()
        try:
            queryLongs = '''
                SELECT ticker, mic, quantity
                FROM portfolios
                WHERE accountId = %s AND tradeType = 'long' AND (closed = FALSE OR closed IS NULL)
            '''
            cursor.execute(queryLongs, (accountId,))
            openLongs = cursor.fetchall()
            for ticker, mic, quantity in openLongs:
                try:
                    quantityFloat = float(quantity)
                    currentPrice = self.getPrice(ticker, mic, currentDate)
                    self.sellLong(accountId, ticker, mic, quantityFloat, currentDate)
                    closedCount += 1
                    print(f'Closed open long: {ticker} {quantityFloat} shares @ ${currentPrice:.2f}')
                except Exception as e:
                    print(f'Warning: Could not close long {ticker}: {e}')
            queryShorts = '''
                SELECT ticker, mic, quantity, strategyName, priceAtShort, entryDate
                FROM portfolios
                WHERE accountId = %s AND tradeType = 'short' AND (closed = FALSE OR closed IS NULL)
            '''
            cursor.execute(queryShorts, (accountId,))
            openShorts = cursor.fetchall()
            for ticker, mic, quantity, strategyName, priceAtShort, entryDate in openShorts:
                try:
                    quantityFloat = float(quantity)
                    priceAtShortFloat = float(priceAtShort)
                    currentPrice = self.getPrice(ticker, mic, currentDate)
                    self.closeShort(accountId, ticker, mic, quantityFloat, currentDate, strategyName=strategyName, priceAtShort=priceAtShortFloat, entryDate=entryDate)
                    closedCount += 1
                    print(f'Closed open short: {ticker} {quantityFloat} shares @ ${currentPrice:.2f}')
                except Exception as e:
                    print(f'Warning: Could not close short {ticker}: {e}')
        except Error as e:
            print(f'ERROR in closeAllOpenPositions: {e}')
        finally:
            cursor.close()
        if agentId:
            self.logPortfolioSnapshot(accountId, agentId, currentDate)
            print(f'Recorded final portfolio snapshot')
        return closedCount

    def recordTradePnL(self, strategyName, entryPrice, exitPrice, quantity, ticker, entryDate, exitDate, tradeType='long'):
        """
        Record P&L for a closed trade to the performance tracker.
        Called when a trade is manually closed (sellLong or closeShort).
        No need for account ID to be passed as the performance tracker for the respective agent is passed on init.
        
        strategyName: Name of the strategy that init this trade.
        entryDate: Entry date (YYYY-MM-DD) str.
        exitDate: Exit date (YYYY-MM-DD) str.
        tradeType: 'long' or 'short' to determine P&L formula.
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
            pnlPct = pnl / (entryPrice_float * quantity_float) * 100 if entryPrice_float > 0 else 0
            print(f'Recorded trade for {strategyName}: {ticker} ({tradeType}) P&L=${pnl:.2f} ({pnlPct:.2f}%) | Entry:${entryPrice_float:.2f} Exit:${exitPrice_float:.2f} Qty:{quantity_float:.0f}')
        except Exception as e:
            print(f'ERROR recording trade P&L: {e}')

    def getNewsForStock(self, ticker, simDate):
        """
        Retrieve news headlines for a stock on a given simulation date.

        This method is data access only. It returns raw DB rows and does not run
        model inference. Strategies are responsible for runtime sentiment scoring.

        NOTE: Only valid for Nasdaq simulations / US-based stocks due to dataset limitation. This is why no MIC is passed.
        
        Accounts for date misalignment in Kaggle dataset by querying headlines from simDate with simDate+1.

        simDate is a date string.
            
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
        logger = logging.getLogger(__name__)
        try:
            simDateObj = datetime.strptime(simDate, '%Y-%m-%d')
            nextDate = simDateObj + timedelta(days=1)
            nextDate_str = nextDate.strftime('%Y-%m-%d')
        except ValueError:
            print(f'ERROR: Invalid date format {simDate}. Expected YYYY-MM-DD.')
            return []
        query = 'SELECT id, headline, sentiment, sentiment_score, url, date, publisher FROM news_headlines WHERE ticker = %s AND date >= %s AND date <= %s ORDER BY date ASC'
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (ticker, simDate, nextDate_str))
            rows = cursor.fetchall()
            headlines = []
            for row in rows:
                headlineId, headlineText, sentiment, score, url, dateValue, publisher = row
                headlines.append({
                    'id': headlineId,
                    'headline': headlineText,
                    'sentiment': sentiment,
                    'score': float(score) if score is not None else None,
                    'url': url,
                    'date': str(dateValue),
                    'publisher': publisher,
                })
            if headlines:
                logger.info(f'[NewsDB] Retrieved {len(headlines)} headlines for {ticker} on {simDate}')
                for i, h in enumerate(headlines[:2]):
                    headlineSentiment = h.get('sentiment') if h.get('sentiment') else 'unknown'
                    headlineScore = h.get('score') if h.get('score') is not None else 0.0
                    logger.debug(f"  {i + 1}. [{headlineSentiment:8}] score={headlineScore:6.3f} | {h.get('headline', '')[:70]}")
            return headlines
        except Exception as e:
            logger.error(f'ERROR querying news for {ticker}: {e}')
            return []

    def cacheHeadlineSentiments(self, sentimentUpdates):
        """
        Cache strategy-inferred sentiment values to the news_headlines table.

        sentimentUpdates: List of dicts with keys:
            id, sentiment, score, confidence

        Returns number of headlines updated
        """
        if not sentimentUpdates:
            return 0
        query = 'UPDATE news_headlines SET sentiment = %s, sentiment_score = %s, confidence = %s WHERE id = %s'
        cursor = self.connection.cursor()
        updatedCount = 0
        try:
            for update in sentimentUpdates:
                headlineId = update.get('id')
                if headlineId is None:
                    continue
                sentiment = update.get('sentiment', 'neutral')
                score = float(update.get('score', 0.0))
                confidence = float(update.get('confidence', 0.0))
                cursor.execute(query, (sentiment, score, confidence, headlineId))
                updatedCount += 1
            self.connection.commit()
            return updatedCount
        except Exception as error:
            print(f'ERROR caching headline sentiment: {error}')
            return 0
        finally:
            cursor.close()

    def getMaxNewsDate(self):
        """
        Get the latest date available in the news_headlines dataset.
        Used for checking if simulation date has gone beyond available historical data.
        
        Returns latest date as YYYY-MM-DD string, or None if dataset is empty.
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

    def logPortfolioSnapshot(self, accountId, agentId, simDate):
        """
        Log current portfolio value snapshot to portfolio_history table.
        Call this once per decision period to track portfolio equity growth.
        Used for performance tracking display via chat GUI.
        
        simDate is a (YYYY-MM-DD) str.
        
        Returns true if successful, False if error
        """
        try:
            portfolioDf = self.checkPortfolio(accountId, simDate)
            cashBalance = self.checkBalance(accountId) or 0
            portfolioValue = 0.0
            if portfolioDf is not None and (not portfolioDf.empty):
                for _, row in portfolioDf.iterrows():
                    ticker = row.get('ticker')
                    mic = row.get('mic')
                    qty = float(row.get('quantity', 0))
                    trade_type = row.get('tradeType')
                    if ticker and qty > 0:
                        try:
                            current_price = self.getPrice(ticker, mic, simDate)
                            if trade_type == 'long':
                                portfolioValue += current_price * qty
                            elif trade_type == 'short':
                                portfolioValue -= current_price * qty
                        except ValueError:
                            pass
            total_value = float(cashBalance) + portfolioValue
            cursor = self.connection.cursor()
            query = 'INSERT INTO portfolio_history (accountId, snapshotDate, cashBalance, portfolioValue, totalValue, agentId) VALUES (%s, %s, %s, %s, %s, %s) ON DUPLICATE KEY UPDATE cashBalance = VALUES(cashBalance), portfolioValue = VALUES(portfolioValue), totalValue = VALUES(totalValue)'
            cursor.execute(query, (accountId, simDate, float(cashBalance), portfolioValue, total_value, agentId))
            self.connection.commit()
            return True
        except Exception as e:
            print(f'ERROR logging portfolio snapshot: {e}')
            return False

    def getPortfolioHistory(self, accountId, startDate=None, endDate=None):
        """
        Retrieve portfolio history snapshots for an account.
        
        startDate: Optional start date (YYYY-MM-DD) str.
        endDate: Optional end date.
        
        Returns pd dataFrame with columns: snapshotDate, cashBalance, portfolioValue, totalValue
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