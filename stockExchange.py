from typing import Any, Dict, List
import yfinance as yf
import mysql.connector
from mysql.connector import Error
import pandas as pd
import os
from datetime import datetime, timedelta
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

    def __init__(self, performanceTracker=None):
        '''
        Initialises DB.
        Creates stockExchange table.
        Creates accounts table.
        Creates portfolio table.
        Creates trades table.
        
        Args:
            performanceTracker: Optional PerformanceTracker instance for strategy performance recording
        '''
        self.connection = self.__createServerConnection()
        self.performanceTracker = performanceTracker  # For close-trade callbacks
        
        query = "CREATE DATABASE IF NOT EXISTS StockExchange;"
        self.__executeDatabaseQuery(query)

        self.connection.database = 'StockExchange'

        query = ("CREATE TABLE IF NOT EXISTS trades ("
                 "accountId INT, "
                 "ticker VARCHAR(255), "
                 "mic ENUM('XNAS','XLON','XHKG','XJPX'), "
                 "tradeType ENUM('long','short'), "
                 "quantity DECIMAL(10,4), "
                 "price FLOAT, "
                 "timestamp DATETIME, "
                 "strategyName VARCHAR(255), "
                 "agentId INT)")
        self.__executeDatabaseQuery(query)

        query = ("CREATE TABLE IF NOT EXISTS portfolios ("
                 "accountId INT, "
                 "ticker VARCHAR(255), "
                 "mic ENUM('XNAS','XLON','XHKG','XJPX'), "
                 "tradeType ENUM('long','short'), "
                 "quantity DECIMAL(10,4), "
                 "closed BOOL, "
                 "priceAtShort FLOAT, "
                 "entryPrice FLOAT, "
                 "strategyName VARCHAR(255), "
                 "agentId INT, "
                 "entryDate DATE, "
                 "UNIQUE KEY (accountId, ticker, tradeType))")
        self.__executeDatabaseQuery(query)

        query = ("CREATE TABLE IF NOT EXISTS accounts ("
                 "accountId INT PRIMARY KEY, "
                 "balance DECIMAL(32,2) DEFAULT 10000.00)")
        self.__executeDatabaseQuery(query)

        query = ("CREATE TABLE IF NOT EXISTS news_headlines ("
                 "id INT PRIMARY KEY, "
                 "ticker VARCHAR(255), "
                 "headline TEXT, "
                 "url TEXT, "
                 "publisher VARCHAR(255), "
                 "date DATE, "
                 "sentiment VARCHAR(20), "
                 "sentiment_score FLOAT, "
                 "confidence FLOAT, "
                 "INDEX idx_ticker_date (ticker, date))")
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

    def getStockData(self, ticker: str, period: str = None, start: str = None, end: str = None) -> Any:
        '''
        Returns a dataframe for the provided ticker.
        If period is provided, returns data for that period.
        If start and end are provided, returns data between those dates.

        Valid periods include: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max.
        '''

        stock = yf.Ticker(ticker)
        if start or end:
            return stock.history(start=start, end=end)
        return stock.history(period=period)

    def getCurrentPrice(self, ticker: str, mic: str, simDate: str = None) -> float:
        '''
        Get current or historical price for a ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            mic: Market Identifier Code (e.g., 'XNAS', 'XLON')
            simDate: Optional date for historical price (YYYY-MM-DD). If None, uses latest.
        
        Returns:
            Price as float. Returns 0.0 if unable to fetch.
        '''
        try:
            micTicker = self.__getMicTicker(ticker, mic)
            if simDate:
                data = self.getStockData(micTicker, end=simDate)
            else:
                data = self.getStockData(micTicker, period='1d')
            
            if data is not None and len(data) > 0:
                return float(data['Close'].iloc[-1])
            else:
                print(f"ERROR: No price data for {ticker} on {simDate}")
                return 0.0
        except Exception as e:
            print(f"ERROR: Unable to fetch price for {ticker}: {e}")
            return 0.0

    def placeShort(self, ticker, mic, quantity, accountId, simDate, strategyName=None, agentId=None):
        '''
        Short (with automated borrowing) - sell a stock not owned via borrowing the stock. 
        Automatically borrows the stock at current price, then immediately sells it.
        Once 14 days pass, the current price is used to purchase a new stock and repay the broker. 
        Idea is that if the price falls, the shorter profits. 

        simDate allows the agent to act on historical data. 
        The agent can act on historical data to enable repeatability of experiments.

        No balance check performed as the stock is not being 'bought'.
        Returns -1 if accountId does not exist.
        
        Args:
            strategyName: Optional name of strategy initiating this trade (for performance tracking)
            agentId: Optional agent ID for tracking
        '''
        # Add current price of shorted stock to account balance.
        price = self.getStockData(self.__getMicTicker(ticker,mic), end=simDate)['Close'].iloc[-1]
        cost = float(price * quantity)
        query = "UPDATE accounts SET balance = balance + %s WHERE accountId = %s"
        rowsAffected = self.__executeDatabaseQuery(query,(cost,accountId))

        # Place the short trade and add to portfolio.
        if rowsAffected:
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'short',%s,%s,%s,%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, closed, priceAtShort, strategyName, agentId, entryDate) VALUES (%s,%s,%s,'short',%s, FALSE, %s,%s,%s,%s) " \
                    "ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f"ERROR: Account with ID {accountId} does not exist.")
            return -1
        
    def placeLong(self, ticker, mic, quantity, accountId, simDate, strategyName=None, agentId=None):
        '''
        Long - purchase stock outright.
        Stock is bought for current market price. Stock is added to portfolio.

        The balance of account with passed ID will be checked: returns -1 if not enough balance / accountId does not exist.
        
        Args:
            strategyName: Optional name of strategy initiating this trade (for performance tracking)
            agentId: Optional agent ID for tracking
        '''
        # Fetch current price as of close.
        price = self.getStockData(self.__getMicTicker(ticker,mic), end=simDate)['Close'].iloc[-1]
        # Calculate total.
        cost = float(price * quantity)
        # Check against balance and deduct.
        query = "UPDATE accounts SET balance = balance - %s WHERE accountId = %s AND balance >= %s;"
        rowsAffected = self.__executeDatabaseQuery(query,(cost,accountId,cost))
        # Check if the balance check passed and the balance was deducted
        if rowsAffected:
            print("Balance deducted, trade is executing...")
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp, strategyName, agentId) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate, strategyName, agentId))
            # If ticker for accountId already in portfolios: update the quantity rather than add a new record.
            # Insert new record otherwise.
            query = "INSERT INTO portfolios (accountId, ticker, mic, tradeType, quantity, entryPrice, strategyName, agentId, entryDate) VALUES (%s,%s,%s,'long',%s,%s,%s,%s,%s) " \
            "ON DUPLICATE KEY UPDATE quantity = quantity + VALUES(quantity);"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, strategyName, agentId, simDate))
        else:
            print(f"ERROR: Balance for account {accountId} too low for trade of cost {cost}")
            return -1
        
    def checkBalance(self, accountId):
        '''
        Returns the balance as a float for the account linked to the passed accountId.
        Gracefully handles exceptions - returns None if failed for any reason.
        '''
        query = "SELECT balance FROM accounts WHERE accountId = %s;"
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (accountId,))
            result = cursor.fetchone()
            if result:
                return result[0]
            else:
                print(f"ERROR: No account found with accountId {accountId}")
                return None
        except Error as e:
            print(f"ERROR: {e}")
            return None

    def checkPortfolio(self, accountId):
        '''
        Returns a Pandas dataframe of all of the currently held positions.
        Includes held assets as well as open shorts. 
        Gracefully handles exceptions - returns None if failed for any reason.
        '''
        # TODO: check if open shorts are 14 days old or more. if 14 days old (or older), close based on price at 14 days old.
        query = "SELECT * FROM portfolios WHERE accountId = %s"
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (accountId,))
            # List of tuples of format [(id,ticker,mic,type,quantity,closed,priceAtShort),...]
            rows = cursor.fetchall()
            # Convert to Pandas dataframe, easier to manipulate for agent.
            columns = [i[0] for i in cursor.description]
            result = pd.DataFrame(rows, columns=columns)
        except Error as e:
            print(f"ERROR: {e}")
            return None

        return result

    def sellLong(self, accountId, ticker, mic, quantity, simDate):
        '''
        Sell a held portfolio asset at current market price.
        Atomic operation - will only succeed if account has sufficient quantity.
        If quantity becomes 0, removes the position record to keep DB clean.
        Returns 0 on success, -1 on failure.
        '''
        # Get current price.
        price = self.getStockData(self.__getMicTicker(ticker, mic), end=simDate)['Close'].iloc[-1]
        proceeds = float(price * quantity)
        
        # Atomically: Reduce quantity only if we have enough
        query = "UPDATE portfolios SET quantity = quantity - %s " \
                "WHERE accountId = %s AND ticker = %s AND tradeType = 'long' " \
                "AND quantity >= %s"
        rowsAffected = self.__executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        
        if rowsAffected > 0:
            # Credit the account with proceeds
            query = "UPDATE accounts SET balance = balance + %s WHERE accountId = %s"
            self.__executeDatabaseQuery(query, (proceeds, accountId))
            
            # Log the trade
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp) " \
                    "VALUES (%s, %s, %s, 'sell', %s, %s, %s)"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate))
            
            # Clean up zero-quantity positions
            query = "DELETE FROM portfolios " \
                    "WHERE accountId = %s AND ticker = %s AND tradeType = 'long' AND quantity <= 0"
            self.__executeDatabaseQuery(query, (accountId, ticker))
            
            print(f"Sold {quantity} of {ticker} at {price}. Proceeds: {proceeds}")
            return 0
        else:
            print(f"ERROR: Insufficient quantity of {ticker} to sell.")
            return -1


    def closeShort(self, accountId, ticker, mic, quantity, simDate):
        '''
        Settle a short early.
        Must purchase the stock at the current price to return it to the broker.
        If the price is lower than the priceAtShort, the difference is profit.
        If the price is higher, then the difference is a loss.
        Uses closed flag to prevent race conditions - won't close an already-closed position.
        Returns 0 if succeeded, -1 if failed.
        '''
        # Get current price
        price = self.getStockData(self.__getMicTicker(ticker, mic), end=simDate)['Close'].iloc[-1]
        cost_to_buyback = float(price * quantity)
        
        # Atomically: Check if open (closed IS FALSE), deduct quantity, and set closed=TRUE
        # This prevents race conditions where the same short is closed twice
        query = "UPDATE portfolios SET quantity = quantity - %s, closed = TRUE " \
                "WHERE accountId = %s AND ticker = %s AND tradeType = 'short' " \
                "AND quantity >= %s AND closed IS FALSE"
        rowsAffected = self.__executeDatabaseQuery(query, (quantity, accountId, ticker, quantity))
        
        if rowsAffected > 0:
            # Deduct cost from balance
            query = "UPDATE accounts SET balance = balance - %s WHERE accountId = %s"
            self.__executeDatabaseQuery(query, (cost_to_buyback, accountId))
            
            # Log the trade
            query = "INSERT INTO trades (accountId, ticker, mic, tradeType, quantity, price, timestamp) " \
                    "VALUES (%s, %s, %s, 'short', %s, %s, %s)"
            self.__executeDatabaseQuery(query, (accountId, ticker, mic, quantity, price, simDate))
            
            print(f"Short for {ticker} closed at {price}. Cost to buy back: {cost_to_buyback}")
            return 0
        else:
            print(f"ERROR: No open short found for {ticker} with sufficient quantity or already closed.")
            return -1
    
    def cleanupClosedPositions(self, accountId):
        '''
        Remove fully closed positions (quantity <= 0 and closed = TRUE) from portfolios.
        Prevents stale data from accumulating in the DB.
        Returns the number of deleted records.
        '''
        query = "DELETE FROM portfolios WHERE accountId = %s AND closed IS TRUE AND quantity <= 0"
        rowsDeleted = self.__executeDatabaseQuery(query, (accountId,))
        if rowsDeleted > 0:
            print(f"Cleaned up {rowsDeleted} closed position(s).")
        return rowsDeleted

    # =========================================================================
    # CLOSED-TRADE DETECTION & PERFORMANCE TRACKING
    # =========================================================================

    def getClosedTradesPnL(self, accountId: int, strategyName: str = None) -> List[Dict[str, Any]]:
        '''
        Retrieve all closed trades with P&L calculations for performance tracking.
        
        For LONG trades: P&L = (exitPrice - entryPrice) * quantity
        For SHORT trades: P&L = (entryPrice - exitPrice) * quantity
        
        Args:
            accountId: Account to query
            strategyName: Optional filter for specific strategy
        
        Returns:
            List of dicts: [{strategy, agentId, pnl, pnlPercent, tradeType, ticker, entryPrice, exitPrice}]
        '''
        # Get all closed trades with both entry and exit prices
        if strategyName:
            query = """
                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker, 
                       p.entryPrice, p.priceAtShort, p.quantity,
                       (SELECT MAX(price) FROM trades 
                        WHERE accountId = %s AND ticker = p.ticker 
                        AND timestamp > p.entryDate) as exitPrice
                FROM portfolios p
                WHERE p.accountId = %s AND p.closed IS TRUE AND p.strategyName = %s
                AND p.quantity <= 0
            """
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, accountId, strategyName))
            except:
                return []
        else:
            query = """
                SELECT p.strategyName, p.agentId, p.tradeType, p.ticker,
                       p.entryPrice, p.priceAtShort, p.quantity,
                       (SELECT MAX(price) FROM trades
                        WHERE accountId = %s AND ticker = p.ticker
                        AND timestamp > p.entryDate) as exitPrice
                FROM portfolios p
                WHERE p.accountId = %s AND p.closed IS TRUE AND p.quantity <= 0
            """
            cursor = self.connection.cursor()
            try:
                cursor.execute(query, (accountId, accountId))
            except:
                return []
        
        closed_trades = []
        try:
            for row in cursor.fetchall():
                strategy, agentId, tradeType, ticker, entryPrice, priceAtShort, quantity, exitPrice = row
                
                if tradeType == 'long':
                    entry = entryPrice or 0
                    exit_val = exitPrice or 0
                    pnl = (exit_val - entry) * abs(quantity)
                    pnl_pct = ((exit_val - entry) / entry * 100) if entry > 0 else 0
                elif tradeType == 'short':
                    entry = priceAtShort or 0
                    exit_val = exitPrice or 0
                    pnl = (entry - exit_val) * abs(quantity)
                    pnl_pct = ((entry - exit_val) / entry * 100) if entry > 0 else 0
                else:
                    continue
                
                closed_trades.append({
                    'strategy': strategy,
                    'agentId': agentId,
                    'pnl': pnl,
                    'pnlPercent': pnl_pct,
                    'tradeType': tradeType,
                    'ticker': ticker,
                    'entryPrice': entry,
                    'exitPrice': exit_val,
                    'quantity': abs(quantity)
                })
        except Error as e:
            print(f"ERROR querying closed trades: {e}")
        finally:
            cursor.close()
        
        return closed_trades

    def checkAndAutoCloseShorts(self, accountId: int, currentDate: str) -> int:
        '''
        Auto-close any short positions that have been open for 14+ days.
        As per the domain model: shorts are held for max 14 days, then auto-repurchased.
        
        Args:
            accountId: Account to process
            currentDate: Current simulation date (YYYY-MM-DD)
        
        Returns:
            Number of shorts auto-closed
        '''
        try:
            current_date_obj = datetime.strptime(currentDate, '%Y-%m-%d')
            cutoff_date = current_date_obj - timedelta(days=14)
            cutoff_date_str = cutoff_date.strftime('%Y-%m-%d')
        except ValueError:
            print(f"ERROR: Invalid date format {currentDate}. Expected YYYY-MM-DD.")
            return 0
        
        # Find all open shorts older than 14 days
        query = """
            SELECT ticker, mic, quantity, priceAtShort, strategyName, agentId, entryDate
            FROM portfolios
            WHERE accountId = %s AND tradeType = 'short' AND closed IS FALSE
            AND entryDate <= %s
        """
        cursor = self.connection.cursor()
        closed_count = 0
        
        try:
            cursor.execute(query, (accountId, cutoff_date_str))
            aged_shorts = cursor.fetchall()
            
            for ticker, mic, quantity, priceAtShort, strategyName, agentId, entryDate in aged_shorts:
                # Get current price to calculate exit price
                current_price = self.getStockData(
                    self.__getMicTicker(ticker, mic), 
                    end=currentDate
                )['Close'].iloc[-1]
                
                # Auto-close by calling closeShort
                result = self.closeShort(accountId, ticker, mic, quantity, currentDate)
                if result == 0:
                    closed_count += 1
                    print(f"Auto-closed aged short: {ticker} ({quantity} shares)")
                    
                    # Calculate and record P&L for performance tracking
                    if self.performanceTracker and strategyName:
                        entryDateStr = entryDate if isinstance(entryDate, str) else entryDate.strftime('%Y-%m-%d')
                        self.recordTradePnL(
                            accountId,
                            strategyName,
                            entryPrice=priceAtShort,
                            exitPrice=current_price,
                            quantity=quantity,
                            ticker=ticker,
                            entryDate=entryDateStr,
                            exitDate=currentDate
                        )
        except Error as e:
            print(f"ERROR in checkAndAutoCloseShorts: {e}")
        finally:
            cursor.close()
        
        return closed_count

    def recordTradePnL(self, accountId: int, strategyName: str, 
                      entryPrice: float, exitPrice: float, 
                      quantity: float, ticker: str,
                      entryDate: str, exitDate: str) -> None:
        '''
        Record P&L for a closed trade to the performance tracker.
        Called when a trade is manually closed (sellLong or closeShort).
        
        Args:
            accountId: Account ID
            strategyName: Name of the strategy that initiated this trade
            entryPrice: Price at which position was opened
            exitPrice: Price at which position was closed
            quantity: Number of shares
            ticker: Stock ticker
            entryDate: Entry date (YYYY-MM-DD)
            exitDate: Exit date (YYYY-MM-DD)
        '''
        if not self.performanceTracker or not strategyName:
            return
        
        try:
            # Record trade to performance tracker
            self.performanceTracker.recordTrade(
                strategyName,
                entryPrice=entryPrice,
                exitPrice=exitPrice,
                quantity=int(quantity),
                ticker=ticker,
                entryDate=entryDate,
                exitDate=exitDate
            )
            
            pnl = (exitPrice - entryPrice) * quantity
            pnl_pct = (pnl / (entryPrice * quantity)) * 100 if entryPrice > 0 else 0
            print(f"Recorded trade for {strategyName}: {ticker} P&L=${pnl:.2f} ({pnl_pct:.2f}%)")
        
        except Exception as e:
            print(f"ERROR recording trade P&L: {e}")

    def getNewsForStock(self, ticker: str, mic: str, simDate: str) -> List[Dict[str, Any]]:

        '''
        Retrieve raw news headlines for a stock on a given simulation date (data layer).
        
        Returns raw headline data without aggregation or intelligence processing.
        Sentiment analysis is the responsibility of the agent/consumer.
        
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
                    'score': float,         # Normalized score [-1.0, +1.0]
                    'url': str,             # Article URL
                    'date': str,            # Publication date (YYYY-MM-DD)
                    'publisher': str        # News source
                },
                ...
            ]
            Returns empty list [] if no headlines found.
        '''
        from datetime import datetime, timedelta
        
        try:
            sim_date_obj = datetime.strptime(simDate, '%Y-%m-%d')
            next_date = sim_date_obj + timedelta(days=1)
            next_date_str = next_date.strftime('%Y-%m-%d')
        except ValueError:
            print(f"ERROR: Invalid date format {simDate}. Expected YYYY-MM-DD.")
            return []
        
        query = "SELECT headline, sentiment, sentiment_score, url, date, publisher " \
                "FROM news_headlines " \
                "WHERE ticker = %s AND date >= %s AND date <= %s " \
                "ORDER BY date ASC"
        
        cursor = self.connection.cursor()
        try:
            cursor.execute(query, (ticker, simDate, next_date_str))
            rows = cursor.fetchall()
            
            headlines = []
            for row in rows:
                headline, sentiment, score, url, date, publisher = row
                headlines.append({
                    'headline': headline,
                    'sentiment': sentiment,
                    'score': float(score),
                    'url': url,
                    'date': str(date),
                    'publisher': publisher
                })
            
            return headlines
        
        except Exception as e:
            print(f"ERROR querying news for {ticker}: {e}")
            return []










    
