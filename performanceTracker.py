"""
Performance Tracker for Trading Strategies.

Thread-safe tracking of strategy performance using profit factor.
Maintains rolling window metrics and all-time statistics.
"""
import threading
import logging
logger = logging.getLogger(__name__)

class PerformanceTracker:
    """
    Tracks strategy performance using profit factor (sum of profits / sum of losses).
    Thread-safe: uses RLock() to prevent read/write conflicts when GUI queries while agent records trades.
    """

    def __init__(self, windowSize=50):
        """
        Initialise performance tracker.
        windowSize is the number of recent trades to consider for sliding window metrics.
        """
        self._lock = threading.RLock()
        self.windowSize = windowSize
        self._tradeHistory = {}
        self._metricsCache = {}
        self._recommendationHistory = {}

    def recordTrade(self, strategyName, entryPrice, exitPrice, quantity, ticker, entryDate, exitDate, tradeType='long'):
        """
        Record a closed trade for a strategy.
        """
        with self._lock:
            if strategyName not in self._tradeHistory:
                self._tradeHistory[strategyName] = []
            if tradeType == 'short':
                pnl = (entryPrice - exitPrice) * quantity
            else:
                pnl = (exitPrice - entryPrice) * quantity
            tradeSignatureTuple = (ticker, quantity, round(entryPrice, 4), round(exitPrice, 4), round(pnl, 4), entryDate)
            recentTrades = self._tradeHistory[strategyName][-3:] if len(self._tradeHistory[strategyName]) > 0 else []
            for recentTrade in recentTrades:
                recentSignature = (recentTrade['ticker'], recentTrade['quantity'], round(recentTrade['entryPrice'], 4), round(recentTrade['exitPrice'], 4), round(recentTrade['pnl'], 4), recentTrade['entryDate'])
                if tradeSignatureTuple == recentSignature:
                    logger.warning(f'[{strategyName}] DUPLICATE TRADE REJECTED: {ticker} {quantity}@{entryPrice:.2f}->{exitPrice:.2f} on {entryDate} P&L ${pnl:.2f}')
                    return
            trade = {'ticker': ticker, 'entryPrice': entryPrice, 'exitPrice': exitPrice, 'quantity': quantity, 'pnl': pnl, 'entryDate': entryDate, 'exitDate': exitDate, 'returnPct': (exitPrice - entryPrice) / entryPrice * 100 if entryPrice != 0 else 0}
            self._tradeHistory[strategyName].append(trade)
            if strategyName in self._metricsCache:
                del self._metricsCache[strategyName]
            allTradesCount = len(self._tradeHistory[strategyName])
            allTradesPnl = sum((t['pnl'] for t in self._tradeHistory[strategyName]))
            tradeSignature = f'{ticker}_{quantity}_{entryPrice:.2f}_{exitPrice:.2f}_{pnl:.2f}'
            logger.info(f'[{strategyName}] Trade #{allTradesCount} [{tradeSignature}] recorded: {ticker} {quantity}@{entryPrice:.2f}->{exitPrice:.2f}, P&L ${pnl:.2f} | Total P&L all-time: ${allTradesPnl:.2f}')

    def getProfitFactor(self, strategyName):
        """
        Get profit factor for a strategy using sliding window.
        Profit Factor = sum(profits) / sum(losses)
        
        Special cases:
        - No trades: return 0.0
        - No losses (all profitable): return 100.0 (capped infinity)
        - No profits (all losing): return 0.0
        - Normal case: return sum(profits) / sum(losses)
        
        Args:
            strategyName: Name of the strategy
            
        Returns:
            Profit factor value (0.0 to 100.0+)
        """
        with self._lock:
            if strategyName not in self._tradeHistory or not self._tradeHistory[strategyName]:
                return 0.0
            recentTrades = self._tradeHistory[strategyName][-self.windowSize:]
            totalProfit = sum((t['pnl'] for t in recentTrades if t['pnl'] > 0))
            totalLoss = abs(sum((t['pnl'] for t in recentTrades if t['pnl'] < 0)))
            if totalLoss < 1e-06:
                if totalProfit > 1e-06:
                    return 100.0
                else:
                    return 0.0
            profitFactor = totalProfit / totalLoss
            return profitFactor

    def getMetrics(self, strategyName):
        """
        Get metrics for a strategy (cached, sliding window).
        Returns a dict with profitFactor, totalTrades, winCount, lossCount, avgWin, avgLoss, avgPnL, totalPnL
        """
        with self._lock:
            if strategyName not in self._tradeHistory or not self._tradeHistory[strategyName]:
                return {'profitFactor': 0.0, 'totalTrades': 0, 'winCount': 0, 'lossCount': 0, 'avgWin': 0.0, 'avgLoss': 0.0, 'avgPnL': 0.0, 'totalPnL': 0.0}
            allTrades = self._tradeHistory[strategyName]
            allTradesPnl = sum((t['pnl'] for t in allTrades))
            recentTrades = allTrades[-self.windowSize:]
            wins = [t for t in recentTrades if t['pnl'] > 0]
            losses = [t for t in recentTrades if t['pnl'] < 0]
            totalProfit = sum((t['pnl'] for t in wins)) if wins else 0
            totalLoss = sum((t['pnl'] for t in losses)) if losses else 0
            metrics = {'profitFactor': self.getProfitFactor(strategyName), 'totalTrades': len(recentTrades), 'winCount': len(wins), 'lossCount': len(losses), 'avgWin': totalProfit / len(wins) if wins else 0.0, 'avgLoss': abs(totalLoss / len(losses)) if losses else 0.0, 'avgPnL': sum((t['pnl'] for t in recentTrades)) / len(recentTrades), 'totalPnL': sum((t['pnl'] for t in recentTrades))}
            if metrics['totalPnL'] < 0.01 and allTradesPnl > 1.0:
                logger.warning(f'[{strategyName}] METRICS MISMATCH: windowPnL=${metrics['totalPnL']:.2f} vs allTimePnL=${allTradesPnl:.2f} ({len(allTrades)} total trades, window={self.windowSize})')
                logger.warning(f'  Window trades: {len(recentTrades)}, Wins={len(wins)} (${totalProfit:.2f}), Losses={len(losses)} (${abs(totalLoss):.2f})')
            return metrics

    def getAllMetrics(self):
        """
        Get metrics for all tracked strategies.
        Returns a dict mapping strategy names to their metrics (from getMetrics())
        """
        with self._lock:
            allMetrics = {}
            for strategyName in self._tradeHistory.keys():
                allMetrics[strategyName] = self.getMetrics(strategyName)
            return allMetrics

    def recordRecommendation(self, strategyName, action, ticker, priceAtRec, confidence, timestamp, mic):
        """
        Record a strategy recommendation so it's quality can be checked later.
        
        action is the recommended action, one of ('long', 'short', 'hold', 'sell')
        priceAtRec is the price at time of recommendation.
        confidence is between 0.0-1.0
        timestamp is a (YYYY-MM-DD) str.
        """
        with self._lock:
            if strategyName not in self._recommendationHistory:
                self._recommendationHistory[strategyName] = []
            recommendation = {'action': action, 'ticker': ticker, 'priceAtRec': priceAtRec, 'confidence': confidence, 'timestamp': timestamp, 'mic': mic, 'outcome': 'PENDING'}
            self._recommendationHistory[strategyName].append(recommendation)

    def scoreRecommendations(self, exchange, simDate, mic, thresholdPct=2.0):
        """
        Score HOLD recommendations against actual price movement.
        Evaluates whether HOLD recommendations were correct based on price change seen.
        
        exchange: StockExchange instance for price lookups
        simDate as a date str.
        thresholdPct is the price movement threshold (%) to classify recommendation outcome.
        """
        with self._lock:
            for strategyName, recommendations in self._recommendationHistory.items():
                for rec in recommendations:
                    if rec['outcome'] != 'PENDING' or rec['action'] != 'hold':
                        continue
                    if rec['timestamp'] >= simDate:
                        continue
                    ticker = rec['ticker']
                    recPrice = rec['priceAtRec']
                    try:
                        currentPrice = exchange.getPrice(ticker, rec['mic'], simDate)
                        priceDelta = (currentPrice - recPrice) / recPrice * 100
                        if priceDelta > thresholdPct:
                            rec['outcome'] = 'MISSED_LONG'
                        elif priceDelta < -thresholdPct:
                            rec['outcome'] = 'MISSED_SHORT'
                        else:
                            rec['outcome'] = 'CORRECT'
                        rec['currentPrice'] = currentPrice
                        rec['priceDelta'] = priceDelta
                    except (ValueError, Exception) as e:
                        rec['outcome'] = 'PENDING'
                        logger.debug(f'Could not score {ticker} recommendation for {strategyName}: {e}')

    def getScoredRecommendations(self, strategyName, simDate=None, actions=None):
        """
        Get scored recommendations for a strategy based on recommendation history.

        strategyName: strategy name to filter by.
        simDate is optional: YYYY-MM-DD str upper bound (inclusive) for recommendation date.
        actions is optional: iterable of actions to include, e.g. ['hold'].

        Returns a list of recommendation dicts with scored outcomes.
        """
        with self._lock:
            recs = self._recommendationHistory.get(strategyName, [])
            allowedActions = set(actions) if actions else None
            scored = []
            for rec in recs:
                if rec.get('outcome') not in ('CORRECT', 'MISSED_LONG', 'MISSED_SHORT'):
                    continue
                if allowedActions and rec.get('action') not in allowedActions:
                    continue
                if simDate and rec.get('timestamp') and rec.get('timestamp') > simDate:
                    continue
                scored.append(rec)
            return scored

    def getAllRecommendations(self, limit=20):
        """
        Get recent recommendations across all strategies.

        limit is the maximum number of recommendations to return (most recent first).

        Returns a list of recommendation dicts with unified keys for GUI formatting.
        """
        with self._lock:
            allRecs = []
            for strategyName, recs in self._recommendationHistory.items():
                for rec in recs:
                    allRecs.append({
                        'strategy': strategyName,
                        'action': rec.get('action', 'hold'),
                        'ticker': rec.get('ticker', ''),
                        'confidence': rec.get('confidence', 0.0),
                        'date': rec.get('timestamp', ''),
                        'outcome': rec.get('outcome', 'PENDING')
                    })
            allRecs.sort(key=lambda r: r.get('date', ''), reverse=True)
            return allRecs[:limit] if limit is not None else allRecs

    def getRecommendationMetrics(self, strategyName):
        """
        Get metrics for all recommendations from a strategy.
        Returns:
            a dict with:
            - totalRecommendations: Total recommendations made
            - holdCount: Number of HOLD recommendations
            - holdCorrect: Number of correct HOLD recommendations
            - holdMissedLong: Number of HOLD recs that should have been LONG
            - holdMissedShort: Number of HOLD recs that should have been SHORT
            - holdAccuracy: Accuracy % for HOLD recommendations
            - longCount: Number of LONG recommendations
            - shortCount: Number of SHORT recommendations
            - totalExecuted: Total LONG + SHORT recommendations
            - pendingCount: Pending recommendations not yet scored
        """
        with self._lock:
            if strategyName not in self._recommendationHistory:
                return {'totalRecommendations': 0, 'holdCount': 0, 'holdCorrect': 0, 'holdMissedLong': 0, 'holdMissedShort': 0, 'holdAccuracy': 0.0, 'longCount': 0, 'shortCount': 0, 'sellCount': 0, 'totalExecuted': 0, 'pendingCount': 0}
            recs = self._recommendationHistory[strategyName]
            longCount = sum((1 for r in recs if r['action'] == 'long'))
            shortCount = sum((1 for r in recs if r['action'] == 'short'))
            sellCount = sum((1 for r in recs if r['action'] == 'sell'))
            holdRecs = [r for r in recs if r['action'] == 'hold']
            holdCorrect = sum((1 for r in holdRecs if r['outcome'] == 'CORRECT'))
            holdMissedLong = sum((1 for r in holdRecs if r['outcome'] == 'MISSED_LONG'))
            holdMissedShort = sum((1 for r in holdRecs if r['outcome'] == 'MISSED_SHORT'))
            pendingCount = sum((1 for r in holdRecs if r['outcome'] == 'PENDING'))
            scoredHolds = len(holdRecs) - pendingCount
            holdAccuracy = holdCorrect / scoredHolds if scoredHolds > 0 else 0.0
            return {'totalRecommendations': len(recs), 'holdCount': len(holdRecs), 'holdCorrect': holdCorrect, 'holdMissedLong': holdMissedLong, 'holdMissedShort': holdMissedShort, 'holdAccuracy': holdAccuracy, 'longCount': longCount, 'shortCount': shortCount, 'sellCount': sellCount, 'totalExecuted': longCount + shortCount + sellCount, 'pendingCount': pendingCount}