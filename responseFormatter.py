"""
Response formatter for converting agent metrics and data into GUI-friendly text.
Formats agent performance, portfolio, trades, and strategy data for display.
"""
class ResponseFormatter:
    """Formats agent data into human-readable responses for the chat interface."""

    @staticmethod
    def formatPortfolioSummary(portfolio, balance):
        """
        Format portfolio holdings as markdown table with current prices and P&L.
        
        portfolio: Dict with held positions {ticker: {long/short, longEntryPrice, longCurrentPrice, etc}}
        balance: Current account balance
        Returns formatted portfolio summary string (markdown table for display).
        """
        if not portfolio:
            return f'Your portfolio is empty. Current balance: ${balance:,.2f}'
        totalHoldingsValue = 0.0
        for ticker, positions in portfolio.items():
            longQty = positions.get('long', 0)
            if longQty > 0:
                currentPrice = float(positions.get('longCurrentPrice', 0))
                totalHoldingsValue += currentPrice * float(longQty)
        totalStanding = float(balance) + totalHoldingsValue
        lines = []
        lines.append(f'Total Standing: ${totalStanding:,.2f}\n')
        lines.append(f'Cash: ${balance:,.2f}\n')
        lines.append(f'Holdings Value: ${totalHoldingsValue:,.2f}\n')
        lines.append('')
        lines.append('| Position | Qty | Entry Price | Current Price | P&L $ | P&L % |')
        lines.append('|---|---|---|---|---|---|')
        for ticker, positions in portfolio.items():
            longQty = positions.get('long', 0)
            shortQty = positions.get('short', 0)
            if longQty > 0:
                entryPrice = float(positions.get('longEntryPrice', 0))
                currentPrice = float(positions.get('longCurrentPrice', entryPrice))
                pnl = (currentPrice - entryPrice) * float(longQty)
                pnlPct = (currentPrice - entryPrice) / entryPrice * 100 if entryPrice != 0 else 0
                lines.append(f'| LONG {ticker} | {longQty} | ${entryPrice:.2f} | ${currentPrice:.2f} | ${pnl:,.2f} | {pnlPct:+.2f}% |')
            if shortQty > 0:
                entryPrice = float(positions.get('shortEntryPrice', 0))
                currentPrice = float(positions.get('shortCurrentPrice', entryPrice))
                pnl = (entryPrice - currentPrice) * float(shortQty)
                pnlPct = (entryPrice - currentPrice) / entryPrice * 100 if entryPrice != 0 else 0
                lines.append(f'| SHORT {ticker} | {shortQty} | ${entryPrice:.2f} | ${currentPrice:.2f} | ${pnl:,.2f} | {pnlPct:+.2f}% |')
        return '\n'.join(lines)

    @staticmethod
    def formatRecentActions(recommendations, limit=20):
        """
        Format recent strategy recommendations (including LONG, SHORT, HOLD, SELL).
        
        recommendations is the list of recommendation dicts from getAllRecommendations().
        limit is the max recommendations to display.

        Returns formatted recent actions (markdown table).
        """
        if not recommendations:
            return 'No recommendation history.'
        lines = []
        lines.append(f'Recent Strategy Recommendations (last {min(limit, len(recommendations))})')
        lines.append('')
        lines.append('| Date | Action | Ticker | Confidence | Strategy |')
        lines.append('|---|---|---|---|---|')
        for rec in recommendations[-limit:]:
            actionStr = rec.get('action', 'HOLD').upper()
            ticker = rec.get('ticker', '?')
            confidence = rec.get('confidence', 0) * 100
            strategy = rec.get('strategy', 'Unknown')
            date = rec.get('date', '?')
            lines.append(f'| {date} | {actionStr} | {ticker} | {confidence:.0f}% | {strategy} |')
        return '\n'.join(lines)

    @staticmethod
    def formatPerformanceChartData(performanceTracker, exchange, accountId):
        """
        Prepare performance chart data including cumulative P&L and portfolio equity curve.
        
        performanceTracker: PerformanceTracker instance.
        exchange: StockExchange instance for portfolio history.
        accountId: Account ID.
        
        Returns a dict with chart data for streamlit GUI display:
            {
                'cumulativePnL': {date: cumulativePnl},
                'portfolioValue': {date: totalValue},
                'strategyPnL': {strategyName: {date: pnl}},
                'allFalseIfNoData': bool
            }
        """
        data = {'cumulativePnL': {}, 'portfolioValue': {}, 'strategyPnL': {}, 'hasData': False}
        portfolioDf = exchange.getPortfolioHistory(accountId)
        if not portfolioDf.empty:
            data['hasData'] = True
            for _, row in portfolioDf.iterrows():
                dateStr = str(row['snapshotDate'])
                data['portfolioValue'][dateStr] = float(row['totalValue'])
        allStrategies = performanceTracker._tradeHistory.keys()
        for strategy in allStrategies:
            trades = performanceTracker._tradeHistory.get(strategy, [])
            strategyPnl = {}
            cumulative = 0.0
            sortedTrades = sorted(trades, key=lambda t: t.get('exitDate', ''))
            for trade in sortedTrades:
                exitDate = trade.get('exitDate', '')
                if exitDate:
                    cumulative += trade.get('pnl', 0)
                    strategyPnl[exitDate] = cumulative
            if strategyPnl:
                data['strategyPnL'][strategy] = strategyPnl
                data['hasData'] = True
        return data

    @staticmethod
    def formatStrategyComparison(performanceTracker, executionLog=None):
        """
        Format comprehensive strategy comparison with win rates and loss analysis.
        
        performanceTracker: PerformanceTracker instance.
        executionLog: List of executed trades for fallback strategy discovery.
        
        Returns a formatted markdown table with strategy metrics,
        """
        metrics = performanceTracker.getAllMetrics()
        if (not metrics or len(metrics) == 0) and executionLog:
            strategyTradeCounts = {}
            for trade in executionLog:
                strategy = trade.get('strategy', 'Unknown')
                if strategy != 'Unknown':
                    strategyTradeCounts[strategy] = strategyTradeCounts.get(strategy, 0) + 1
            allStrategies = set(strategyTradeCounts.keys())
            metrics = {strategy: {'totalTrades': strategyTradeCounts.get(strategy, 0), 'winCount': 0, 'lossCount': 0, 'winRate': 0, 'avgWin': 0, 'avgLoss': 0, 'profitFactor': 0, 'totalPnL': 0} for strategy in allStrategies}
        if not metrics or len(metrics) == 0:
            return 'No strategy performance data available yet. Run the simulation to generate trades.'
        lines = []
        lines.append('Strategy Performance Comparison')
        lines.append('')
        lines.append('| Strategy | Total Trades | Wins | Losses | Win Rate | Avg Win | Avg Loss | Profit Factor | Total P&L |')
        lines.append('|---|---|---|---|---|---|---|---|---|')
        for strategy in sorted(metrics.keys()):
            stats = metrics[strategy]
            totalCount = stats.get('totalTrades', 0)
            winCount = stats.get('winCount', 0)
            lossCount = stats.get('lossCount', 0)
            winRate = stats.get('winRate', 0) * 100 if 'winRate' in stats else winCount / totalCount * 100 if totalCount > 0 else 0
            avgWin = stats.get('avgWin', 0)
            avgLoss = stats.get('avgLoss', 0)
            profitFactor = stats.get('profitFactor', 0)
            totalPnl = stats.get('totalPnL', 0)
            lines.append(f'| {strategy} | {totalCount} | {winCount} | {lossCount} | {winRate:.1f}% | ${avgWin:,.2f} | ${avgLoss:,.2f} | {profitFactor:.2f} | ${totalPnl:,.2f} |')
        return '\n'.join(lines)

    @staticmethod
    def formatRecommendationQuality(recommendationMetrics):
        """
        Format strategy recommendation quality metrics.
        Shows all recommendations (including HOLD) and their outcomes.
        
        recommendationMetrics param is a dict mapping strategy names to:
            {totalRecommendations, holdCount, holdCorrect, holdMissedLong, holdMissedShort, 
                holdAccuracy, longCount, shortCount, totalExecuted, pendingCount}
        
        Returns a formatted recommendation quality report in natural language.
        Uses template filling NLG / aggregation.
        """
        if not recommendationMetrics:
            return 'No recommendation data available yet.'
        lines = ['## Analysis of HOLD recommendations by strategies:\n']
        for strategy, metrics in recommendationMetrics.items():
            totalRecs = metrics.get('totalRecommendations', 0)
            if totalRecs == 0:
                continue
            holdCount = metrics.get('holdCount', 0)
            holdCorrect = metrics.get('holdCorrect', 0)
            holdMissedLong = metrics.get('holdMissedLong', 0)
            holdMissedShort = metrics.get('holdMissedShort', 0)
            holdAccuracy = metrics.get('holdAccuracy', 0.0)
            longCount = metrics.get('longCount', 0)
            shortCount = metrics.get('shortCount', 0)
            sellCount = metrics.get('sellCount', 0)
            totalExecuted = metrics.get('totalExecuted', 0)
            pendingCount = metrics.get('pendingCount', 0)
            lines.append(f'**For the {strategy} approach:**\n')
            lines.append(f'It made a total of {totalRecs} recommended actions.\n')
            if holdCount > 0:
                holdPct = holdCount / totalRecs * 100
                lines.append(f'Of these, {holdCount} ({holdPct:.1f}%) were HOLD recommendations.\n')
                if holdCorrect:
                    lines.append(f'Holding was successful {holdCorrect} times ({holdAccuracy * 100:.1f}%) as the price was stable for a period following the recommendation.\n')
                if holdMissedLong:
                    lines.append(f'However, {holdMissedLong} times the price later increased, so a LONG would have been better.\n')
                if holdMissedShort:
                    lines.append(f'Additionally, {holdMissedShort} times the price later decreased, meaning a SHORT would have been more successful.\n')
                if pendingCount > 0:
                    lines.append(f'Note that {pendingCount} recommendations are yet to be evaluated as not enough time has passed.\n\n')
            if longCount > 0 or shortCount > 0 or sellCount > 0:
                actionPct = totalExecuted / totalRecs * 100
                lines.append(f'There were {totalExecuted} ({actionPct:.1f}%) LONG / SHORT / SELL recommendation(s).\n')
                if longCount > 0:
                    lines.append(f'LONG: {longCount}.\n')
                if shortCount > 0:
                    lines.append(f'SHORT: {shortCount}.\n')
                if sellCount > 0:
                    lines.append(f'SELL: {sellCount}.\n')
            lines.append('---\n')
        return '\n'.join(lines)
